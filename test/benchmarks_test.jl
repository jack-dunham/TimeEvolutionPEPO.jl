@testsetup module BenchmarkSetup
using Reexport
@reexport using TimeEvolutionPEPO
@reexport using TensorKit
@reexport using LinearAlgebra
@reexport using Logging
@reexport using Random

export quiet, dense_liouvillian, exact_derivative, exact_derivative_stripe, onesite, derivative

const SEED = 1234

quiet(f) = with_logger(f, ConsoleLogger(stderr, Logging.Warn))

"""Dense Lindblad generator on `vec(rho)`, built independently of the package. Column-major."""
function dense_liouvillian(H::AbstractMatrix, jumps::AbstractVector)
    d = size(H, 1)
    Id = Matrix{ComplexF64}(I, d, d)
    G = -im * (kron(Id, H) - kron(transpose(H), Id))
    for L in jumps
        LdL = L' * L
        G += kron(conj(L), L) - (kron(Id, LdL) + kron(transpose(LdL), Id)) / 2
    end
    return G
end

"""Exact `d⟨O⟩/dt` at `t = 0` on a uniform product state: four `ZZ` bonds plus one local generator."""
function exact_derivative(O::AbstractMatrix, rho::AbstractMatrix, J::Real, jump::AbstractMatrix)
    Id = Matrix{ComplexF64}(I, 2, 2)
    Gloc = dense_liouvillian(zeros(ComplexF64, 2, 2), [jump])
    dOdt = real(tr(O * reshape(Gloc * vec(rho), 2, 2)))

    Gbond = dense_liouvillian((J / 4) * kron(Z_PAULI, Z_PAULI), Matrix{ComplexF64}[])
    drho2 = reshape(Gbond * vec(kron(rho, rho)), 4, 4)
    dOdt += 4 * real(tr(kron(O, Id) * drho2))

    return dOdt
end

const Z_PAULI = ComplexF64[1 0; 0 -1]

"""
Exact `d⟨O⟩/dt` on site `A` of a three-period stripe `A, B, C`, uniform along the other axis. `A` is the
first factor of `h2` on two of its four bonds and the second factor on the other two, with `B` on one
side and `C` on the other; a uniform or checkerboard state cannot distinguish the two orderings, since
exchanging the factors of every bond term is a lattice inversion.
"""
function exact_derivative_stripe(
        O::AbstractMatrix, rhoA::AbstractMatrix, rhoB::AbstractMatrix, rhoC::AbstractMatrix,
        h2::AbstractMatrix, jump::AbstractMatrix
    )
    Id = Matrix{ComplexF64}(I, 2, 2)
    Gloc = dense_liouvillian(zeros(ComplexF64, 2, 2), [jump])
    dOdt = real(tr(O * reshape(Gloc * vec(rhoA), 2, 2)))

    Gbond = dense_liouvillian(h2, Matrix{ComplexF64}[])
    bond(r1, r2) = reshape(Gbond * vec(kron(r1, r2)), 4, 4)

    dOdt += real(tr(kron(O, Id) * bond(rhoA, rhoA))) + real(tr(kron(Id, O) * bond(rhoA, rhoA)))
    dOdt += real(tr(kron(O, Id) * bond(rhoA, rhoB)))
    dOdt += real(tr(kron(Id, O) * bond(rhoC, rhoA)))

    return dOdt
end

function onesite(pepo, ind = (1, 1); bonddim = 8)
    dm = quiet(() -> DensityMatrix(pepo, bonddim))
    return partialtrace(dm, ind)
end

"""Finite-difference derivative of `observables` over one step, Richardson-extrapolated in `dt`."""
function derivative(state0, model, observables, dts; D_env = 8)
    values = map(dts) do dt
        Random.seed!(SEED)
        sim = quiet(() -> Simulation(deepcopy(state0), model; timestep = dt))
        before = [real(expval(onesite(quantumstate(sim); bonddim = D_env), O)) for O in observables]
        quiet(() -> simulate!(sim; numsteps = 1, verbose = false))
        after = [real(expval(onesite(quantumstate(sim); bonddim = D_env), O)) for O in observables]
        return (after .- before) ./ dt
    end
    return 2 .* values[2] .- values[1]   # cancels the O(dt) term for halved steps
end
end # BenchmarkSetup

@testitem "B2: interacting Lindblad derivative" setup = [BenchmarkSetup] begin
    X, Y, Z = PAULI
    jump = 0.5 * (Y - im * Z)
    J = 1.5
    v0 = (0.36, 0.48, -0.8)     # unit length, so the state is pure and exactly representable

    model = J / 4 * LocalOp(Z, Z) + Dissipator(jump)
    got = derivative(productpepo(fill(v0, 2, 2), 8), model, (X, Z), (0.004, 0.002))
    want = [exact_derivative(O, bloch(v0), J, jump) for O in (X, Z)]
    flipped = [exact_derivative(O, bloch(v0), -J, jump) for O in (X, Z)]

    @test maximum(abs.(got .- want)) < 1e-3
    # The reversed Hamiltonian sign must be excluded, not merely disfavoured.
    @test maximum(abs.(got .- flipped)) > 1
end

@testitem "B4: high-temperature limit of thermal annealing" setup = [BenchmarkSetup] begin
    X, Y, Z = PAULI
    J, h, dt = 1.5, 1.0, 0.005

    # rho(beta) = exp(-beta H): d<ZZ>/dbeta = -J/4 and d<X>/dbeta = h at beta = 0.
    model = J / 4 * LocalOp(Z, Z) - h * LocalOp(X)
    sim = quiet(
        () -> Simulation(productpepo(fill(ThermalState(), 2, 2), 4), model; timestep = dt)
    )

    rows = NamedTuple[]
    callback = function (s)
        t = simulationinfo(s).simtime
        t == 0 && return nothing
        dm = quiet(() -> DensityMatrix(quantumstate(s), 8))
        zz = real(expval(partialtrace(dm, (1, 1), (1, 2)), Z, Z))
        x = real(expval(partialtrace(dm, (1, 1)), X))
        push!(rows, (t = t, dzz = zz / t, dx = x / t))
        return nothing
    end
    quiet(() -> simulate!(callback, sim; numsteps = 4, verbose = false))

    last_row = last(rows)
    @test last_row.dzz ≈ -J / 4 atol = 5e-2
    @test last_row.dx ≈ h atol = 5e-2
    # The measured slopes also fix beta = t rather than beta = 2t.
    @test abs(last_row.dzz / (-J / 4) - 1) < 5e-2
end

@testitem "B7: bond dimension above the rank the propagator generates" setup = [BenchmarkSetup] begin
    X, Y, Z = PAULI
    jump = 0.5 * (Y - im * Z)
    dt, numsteps = 0.02, 5
    v0 = (0.0, 0.0, -1.0)

    # The old code inverted the exact zeros in padded bond weights and produced garbage beyond D = 2.
    # PEPSKit guards the same inversion with `sdiag_pow(s, pow; tol)`.
    G = dense_liouvillian(zeros(ComplexF64, 2, 2), [jump])
    model = zero(Hamiltonian) + Dissipator(jump)

    for D in (1, 2, 4, 8)
        Random.seed!(1234)
        sim = quiet(() -> Simulation(productpepo(fill(v0, 2, 2), D), model; timestep = dt))
        deviation = 0.0
        callback = function (s)
            t = simulationinfo(s).simtime
            rdm = onesite(quantumstate(s); bonddim = 8)
            exact = reshape(exp(t * G) * vec(bloch(v0)), 2, 2)
            deviation = max(
                deviation,
                abs(real(expval(rdm, X)) - real(tr(exact * X))),
                abs(real(expval(rdm, Z)) - real(tr(exact * Z))),
            )
            return nothing
        end
        quiet(() -> simulate!(callback, sim; numsteps, verbose = false))
        @test deviation < 1e-8
    end
end

@testitem "B9: site ordering of an asymmetric two-site term" setup = [BenchmarkSetup] begin
    X, Y, Z = PAULI
    jump = 0.5 * (Y - im * Z)
    a = 1.0

    # Three-period stripe: the two neighbours along an axis differ, which a uniform or checkerboard
    # state cannot achieve, so this is the only configuration that constrains the site order.
    vA, vB, vC = (0.36, 0.48, -0.8), (-0.8, 0.0, 0.6), (0.0, -1.0, 0.0)
    rhoA, rhoB, rhoC = bloch(vA), bloch(vB), bloch(vC)
    init = [vA vB vC; vA vB vC]

    model = a * LocalOp(X, Y) + Dissipator(jump)
    got = derivative(productpepo(init, 8), model, (X, Y, Z), (0.004, 0.002))

    reference(h2) = [
        exact_derivative_stripe(O, rhoA, rhoB, rhoC, h2, jump) for O in (X, Y, Z)
    ]
    want = reference(a * kron(X, Y))
    exchanged = reference(a * kron(Y, X))

    @test maximum(abs.(got .- want)) < 1e-3
    @test maximum(abs.(got .- exchanged)) > 1
end
