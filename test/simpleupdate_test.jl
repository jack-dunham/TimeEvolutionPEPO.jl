@testsetup module SimpleUpdateSetup
using Reexport
@reexport using TimeEvolutionPEPO
@reexport using TensorKit
@reexport using LinearAlgebra
@reexport using PEPSKit
@reexport using MatrixAlgebraKit: truncrank

export dense_liouvillian, evolve_dense, onesite, evolve

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

evolve_dense(rho, G, t) = reshape(exp(t * G) * vec(rho), size(rho))

"""One-site reduced density matrix as a plain matrix."""
function onesite(pepo, ind = (1, 1); bonddim = 4)
    dm = DensityMatrix(pepo, bonddim)
    return reshape(convert(Array, partialtrace(dm, ind)), 2, 2)
end

"""A few steps of an interacting Lindblad model, run at damping parameter `zeta`."""
function evolve(; zeta = Inf, D = 4, numsteps = 5)
    X, Y, Z = PAULI
    sim = Simulation(
        productpepo(fill((0, 0, -1), 2, 2), 1),
        1.5 / 4 * LocalOp(Z, Z) + Dissipator(0.5 * (Y - im * Z)); timestep = 0.05
    )
    simulate!(sim; numsteps, verbose = false, trunc = truncrank(D), zeta)
    return sim
end
end # SimpleUpdateSetup

@testitem "Simple update sweep" setup = [SimpleUpdateSetup] begin
    X, Y, Z = PAULI
    P = ComplexSpace(2)

    @testset "one-local Lindblad dynamics is exact at D = 1" begin
        # No interaction, so the state stays a product state and D = 1 truncates nothing. Every bond
        # generator commutes with every other, so there is no Trotter error either: agreement with
        # exp(t L) should be at machine precision.
        jump = 0.5 * (Y - im * Z)
        v0 = (0.0, 0.0, -1.0)
        dt, numsteps = 0.02, 10

        # `trotterise` halves the one-local part, since a full step sweeps both axes.
        model = zero(Hamiltonian) + Dissipator(jump)
        decomposition = trotterise(model, P, dt)

        state = productpepo(fill(v0, 2, 2), 1)
        weights = SUWeight(state)
        G = dense_liouvillian(zeros(ComplexF64, 2, 2), [jump])

        deviation = 0.0
        for step in 1:numsteps
            timestep!(state, weights, decomposition)
            m = onesite(state)
            exact = evolve_dense(bloch(v0), G, step * dt)
            deviation = max(
                deviation,
                abs(real(tr(m * X)) - real(tr(exact * X))),
                abs(real(tr(m * Z)) - real(tr(exact * Z))),
            )
        end
        @test deviation < 1e-8
    end

    @testset "trace and hermiticity survive an interacting sweep" begin
        jump = 0.5 * (Y - im * Z)
        model = 1.5 / 4 * LocalOp(Z, Z) + Dissipator(jump)
        decomposition = trotterise(model, P, 0.025)

        state = productpepo(fill((0, 0, -1), 2, 2), 4)
        weights = SUWeight(state)

        for _ in 1:5
            timestep!(state, weights, decomposition)
        end
        m = onesite(state)
        @test abs(real(tr(m)) - 1) < 1e-8
        @test norm(m - m') < 1e-8
        @test minimum(real, eigvals(Hermitian((m + m') / 2))) > -1e-8
    end

    @testset "a unit cell smaller than 2x2 is rejected" begin
        u = bondpropagator(zero(Hamiltonian) + Dissipator(0.0 * Z), P, 0.01)
        state = productpepo((0, 0, -1), 1)
        @test_throws ArgumentError axissweep!(state, SUWeight(state), u, 1)
    end

    @testset "a propagator matrix is indexed per bond" begin
        state = productpepo(fill((0, 0, -1), 2, 2), 2)
        weights = SUWeight(state)

        # A 1x1 matrix must behave exactly as the bare tensor did.
        prop = first(first(trotterise(LocalOp(Z, Z), P, 0.05)))
        @test size(prop) == (1, 1)
        ϵ = axissweep!(state, weights, prop, 1)
        @test isfinite(ϵ)
    end
end

@testitem "Damping matrix" setup = [SimpleUpdateSetup] begin
    @testset "Ξ(ζ) damps by rank and normalises to unit trace" begin
        s = DiagonalTensorMap([0.5, 1.0, 0.25, 0.125], ComplexSpace(4))
        damped = dampweight(s, 1.0)
        # ranks are 2nd, 1st, 3rd, 4th, so the factors are exp(-1), 1, exp(-2), exp(-3)
        raw = [0.5exp(-1), 1.0, 0.25exp(-2), 0.125exp(-3)]
        @test damped.data ≈ raw ./ sum(raw)
        @test sum(damped.data) ≈ 1
    end

    @testset "the limiting values of ζ" begin
        s = DiagonalTensorMap([1.0, 0.5, 0.25], ComplexSpace(3))
        @test dampweight(s, Inf) === s
        @test dampweight(s, 0).data ≈ [1.0, 0.0, 0.0]
        @test_throws ArgumentError dampweight(s, -1.0)
    end

    @testset "a finite ζ suppresses the bond spectrum during evolution" begin
        # The damping is applied at every bond update, so its effect compounds over the sweep.
        tail(sim) = sum(sum(x.data) - maximum(x.data) for x in collect(bondweights(sim)))

        undamped = evolve(; zeta = Inf)
        damped = evolve(; zeta = 1.0)

        @test tail(damped) < tail(undamped)
        @test !isapprox(onesite(quantumstate(damped)), onesite(quantumstate(undamped)); atol = 1.0e-6)
    end

    @testset "ζ = Inf reproduces the undamped sweep exactly" begin
        a = evolve(; zeta = Inf)
        b = evolve()
        for (x, y) in zip(collect(bondweights(a)), collect(bondweights(b)))
            @test x.data == y.data
        end
    end
end
