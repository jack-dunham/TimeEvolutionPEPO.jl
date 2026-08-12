# Exact benchmarks for task 0.1 of PEPSKIT_REFACTOR_PLAN.md.
#
# Each benchmark compares the package against a reference computed independently with dense linear
# algebra, so a disagreement indicates an error in the package rather than a change in it.
#
#   julia> include("test/exact_benchmarks.jl")
#   julia> run_benchmarks()          # all four, writes test/reference/benchmarks.txt
#   julia> b1()                      # one at a time
#
# Vectorisation here is column-major (`vec`, `kron`), independent of the package's row-major
# ket-before-bra convention; only basis-independent quantities are compared.

using TimeEvolutionPEPO
using TensorKit
using LinearAlgebra
using Random
using Logging
using Printf

const BENCHMARK_DIR = joinpath(@__DIR__, "reference")
const SEED = 1234

quiet(f) = with_logger(f, ConsoleLogger(stderr, Logging.Warn))

"""
Dense Lindblad generator acting on `vec(rho)`, from `dρ/dt = -i[H,ρ] + Σ(LρL' - {L'L,ρ}/2)`.
"""
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

evolve_dense(rho::AbstractMatrix, G::AbstractMatrix, t::Real) =
    reshape(exp(t * G) * vec(rho), size(rho))

bloch(v) = (one(ComplexF64) * I(2) + v[1] * PAULI[1] + v[2] * PAULI[2] + v[3] * PAULI[3]) / 2

onesite_rdm(pepo, alg) = partialtrace(quiet(() -> DensityMatrix(pepo, alg)), (1, 1))

function twosite_rdm(pepo, alg, offset)
    dm = quiet(() -> DensityMatrix(pepo, alg))
    return partialtrace(dm, (1, 1), (1 + offset[1], 1 + offset[2]))
end

"""
B1: purely one-local Lindblad dynamics. Every site obeys the single-qubit master equation, the state
stays a product state, and `D = 1` is exact, so the package must reproduce `exp(t𝓛)` at every step.
"""
function b1(; dt = 0.02, numsteps = 20)
    X, Y, Z = PAULI
    jump = 0.5 * (Y - im * Z)
    rho0 = bloch((0.0, 0.0, -1.0))
    G = dense_liouvillian(zeros(ComplexF64, 2, 2), [jump])

    # Zero coupling keeps the layer type `Liouvillian` while removing the interaction.
    model = Model(0 * LocalOp(Z, Z) + Dissipator(jump))
    alg = CTMRG(; bonddim = 4, tol = 1.0e-10, maxiter = 100, verbose = false)

    Random.seed!(SEED)
    sim = quiet(() -> Simulation(PEPO(fill((0, 0, -1), 2, 2), 2, 1), model, TEBD(); timestep = dt))

    rows = NamedTuple[]
    callback = function (s)
        info = simulationinfo(s)
        rdm = onesite_rdm(quantumstate(s), alg)
        exact = evolve_dense(rho0, G, info.simtime)
        push!(
            rows,
            (
                step = info.iterations, t = info.simtime,
                x = real(expval(rdm, X)), z = real(expval(rdm, Z)),
                x_exact = real(tr(exact * X)), z_exact = real(tr(exact * Z)),
                trace = real(tr(exact)),
            ),
        )
        return nothing
    end
    quiet(() -> simulate!(callback, sim; numsteps, maxshots = numsteps, verbose = false))

    dx = maximum(abs(r.x - r.x_exact) for r in rows)
    dz = maximum(abs(r.z - r.z_exact) for r in rows)
    return (
        name = "B1", description = "one-local Lindblad, D=1, exact single-site exponential",
        deviation = max(dx, dz), tolerance = 1.0e-8, rows = rows,
    )
end

"""
Exact `d⟨O_i⟩/dt` at `t = 0` for a product state, from the physical generator: nearest-neighbour
`ZZ` bonds plus one one-local generator per site. Independent of how the package distributes local
terms over Trotter layers.
"""
function exact_derivative(O::AbstractMatrix, rho::AbstractMatrix, J::Real, jump::AbstractMatrix)
    X, Y, Z = PAULI
    Id = Matrix{ComplexF64}(I, 2, 2)

    Gloc = dense_liouvillian(zeros(ComplexF64, 2, 2), [jump])
    dOdt = real(tr(O * reshape(Gloc * vec(rho), 2, 2)))

    Hbond = (J / 4) * kron(Z, Z)
    Gbond = dense_liouvillian(Hbond, Matrix{ComplexF64}[])
    rho2 = kron(rho, rho)
    drho2 = reshape(Gbond * vec(rho2), 4, 4)
    OI = kron(O, Id)
    dOdt += 4 * real(tr(OI * drho2))   # four nearest neighbours on the square lattice

    return dOdt
end

"""
B2: interacting Lindblad model. The finite-difference derivative of ⟨X⟩ and ⟨Z⟩ over one step must
converge to the exact derivative linearly in `dt`; a nonzero limit indicates a wrong prefactor.
"""
function b2(; dts = (0.004, 0.002, 0.001), J = 1.5, D = 8)
    X, Y, Z = PAULI
    jump = 0.5 * (Y - im * Z)
    # Unit-length Bloch vector: `pepositedata` prepares rho^4, which equals rho only for pure states.
    v0 = (0.36, 0.48, -0.8)
    rho0 = bloch(v0)
    model = Model(J / 4 * LocalOp(Z, Z) + Dissipator(jump))
    alg = CTMRG(; bonddim = 8, tol = 1.0e-10, maxiter = 100, verbose = false)

    exact = (x = exact_derivative(X, rho0, J, jump), z = exact_derivative(Z, rho0, J, jump))
    # Same reference with the Hamiltonian sign reversed, to identify which convention is in use.
    flipped = (x = exact_derivative(X, rho0, -J, jump), z = exact_derivative(Z, rho0, -J, jump))

    rows = NamedTuple[]
    for dt in dts
        Random.seed!(SEED)
        sim = quiet(
            () -> Simulation(PEPO(fill(v0, 2, 2), 2, D), model, TEBD(); timestep = dt),
        )
        rdm0 = onesite_rdm(quantumstate(sim), alg)
        x0, z0 = real(expval(rdm0, X)), real(expval(rdm0, Z))
        quiet(() -> simulate!(sim; numsteps = 1, verbose = false))
        rdm1 = onesite_rdm(quantumstate(sim), alg)
        x1, z1 = real(expval(rdm1, X)), real(expval(rdm1, Z))
        push!(
            rows,
            (
                dt = dt, dx = (x1 - x0) / dt, dz = (z1 - z0) / dt,
                err_x = abs((x1 - x0) / dt - exact.x), err_z = abs((z1 - z0) / dt - exact.z),
            ),
        )
    end

    # Richardson extrapolation to dt -> 0 over the two smallest steps.
    r1, r2 = rows[end - 1], rows[end]
    dx, dz = 2 * r2.dx - r1.dx, 2 * r2.dz - r1.dz
    return (
        name = "B2", description = "interacting Lindblad, derivative vs exact generator",
        deviation = max(abs(dx - exact.x), abs(dz - exact.z)), tolerance = 1.0e-3,
        deviation_flipped_hamiltonian = max(abs(dx - flipped.x), abs(dz - flipped.z)),
        extrapolated = (x = dx, z = dz), exact = exact, flipped = flipped, rows = rows,
    )
end

"""
B5: the product-state constructor must prepare the requested density matrix. `pepositedata` chains
four copies of the input through the physical index, so it prepares `rho^4` instead.
"""
function b5()
    X, Y, Z = PAULI
    alg = CTMRG(; bonddim = 4, tol = 1.0e-10, maxiter = 100, verbose = false)
    rho4(r) = (r^4) / tr(r^4)

    rows = NamedTuple[]
    for v in ((0.0, 0.0, -0.5), (0.4, 0.5, -0.6), (0.2, -0.3, 0.1))
        rho = bloch(v)
        rdm = onesite_rdm(PEPO(fill(rho, 2, 2), 2, 1), alg)
        got = [real(expval(rdm, O)) for O in (X, Y, Z)]
        push!(
            rows,
            (
                bloch = v,
                deviation_vs_rho = maximum(abs.(got .- [real(tr(rho * O)) for O in (X, Y, Z)])),
                deviation_vs_rho4 = maximum(
                    abs.(got .- [real(tr(rho4(rho) * O)) for O in (X, Y, Z)])
                ),
            ),
        )
    end

    return (
        name = "B5", description = "product-state constructor prepares the requested rho",
        deviation = maximum(r.deviation_vs_rho for r in rows), tolerance = 1.0e-8,
        deviation_if_rho4_intended = maximum(r.deviation_vs_rho4 for r in rows), rows = rows,
    )
end

"""
B6: Larmor precession under `H = (h/2) Z` from a pure state along `+x`, with no dissipation and
`D = 1`, so the result is exact. Distinguishes `dρ/dt = -i[H,ρ]` from `+i[H,ρ]`.
"""
function b6(; h = 2.0, dt = 0.05, numsteps = 6)
    X, Y, Z = PAULI
    alg = CTMRG(; bonddim = 4, tol = 1.0e-10, maxiter = 100, verbose = false)
    rho0 = bloch((1.0, 0.0, 0.0))
    G = dense_liouvillian((h / 2) * Z, Matrix{ComplexF64}[])
    model = Model(h / 2 * LocalOp(Z) + Dissipator(0.0 * Z))

    Random.seed!(SEED)
    sim = quiet(
        () -> Simulation(PEPO(fill((1.0, 0.0, 0.0), 2, 2), 2, 1), model, TEBD(); timestep = dt),
    )
    rows = NamedTuple[]
    callback = function (s)
        info = simulationinfo(s)
        rdm = onesite_rdm(quantumstate(s), alg)
        exact = evolve_dense(rho0, G, info.simtime)
        push!(
            rows,
            (
                t = info.simtime,
                y = real(expval(rdm, Y)), y_exact = real(tr(exact * Y)),
                x = real(expval(rdm, X)), x_exact = real(tr(exact * X)),
            ),
        )
        return nothing
    end
    quiet(() -> simulate!(callback, sim; numsteps, maxshots = numsteps, verbose = false))

    return (
        name = "B6", description = "Larmor precession, sign of the Hamiltonian in a Liouvillian",
        deviation = maximum(max(abs(r.y - r.y_exact), abs(r.x - r.x_exact)) for r in rows),
        tolerance = 1.0e-8,
        deviation_flipped_hamiltonian = maximum(
            max(abs(r.y + r.y_exact), abs(r.x - r.x_exact)) for r in rows
        ),
        rows = rows,
    )
end

"""
B7: bond dimension in excess of the rank the Trotter gate can generate. The padded bond weights
contain exact zeros, which simple update inverts. `D = 1, 2` are exact for a one-local model.
"""
function b7(; dt = 0.02, numsteps = 5)
    X, Y, Z = PAULI
    jump = 0.5 * (Y - im * Z)
    alg = CTMRG(; bonddim = 4, tol = 1.0e-10, maxiter = 100, verbose = false)
    rho0 = bloch((0.0, 0.0, -1.0))
    G = dense_liouvillian(zeros(ComplexF64, 2, 2), [jump])
    model = Model(0 * LocalOp(Z, Z) + Dissipator(jump))

    rows = NamedTuple[]
    for D in (1, 2, 4, 8)
        Random.seed!(SEED)
        dev = try
            sim = quiet(
                () -> Simulation(
                    PEPO(fill((0, 0, -1), 2, 2), 2, D), model, TEBD(); timestep = dt
                ),
            )
            worst = 0.0
            callback = function (s)
                info = simulationinfo(s)
                rdm = onesite_rdm(quantumstate(s), alg)
                exact = evolve_dense(rho0, G, info.simtime)
                worst = max(
                    worst,
                    abs(real(expval(rdm, X)) - real(tr(exact * X))),
                    abs(real(expval(rdm, Z)) - real(tr(exact * Z))),
                )
                return nothing
            end
            quiet(() -> simulate!(callback, sim; numsteps, maxshots = numsteps, verbose = false))
            worst
        catch e
            NaN
        end
        push!(rows, (D = D, deviation = dev))
    end

    return (
        name = "B7", description = "one-local model at bond dimension above the gate rank",
        deviation = maximum(r.deviation for r in rows), tolerance = 1.0e-8, rows = rows,
    )
end

"""
B3: invariants that hold for any correct Lindblad trajectory — unit trace, a Hermitian one-site
reduced density matrix, and no negative eigenvalues beyond numerical noise.
"""
function b3(; dt = 0.025, numsteps = 20, J = 1.5, D = 4)
    X, Y, Z = PAULI
    jump = 0.5 * (Y - im * Z)
    model = Model(J / 4 * LocalOp(Z, Z) + Dissipator(jump))
    alg = CTMRG(; bonddim = 8, tol = 1.0e-10, maxiter = 100, verbose = false)

    Random.seed!(SEED)
    sim = quiet(
        () -> Simulation(PEPO(fill((0, 0, -1), 2, 2), 2, D), model, TEBD(); timestep = dt),
    )

    rows = NamedTuple[]
    callback = function (s)
        info = simulationinfo(s)
        rdm = onesite_rdm(quantumstate(s), alg)
        m = convert(Array, rdm)
        m = reshape(m, 2, 2)
        evals = eigvals(Hermitian((m + m') / 2))
        push!(
            rows,
            (
                step = info.iterations, t = info.simtime,
                trace_error = abs(real(tr(m)) - 1),
                hermiticity = norm(m - m'),
                min_eigenvalue = minimum(real, evals),
            ),
        )
        return nothing
    end
    quiet(() -> simulate!(callback, sim; numsteps, maxshots = numsteps, verbose = false))

    return (
        name = "B3", description = "trace, hermiticity and positivity of the one-site RDM",
        deviation = maximum(
            max(r.trace_error, r.hermiticity, max(0.0, -r.min_eigenvalue)) for r in rows
        ),
        tolerance = 1.0e-6, rows = rows,
    )
end

"""
B4: high-temperature limit of thermal annealing. At `β → 0` from the maximally mixed state,
`d⟨Z_iZ_j⟩/dβ = -J/4` and `d⟨X⟩/dβ = h` exactly. The measured ratio also fixes the relation between
the package's simulated time and `β`.
"""
function b4(; dt = 0.005, numsteps = 4, J = 1.5, h = 1.0, D = 4)
    X, Y, Z = PAULI
    model = Model(J / 4 * LocalOp(Z, Z) - h * LocalOp(X))
    alg = CTMRG(; bonddim = 8, tol = 1.0e-10, maxiter = 100, verbose = false)

    Random.seed!(SEED)
    sim = quiet(
        () -> Simulation(
            PEPO(fill(ThermalState(), 2, 2), 2, D), model, TEBD(); timestep = dt
        ),
    )

    rows = NamedTuple[]
    callback = function (s)
        info = simulationinfo(s)
        info.iterations == 0 && return nothing
        pepo = quantumstate(s)
        rdm1 = onesite_rdm(pepo, alg)
        rdm2 = twosite_rdm(pepo, alg, (1, 0))
        t = info.simtime
        push!(
            rows,
            (
                step = info.iterations, t = t,
                dzz = real(expval(rdm2, Z, Z)) / t, dzz_exact = -J / 4,
                dx = real(expval(rdm1, X)) / t, dx_exact = h,
            ),
        )
        return nothing
    end
    quiet(() -> simulate!(callback, sim; numsteps, maxshots = numsteps, verbose = false))

    last_row = rows[end]
    return (
        name = "B4", description = "high-temperature limit of thermal annealing",
        deviation = max(
            abs(last_row.dzz - last_row.dzz_exact), abs(last_row.dx - last_row.dx_exact)
        ),
        tolerance = 5.0e-2,
        ratio_zz = last_row.dzz / last_row.dzz_exact, ratio_x = last_row.dx / last_row.dx_exact,
        rows = rows,
    )
end

"""
B8: the `Ising` fast path against the generic path. At zero coupling the Ising gate reduces to a pure
local exponential (`cosh(0) = 1`, `sinh(0) = 0`), so the two paths build the same operator and must
agree exactly; the generic path's absolute correctness comes from B1 and B6. The interacting rows are
diagnostics: the two paths split the Trotter step differently, so a difference shrinking with `dt` is
expected, while a `dt`-independent difference is not.
"""
function b8(; h = 1.0, a = 1.5, dt = 0.05, numsteps = 6)
    X, Y, Z = PAULI
    alg = CTMRG(; bonddim = 4, tol = 1.0e-10, maxiter = 100, verbose = false)
    v0 = (1.0, 0.0, 0.0)

    function trajectory(model, Dbond, τ, nsteps)
        Random.seed!(SEED)
        sim = quiet(
            () -> Simulation(PEPO(fill(v0, 2, 2), 2, Dbond), model, TEBD(); timestep = τ),
        )
        out = NamedTuple[]
        callback = function (s)
            rdm = onesite_rdm(quantumstate(s), alg)
            push!(
                out,
                (
                    t = simulationinfo(s).simtime,
                    x = real(expval(rdm, X)), y = real(expval(rdm, Y)),
                    z = real(expval(rdm, Z)),
                ),
            )
            return nothing
        end
        quiet(() -> simulate!(callback, sim; numsteps = nsteps, maxshots = nsteps, verbose = false))
        return out
    end

    discrepancy(r1, r2) = maximum(
        max(abs(p.x - q.x), abs(p.y - q.y), abs(p.z - q.z)) for (p, q) in zip(r1, r2)
    )

    nojump = Dissipator(0.0 * Z)
    free_ising = Model(Ising(0.0 * LocalOp(X, X), [h * LocalOp(Z)], nojump))
    free_generic = Model(0.0 * LocalOp(X, X) + h * LocalOp(Z) + nojump)
    deviation = discrepancy(
        trajectory(free_ising, 1, dt, numsteps), trajectory(free_generic, 1, dt, numsteps)
    )

    jump = 0.5 * (Y - im * Z)
    rows = NamedTuple[]
    for τ in (dt, dt / 2)
        steps = round(Int, numsteps * dt / τ)
        d = discrepancy(
            trajectory(Model(Ising(a * LocalOp(X, X), [h * LocalOp(Z)], Dissipator(jump))), 4, τ, steps),
            trajectory(Model(a * LocalOp(X, X) + h * LocalOp(Z) + Dissipator(jump)), 4, τ, steps),
        )
        push!(rows, (dt = τ, steps = steps, interacting_discrepancy = d))
    end

    return (
        name = "B8", description = "Ising fast path against the generic path at zero coupling",
        deviation = deviation, tolerance = 1.0e-8, rows = rows,
    )
end

function run_benchmarks(dir::AbstractString = BENCHMARK_DIR)
    mkpath(dir)
    results = []
    for f in (b1, b2, b3, b4, b5, b6, b7, b8)
        try
            push!(results, f())
        catch e
            push!(
                results,
                (
                    name = string(f), description = "failed to run",
                    deviation = NaN, tolerance = 0.0,
                    error = sprint(showerror, e), rows = NamedTuple[],
                ),
            )
            @error "benchmark $(f) failed" exception = e
        end
    end

    path = joinpath(dir, "benchmarks.txt")
    open(path, "w") do io
        println(io, "# exact benchmarks, commit ", readchomp(`git rev-parse --short HEAD`))
        println(io, "# working tree: ", replace(readchomp(`git status --short`), "\n" => "; "))
        for r in results
            status = isnan(r.deviation) ? "ERROR" :
                (r.deviation <= r.tolerance ? "AGREES" : "DISAGREES")
            @printf(
                io, "\n%s  %s\n  %s\n  deviation %.3e  tolerance %.1e\n",
                r.name, status, r.description, r.deviation, r.tolerance
            )
            haskey(r, :error) && println(io, "  error: ", r.error)
            haskey(r, :ratio_zz) &&
                @printf(io, "  measured/exact: zz %.6f  x %.6f\n", r.ratio_zz, r.ratio_x)
            haskey(r, :deviation_flipped_hamiltonian) && @printf(
                io, "  deviation with the Hamiltonian sign reversed: %.3e\n",
                r.deviation_flipped_hamiltonian
            )
            haskey(r, :deviation_if_rho4_intended) && @printf(
                io, "  deviation against rho^4: %.3e\n", r.deviation_if_rho4_intended
            )
            for row in r.rows
                println(io, "    ", row)
            end
        end
    end

    for r in results
        status = isnan(r.deviation) ? "ERROR" :
            (r.deviation <= r.tolerance ? "AGREES" : "DISAGREES")
        @info "$(r.name): $status (deviation $(r.deviation), tolerance $(r.tolerance))"
    end

    @info "wrote $path"
    return results
end
