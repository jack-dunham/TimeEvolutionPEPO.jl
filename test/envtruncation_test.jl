@testsetup module EnvTruncationSetup
using Reexport
@reexport using TimeEvolutionPEPO
@reexport using TensorKit
@reexport using LinearAlgebra
@reexport using Logging

export quiet, dense_liouvillian, evolve_dense, onesite, lindblad

quiet(f) = with_logger(f, ConsoleLogger(stderr, Logging.Error))

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

function onesite(state; bonddim = 8)
    dm = quiet(() -> DensityMatrix(state, bonddim))
    return reshape(convert(Array, partialtrace(dm, (1, 1))), 2, 2)
end

lindblad() = (X = PAULI[1]; Y = PAULI[2]; Z = PAULI[3];
    1.5 / 4 * LocalOp(Z, Z) + Dissipator(0.5 * (Y - im * Z)))
end # EnvTruncationSetup

@testitem "Environment-based truncation" setup = [EnvTruncationSetup] begin
    X, Y, Z = PAULI
    P = ComplexSpace(2)

    @testset "exact when nothing must be truncated" begin
        # One-local model at D = 1: the propagator has rank one across the bond, so the environment
        # optimisation has nothing to discard and must reproduce exp(t L) like the bare SVD does.
        jump = 0.5 * (Y - im * Z)
        v0 = (0.0, 0.0, -1.0)
        dt, nsteps = 0.02, 5
        G = dense_liouvillian(zeros(ComplexF64, 2, 2), [jump])

        state = productpepo(fill(v0, 2, 2), 1)
        weights = SUWeight(state)
        decomposition = trotterise(zero(Hamiltonian) + Dissipator(jump), P, dt)
        alg = FullEnvTruncation(; trunc = truncspace(ComplexSpace(1)), maxiter = 20)

        envalg = BondEnvironment(4)
        deviation = 0.0
        for step in 1:nsteps
            quiet(() -> timestep!(state, weights, decomposition; truncalg = alg, envalg))
            m = onesite(state; bonddim = 4)
            exact = evolve_dense(bloch(v0), G, step * dt)
            deviation = max(
                deviation,
                abs(real(tr(m * X)) - real(tr(exact * X))),
                abs(real(tr(m * Z)) - real(tr(exact * Z))),
            )
        end
        @test deviation < 1e-8
    end

    @testset "full environment truncation against a bare SVD" begin
        # Both truncate to the same bond dimension from the same state; the environment-optimised
        # result must be at least as faithful, since the SVD ignores the surroundings.
        dt, D = 0.05, 2
        decomposition = trotterise(lindblad(), P, dt)

        reference = quiet(
            () -> begin
                sim = Simulation(productpepo(fill((0, 0, -1), 2, 2), 4), lindblad(); timestep = dt)
                simulate!(sim; numsteps = 6, verbose = false)
                sim
            end,
        )
        target = onesite(quantumstate(reference))

        svd_state = productpepo(fill((0, 0, -1), 2, 2), D)
        svd_weights = SUWeight(svd_state)
        fet_state = productpepo(fill((0, 0, -1), 2, 2), D)
        fet_weights = SUWeight(fet_state)
        alg = FullEnvTruncation(; trunc = truncspace(ComplexSpace(D)), maxiter = 30)

        envalg = BondEnvironment(8)
        for _ in 1:6
            quiet(() -> timestep!(svd_state, svd_weights, decomposition))
            quiet(() -> timestep!(fet_state, fet_weights, decomposition; truncalg = alg, envalg))
        end

        svd_error = norm(onesite(svd_state) - target)
        fet_error = norm(onesite(fet_state) - target)
        @test isfinite(fet_error)
        @test fet_error <= svd_error * 1.05
    end

    @testset "ALS truncation runs and preserves the state's invariants" begin
        dt, D = 0.05, 2
        state = productpepo(fill((0, 0, -1), 2, 2), D)
        weights = SUWeight(state)
        decomposition = trotterise(lindblad(), P, dt)
        alg = ALSTruncation(; trunc = truncspace(ComplexSpace(D)), maxiter = 20)

        envalg = BondEnvironment(8)
        for _ in 1:3
            quiet(() -> timestep!(state, weights, decomposition; truncalg = alg, envalg))
        end

        m = onesite(state)
        @test abs(real(tr(m)) - 1) < 1e-6
        @test norm(m - m') < 1e-6
        @test minimum(real, eigvals(Hermitian((m + m') / 2))) > -1e-6
    end
end
