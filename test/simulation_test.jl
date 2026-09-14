@testsetup module SimulationSetup
using Reexport
@reexport using TimeEvolutionPEPO
@reexport using TensorKit
@reexport using LinearAlgebra
@reexport using Logging

export dense_liouvillian, evolve_dense, onesite, quiet

quiet(f) = with_logger(f, ConsoleLogger(stderr, Logging.Warn))

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

function onesite(pepo, ind = (1, 1); bonddim = 4)
    dm = quiet(() -> DensityMatrix(pepo, bonddim))
    return reshape(convert(Array, partialtrace(dm, ind)), 2, 2)
end
end # SimulationSetup

@testitem "Simulation driver" setup = [SimulationSetup] begin
    X, Y, Z = PAULI
    jump = 0.5 * (Y - im * Z)

    @testset "one-local Lindblad trajectory against exp(t L)" begin
        # Benchmark B1 through the driver: exact at D = 1, no truncation and no Trotter error.
        v0 = (0.0, 0.0, -1.0)
        dt = 0.02
        G = dense_liouvillian(zeros(ComplexF64, 2, 2), [jump])

        sim = Simulation(
            productpepo(fill(v0, 2, 2), 1), zero(Hamiltonian) + Dissipator(jump); timestep = dt
        )

        deviation = 0.0
        callback = function (s)
            info = simulationinfo(s)
            m = onesite(quantumstate(s))
            exact = evolve_dense(bloch(v0), G, info.simtime)
            deviation = max(
                deviation,
                abs(real(tr(m * X)) - real(tr(exact * X))),
                abs(real(tr(m * Z)) - real(tr(exact * Z))),
            )
            return nothing
        end
        quiet(() -> simulate!(callback, sim; numsteps = 10, verbose = false))

        @test deviation < 1e-8
        @test simulationinfo(sim).iterations == 10
        @test simulationinfo(sim).simtime ≈ 10 * dt
        @test simulationinfo(sim).finished
    end

    @testset "Larmor precession fixes the sign of the Hamiltonian" begin
        # Benchmark B6: <Y> = +sin(omega t) for H = (h/2) Z from a state along +x.
        h, dt = 2.0, 0.05
        G = dense_liouvillian((h / 2) * Z, Matrix{ComplexF64}[])
        model = h / 2 * LocalOp(Z) + Dissipator(0.0 * Z)

        sim = Simulation(productpepo(fill((1.0, 0.0, 0.0), 2, 2), 1), model; timestep = dt)

        deviation, reversed = 0.0, 0.0
        callback = function (s)
            t = simulationinfo(s).simtime
            m = onesite(quantumstate(s))
            exact = evolve_dense(bloch((1.0, 0.0, 0.0)), G, t)
            got, want = real(tr(m * Y)), real(tr(exact * Y))
            deviation = max(deviation, abs(got - want))
            reversed = max(reversed, abs(got + want))
            return nothing
        end
        quiet(() -> simulate!(callback, sim; numsteps = 6, verbose = false))

        @test deviation < 1e-8
        @test reversed > 0.1        # the reversed convention is excluded, not merely unfavoured
    end

    @testset "callback schedule and accessors" begin
        sim = Simulation(
            productpepo(fill((0, 0, -1), 2, 2), 2),
            1.5 / 4 * LocalOp(Z, Z) + Dissipator(jump); timestep = 0.05
        )
        @test quantumstate(sim) isa InfinitePEPO
        @test bondweights(sim) isa SUWeight
        @test timestep(sim) == 0.05

        shots = Int[]
        quiet(
            () -> simulate!(
                s -> push!(shots, simulationinfo(s).iterations), sim;
                numsteps = 8, maxshots = 4, verbose = false
            ),
        )
        # Step 0 plus at most `maxshots` evenly spaced steps.
        @test first(shots) == 0
        @test length(shots) <= 5
        @test issorted(shots)
        @test last(shots) == 8
    end

    @testset "convergence measure and early stop" begin
        weights = SUWeight(productpepo(fill((0, 0, -1), 2, 2), 2))
        @test weightdistance(weights, weights, 0.1) == 0.0

        sim = Simulation(
            productpepo(fill((0, 0, -1), 2, 2), 1),
            zero(Hamiltonian) + Dissipator(0.0 * Z); timestep = 0.05
        )
        # A propagator that does nothing leaves the weights fixed, so this stops immediately.
        quiet(() -> simulate!(sim; numsteps = 20, breaktol = 1e-12, verbose = false))
        @test simulationinfo(sim).converged
        @test simulationinfo(sim).iterations < 20
    end
end

@testitem "Thermal evolution" setup = [SimulationSetup] begin
    using MatrixAlgebraKit: truncrank

    # rho = exp(-beta H) for the classical Ising model factorises over bonds, so each bond carries
    # singular values proportional to (cosh(beta J), sinh(beta J)) and D = 2 is exact at every beta.
    J, dβ = 1.0, 0.002
    H = J * LocalOp(PAULI[3], PAULI[3])

    function ratio(β, D; gaugeevery = 0)
        sim = quiet(
            () -> Simulation(productpepo(fill(ThermalState(), 2, 2), D), H; timestep = dβ)
        )
        quiet(
            () -> simulate!(
                sim; numsteps = round(Int, β / dβ), verbose = false,
                trunc = truncrank(D), gaugeevery
            ),
        )
        w = sort(abs.(bondweights(sim)[1, 1, 1].data); rev = true)
        return w[2] / w[1]
    end

    @testset "a bare Hamiltonian evolves in imaginary time" begin
        for β in (0.05, 0.15, 0.25)
            @test ratio(β, 2) ≈ tanh(β * J) rtol = 1.0e-5
        end
    end

    @testset "re-gauging carries it past the critical point" begin
        # betac = log(1 + sqrt(2)) / 2 = 0.4407. Without re-gauging the sweep loses the state near
        # beta = 0.28: the truncation error stays at 1e-16 and it happens at D = 2 where the state is
        # exactly representable, so the drift is in the gauge, not the truncation.
        # betac = 0.4407 is not a multiple of dbeta, so this uses the step grid: 0.44 is one step
        # below it and 0.46 one above.
        for β in (0.35, 0.44, 0.46, 0.6)
            @test ratio(β, 2; gaugeevery = 25) ≈ tanh(β * J) rtol = 1.0e-5
        end
        @test ratio(0.35, 2) < 0.5 * tanh(0.35 * J)            # collapsed without it
    end

    @testset "re-gauging recovers a state that has already collapsed" begin
        # The gauge drift destroys no information: fixing it after the collapse restores the spectrum.
        @test ratio(0.6, 2; gaugeevery = 50) ≈ tanh(0.6 * J) rtol = 1.0e-5
    end
end
