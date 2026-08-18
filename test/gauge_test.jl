@testsetup module GaugeSetup
using Reexport
@reexport using TimeEvolutionPEPO
@reexport using TensorKit
@reexport using LinearAlgebra
@reexport using Logging

export quiet, evolved, observables, weightspread

quiet(f) = with_logger(f, ConsoleLogger(stderr, Logging.Error))

"""A state with genuine entanglement to gauge: five steps of an interacting Lindblad model."""
function evolved(; D = 4, numsteps = 5)
    X, Y, Z = PAULI
    sim = quiet(
        () -> Simulation(
            productpepo(fill((0, 0, -1), 2, 2), D),
            1.5 / 4 * LocalOp(Z, Z) + Dissipator(0.5 * (Y - im * Z)); timestep = 0.05
        ),
    )
    quiet(() -> simulate!(sim; numsteps, verbose = false))
    return sim
end

"""One-site expectation values, which any gauge transformation must leave alone."""
function observables(state; bonddim = 8)
    X, Y, Z = PAULI
    dm = quiet(() -> DensityMatrix(state, bonddim))
    return [real(expval(partialtrace(dm, (1, 1)), O)) for O in (X, Y, Z)]
end

"""Ratio of largest to smallest bond weight, summed over bonds: how far from flat the weights are."""
weightspread(w) = sum(maximum(diag(convert(Array, x))) / max(minimum(diag(convert(Array, x))), 1e-300)
    for x in collect(w))
end # GaugeSetup

@testitem "Gauge fixing" setup = [GaugeSetup] begin
    @testset "SUGauge leaves observables alone" begin
        sim = evolved()
        before = observables(quantumstate(sim))
        weights_before = deepcopy(bondweights(sim))

        ϵ = quiet(() -> gaugefix!(sim))

        @test ϵ < 1e-8                                    # trivial simple update converged
        @test observables(quantumstate(sim)) ≈ before atol = 1e-8
        # The gauge actually changed: weights moved, and they are no longer the trivial ones.
        @test !isapprox(bondweights(sim), weights_before; atol = 1e-10)
    end

    @testset "SUGauge is idempotent" begin
        sim = evolved()
        quiet(() -> gaugefix!(sim))
        fixed = deepcopy(bondweights(sim))
        ϵ = quiet(() -> gaugefix!(sim))
        @test ϵ < 1e-8
        @test bondweights(sim) ≈ fixed atol = 1e-8       # already in the fixed-point gauge
    end

    @testset "belief propagation gauge" begin
        sim = evolved()
        before = observables(quantumstate(sim))

        env, info = quiet(() -> bpenvironment(quantumstate(sim)))
        quiet(() -> gaugefix!(sim, BPGauge(), env))

        @test observables(quantumstate(sim)) ≈ before atol = 1e-6
    end

    @testset "a product state is already gauge fixed" begin
        state = productpepo(fill((0, 0, -1), 2, 2), 1)
        weights = SUWeight(state)
        before = observables(state; bonddim = 4)
        quiet(() -> gaugefix!(state, weights))
        @test observables(state; bonddim = 4) ≈ before atol = 1e-8
    end
end
