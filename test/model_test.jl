@testsetup module ModelSetup
using Reexport
@reexport using TimeEvolutionPEPO
@reexport using TensorKit
@reexport using LinearAlgebra
@reexport using Logging

export quiet
quiet(f) = with_logger(f, ConsoleLogger(stderr, Logging.Error))
end # ModelSetup

@testitem "Trotter layers" setup = [ModelSetup] begin
    X, Y, Z = PAULI

    @testset "a bare term is a 1x1 layer" begin
        layer = TrotterLayer(LocalOp(Z, Z))
        @test size(layer) == (1, 1)
        @test layer.axis == 0
        @test layer[1, 1] isa Hamiltonian
        # A 1x1 layer wraps periodically, so it serves any unit cell.
        @test layer[3, 7] === layer[1, 1]
    end

    @testset "a matrix of terms keeps its shape" begin
        data = [LocalOp(Z, Z) LocalOp(X, X); LocalOp(X, X) LocalOp(Z, Z)]
        layer = TrotterLayer(data, 1)
        @test size(layer) == (2, 2)
        @test layer.axis == 1
        @test layer[3, 3] === layer[1, 1]
    end

    @testset "a non-axial term cannot be pinned to an axis" begin
        @test_throws ArgumentError TrotterLayer(Dissipator(Z), 1)
    end

    @testset "an empty dissipator does not block pinning to an axis" begin
        layer = TrotterLayer(Liouvillian(zero(Hamiltonian) + LocalOp(Z, Z)), 1)
        @test layer.axis == 1
        @test_throws ArgumentError TrotterLayer(
            zero(Hamiltonian) + LocalOp(Z, Z) + Dissipator(Z), 1
        )
    end
end

@testitem "Model container" setup = [ModelSetup] begin
    X, Y, Z = PAULI

    @testset "layers are appended, never distributed" begin
        model = Model(LocalOp(X, X), LocalOp(Y, Y))
        @test length(model) == 2
        push!(model, LocalOp(Z, Z))
        @test length(model) == 3
        # push! appends whether or not the term is one-local.
        push!(model, LocalOp(X))
        @test length(model) == 4
    end

    @testset "distribute! spreads over existing layers" begin
        model = Model(LocalOp(X, X), LocalOp(Y, Y))
        distribute!(model, LocalOp(Z))
        # The layer count is unchanged; the numerical share is checked in the next testitem, once
        # trotterise accepts a Model.
        @test length(model) == 2
        @test length(oneterms(model[1][1, 1])) == 1
    end

    @testset "distributing over no layers is an error" begin
        @test_throws ArgumentError distribute!(Model{Hamiltonian}(), LocalOp(Z))
    end

    @testset "a Hamiltonian model rejects dissipation" begin
        model = Model{Hamiltonian}()
        push!(model, LocalOp(X, X))
        @test_throws ArgumentError push!(model, Dissipator(Z))
    end

    @testset "distribute! rejects a term outside the model's family" begin
        model = Model{Hamiltonian}()
        push!(model, LocalOp(X, X))
        @test_throws ArgumentError distribute!(model, Dissipator(Z))
        # The rejected call must not have mutated any layer before throwing.
        @test length(model) == 1
        @test dissipator(model[1][1, 1]) === nothing
    end
end

@testitem "Trotter decomposition of a model" setup = [ModelSetup] begin
    X, Y, Z = PAULI
    P = ComplexSpace(2)

    @testset "a uniform layer yields a 1x1 propagator matrix" begin
        decomposition = trotterise(LocalOp(Z, Z), P, 0.1)
        @test length(decomposition) == 2                   # one layer over both axes
        propagators, axis = first(decomposition)
        @test size(propagators) == (1, 1)
        @test axis == 1
    end

    @testset "a site-dependent layer yields one propagator per site" begin
        data = [LocalOp(Z, Z) 0.5 * LocalOp(Z, Z); 0.5 * LocalOp(Z, Z) LocalOp(Z, Z)]
        decomposition = trotterise(Model([TrotterLayer(data)]), P, 0.1)
        propagators, = first(decomposition)
        @test size(propagators) == (2, 2)
        @test !(propagators[1, 1] ≈ propagators[1, 2])
    end

    @testset "distribute! gives each layer a 1/n share" begin
        # Moved here from Task 6: the check needs trotterise to accept a Model.
        distributed = Model(LocalOp(X, X), LocalOp(Y, Y))
        distribute!(distributed, LocalOp(Z))
        byhand = Model(LocalOp(X, X) + 0.5 * LocalOp(Z), LocalOp(Y, Y) + 0.5 * LocalOp(Z))
        got = trotterise(distributed, P, 0.1)
        want = trotterise(byhand, P, 0.1)
        @test length(got) == length(want)
        @test all(first(g)[1, 1] ≈ first(w)[1, 1] for (g, w) in zip(got, want))
    end

    @testset "the one-local part is halved across the two axes" begin
        # A layer applied on both axes must halve its one-local part; pinned to one axis it must not.
        both = first(first(trotterise(zero(Hamiltonian) + LocalOp(X), P, 0.1)))[1, 1]
        pinned = first(first(trotterise([(zero(Hamiltonian) + LocalOp(X)) => 1], P, 0.1)))[1, 1]
        @test !(both ≈ pinned)
    end
end
