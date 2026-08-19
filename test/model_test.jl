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
end
