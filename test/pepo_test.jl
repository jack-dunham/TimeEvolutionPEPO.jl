@testsetup module PEPOImports

using Reexport
@reexport using TimeEvolutionPEPO
@reexport using TensorKit

end

@testitem "PEPO" setup = [PEPOImports] begin
    @testset "unit cell size" begin
        @test size(PEPO(rand, 2, 2)) == (1, 1)
        @test size(PEPO(rand, 2, fill(2, 3, 4))) == (3, 4)
        @test size(PEPO(rand, 2, fill(2, 1, 1))) == (1, 1)
        @test size(PEPO(rand, fill(2, 3, 4), 2)) == (3, 4)
        @test size(PEPO(fill(rand(2, 2), 3, 4), 2, 2)) == (3, 4)
    end
    @testset "spaces" begin
        # automatic
        @test only(physicalspace(PEPO((1, 0, 0), 3))) == ComplexSpace(2)
        @test only(physicalspace(PEPO(rand(3, 3), 4))) == ComplexSpace(3)

        let s = U1Space(-1 => 1, 1 => 1), S = U1Space(-1 => 1, 0 => 1, 1 => 1)
            @test only(physicalspace(PEPO((0, 0, -1), s, S))) == s
        end

        let s = [2 3; 4 5]
            @test all(physicalspace(PEPO(rand, s, 3)) .== ComplexSpace.(s))
        end
    end
    @testset "elements" begin
        let P = only(PEPO((1, 0, 0), 3))
            @test P[:, :, 1, 1, 1, 1] == (0.5 * (one(PAULI[1]) + PAULI[1]))^4
            @test P[:, :, 1, 1, 1, 2] == zero(PAULI[1])
            @test P[:, :, 3, 3, 3, 3] == zero(PAULI[1])
        end
        let P = rand(2, 2, 3, 3, 3, 3) # random tensor
            @test only(PEPO(P, 2, 3))[:, :, :, :, :, :] == P
        end
    end
    @testset "weights and `bondentropy`" begin
        let (H, V) = only(bondweights(PEPO(rand, 3, 3, 2)))
            @test !(H === V)
            @test H[:, :] == [1 0 0; 0 1 0; 0 0 1]
            @test V[:, :] == [1 0; 0 1]
            @test space(H) == (ComplexSpace(3) ← ComplexSpace(3))
            @test space(V) == (ComplexSpace(2) ← ComplexSpace(2))
        end

        @test size(bondweights(PEPO(rand, 2, [3 3; 3 3]))) == (2, 2)
        @test bondweights(PEPO{SquareSymmetric}(rand, 2, [3 2; 2 3])) isa UnitCellBonds

        w = zeros(3, 3)
        w[1, 1] = w[2, 2] = w[3, 3] = 1 / 3

        @test_throws MethodError bondentropy(w)

        wt = TensorMap(w, ComplexSpace(3), ComplexSpace(3))
        @test bondentropy(wt) ≈ log(3)
        @test bondentropy(wt; normalized=true) ≈ 1
    end
end
