@testsetup module StateSetup
using Reexport
@reexport using TimeEvolutionPEPO
@reexport using TensorKit
@reexport using LinearAlgebra

export onesite

"""One-site reduced density matrix as a plain matrix, via the traced-network boundary."""
function onesite(pepo, ind = (1, 1); bonddim = 4)
    dm = DensityMatrix(pepo, bonddim)
    return reshape(convert(Array, partialtrace(dm, ind)), 2, 2), dm
end
end # StateSetup

@testitem "Product state preparation" setup = [StateSetup] begin
    X, Y, Z = PAULI

    @testset "mixed states are prepared as requested" begin
        # The constructor this replaces prepared rho^4, which agrees only for pure states.
        for v in ((0.0, 0.0, -0.5), (0.4, 0.5, -0.6), (0.2, -0.3, 0.1), (0.0, 0.0, 0.0))
            rho = bloch(v)
            m, dm = onesite(productpepo(fill(rho, 2, 2), 1))
            @test isconverged(dm)
            @test m ≈ rho atol = 1e-10
            for (component, pauli) in zip(v, (X, Y, Z))
                @test real(expval(partialtrace(dm, (1, 1)), pauli)) ≈ component atol = 1e-10
            end
        end
    end

    @testset "pure states and the thermal state" begin
        m, _ = onesite(productpepo((0, 0, -1), 1))
        @test m ≈ bloch((0, 0, -1)) atol = 1e-10

        m, _ = onesite(productpepo(ThermalState(), 1))
        @test m ≈ Matrix{ComplexF64}(I, 2, 2) / 2 atol = 1e-10
    end

    @testset "unit cells and bond padding" begin
        @test size(productpepo(fill((0, 0, -1), 2, 3), 2)) == (2, 3, 1)
        @test size(productpepo((0, 0, -1), 4)) == (1, 1, 1)

        # Padding the bond space must not change the state.
        rho = bloch((0.3, 0.0, -0.7))
        m1, _ = onesite(productpepo(fill(rho, 2, 2), 1))
        m4, _ = onesite(productpepo(fill(rho, 2, 2), 4))
        @test m1 ≈ m4 atol = 1e-10
    end

    @testset "Bloch vectors longer than one are rejected" begin
        @test_throws ArgumentError bloch((1.0, 1.0, 0.0))
    end

    @testset "spaces follow the PEPOTensor convention" begin
        pepo = productpepo((0, 0, -1), 3)
        t = pepo[1, 1]
        P, V = ComplexSpace(2), ComplexSpace(3)
        # P ⊗ P´ ← N ⊗ E ⊗ S ⊗ W, index 1 outgoing (ket) and index 2 incoming (bra).
        @test (numout(t), numin(t)) == (2, 4)
        @test codomain(t) == P ⊗ P'
        @test domain(t) == V ⊗ V ⊗ V' ⊗ V'
    end

    @testset "trivial weights and bond entropy" begin
        pepo = productpepo(fill((0, 0, -1), 2, 2), 3)
        w = SUWeight(pepo)
        @test size(w) == (2, 2, 2)
        @test all(≈(log(3)), bondentropy(w))                       # trivial weights are flat
        @test all(≈(1.0), bondentropy(w; normalized = true))
    end

    @testset "bond entropy uses the Schmidt probabilities" begin
        # A flat spectrum cannot distinguish normalising λ from normalising λ², so these use a
        # non-flat one. Normalising λ instead would give 0.63651, not 0.50040.
        pepo = productpepo(fill((0, 0, -1), 2, 2), 2)
        w = SUWeight(pepo)
        setall!(vals) = for d in 1:2, r in 1:2, c in 1:2
            w[d, r, c] = DiagonalTensorMap(vals, w[d, r, c].domain)
        end

        setall!([1.0, 0.5])
        p = [1.0, 0.25] ./ 1.25
        @test all(≈(-sum(x -> x * log(x), p)), bondentropy(w))
        @test all(≈(0.5004024235381879), bondentropy(w))

        # Limiting cases: one Schmidt value carries everything, or the spectrum is flat.
        setall!([1.0, 0.0])
        @test all(≈(0.0), bondentropy(w))
        setall!([1.0, 1.0])
        @test all(≈(log(2)), bondentropy(w))
    end

    @testset "a bare density matrix is one site, not a unit cell" begin
        rho = bloch((0.0, 0.0, -1.0))
        @test size(productpepo(rho, 4)) == (1, 1, 1)
        @test convert(Array, productpepo(rho, 4)[1, 1])[:, :, 1, 1, 1, 1] ≈ rho

        # A unit cell of density matrices still works. `[A B; C D]` with matrix entries block-concatenates
        # into one large matrix rather than building a matrix of matrices, so this uses `reshape` instead.
        cell = reshape(
            [bloch((0, 0, 1)), bloch((0, 0, -1)), bloch((0, 0, -1)), bloch((0, 0, 1))], 2, 2
        )
        @test size(productpepo(cell, 2)) == (2, 2, 1)
    end
end
