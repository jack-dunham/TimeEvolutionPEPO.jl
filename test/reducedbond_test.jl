@testsetup module ReducedBondSetup
using Reexport
@reexport using TimeEvolutionPEPO
@reexport using TensorKit
@reexport using LinearAlgebra
@reexport using PEPSKit

export randpepo, dense_liouvillian

"""Random PEPO tensor on the PEPOTensor convention, physical `d` and bond `D`."""
function randpepo(d::Int, D::Int)
    P, V = ComplexSpace(d), ComplexSpace(D)
    return randn(ComplexF64, P ⊗ P' ← V ⊗ V ⊗ V' ⊗ V')
end

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
end # ReducedBondSetup

@testitem "Reduced bond tensors" setup = [ReducedBondSetup] begin
    X, Y, Z = PAULI
    P = ComplexSpace(2)

    @testset "splitting is invertible" begin
        for D in (1, 2, 4)
            A = randpepo(2, D)
            a, Xrem = reducedbond_first(A)
            @test undo_reducedbond_first(a, Xrem) ≈ A atol = 1e-10

            b, Yrem = reducedbond_last(A)
            @test undo_reducedbond_last(b, Yrem) ≈ A atol = 1e-10
        end
    end

    @testset "reduced tensors carry both physical legs" begin
        A = randpepo(2, 3)
        a, _ = reducedbond_first(A)
        b, _ = reducedbond_last(A)
        # (χ ⊗ ket ⊗ bra) ← bond, so three codomain indices rather than PEPSKit's two.
        @test (numout(a), numin(a)) == (3, 1)
        @test (numout(b), numin(b)) == (3, 1)
        @test space(a, 2) == P && space(a, 3) == P'
        @test space(b, 2) == P && space(b, 3) == P'
    end

    @testset "a propagator that is the identity leaves the pair unchanged" begin
        A, B = randpepo(2, 2), randpepo(2, 2)
        a, _ = reducedbond_first(A)
        b, _ = reducedbond_last(B)

        identity_propagator = bondpropagator(zero(Hamiltonian) + Dissipator(0.0 * Z), P, 0.0)
        trunc = truncspace(domain(a, 1))
        a2, s, b2, ϵ = applypropagator(a, b, identity_propagator, trunc)

        @tensor before[-1 -2 -3; -4 -5 -6] := a[-1 -2 -3 x] * b[x -5 -6 -4]
        @tensor after[-1 -2 -3; -4 -5 -6] := a2[-1 -2 -3 x] * b2[x -5 -6 -4]
        @test after ≈ before atol = 1e-9
        @test ϵ < 1e-12
    end
end
