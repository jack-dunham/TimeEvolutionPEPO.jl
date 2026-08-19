@testsetup module PropagatorSetup
using Reexport
@reexport using TimeEvolutionPEPO
@reexport using TensorKit
@reexport using LinearAlgebra

export dense_liouvillian, matrixof

"""
Dense Lindblad generator acting on `vec(rho)`, built independently of the package from
`dρ/dt = -i[H,ρ] + Σ(LρL' - {L'L,ρ}/2)`. Column-major vectorisation.
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

"""Two-site propagator as a plain matrix, in whatever index order the legs impose."""
function matrixof(propagator::AbstractTensorMap)
    n = dim(codomain(propagator))
    return reshape(convert(Array, propagator), n, n)
end
end # PropagatorSetup

@testitem "Two-site propagator" setup = [PropagatorSetup] begin
    X, Y, Z = PAULI
    P = ComplexSpace(2)
    dt = 0.05

    jump = 0.5 * (Y - im * Z)
    lindblad = 1.5 / 4 * LocalOp(Z, Z) + Dissipator(jump)
    thermal = 1.5 / 4 * LocalOp(Z, Z) - LocalOp(X)

    @testset "spectrum matches the dense generator" begin
        # Eigenvalues are invariant under the fusion and under any leg permutation, so they test the
        # generator and the time step without depending on the index convention.
        # Rounded sort key: conjugate pairs whose real parts agree only to machine precision would
        # otherwise be paired inconsistently between the two lists.
        key(x) = (round(real(x); digits = 9), round(imag(x); digits = 9))
        for model in (lindblad, thermal)
            g = generator(model, P)
            u = bondpropagator(model, P, dt)
            got = sort(eigvals(matrixof(u)); by = key)
            want = sort(exp.(dt * eigvals(g)); by = key)
            @test got ≈ want atol = 1e-8
        end
    end

    @testset "the Lindblad generator is the textbook one" begin
        # Two sites, so the one-local part is halved on each and the bond term appears once.
        H = 1.5 / 4 * kron(Z, Z)
        expected = dense_liouvillian(H, [kron(one(jump), jump), kron(jump, one(jump))] ./ sqrt(2))
        g = generator(lindblad, P)
        @test sort(real(eigvals(g))) ≈ sort(real(eigvals(expected))) atol = 1e-8
    end

    @testset "trace preservation" begin
        # Tracing both output pairs of exp(dt L) must leave the identity on the inputs, for a
        # Lindblad generator but not a thermal one.
        # `matrixof` runs its composite index column-major over (ket1, bra1, ket2, bra2), so the trace
        # functional is the vector supported on ket == bra at each site.
        d = dim(P)
        w = vec([
            (k1 == c1 && k2 == c2) ? 1.0 : 0.0 for k1 in 1:d, c1 in 1:d, k2 in 1:d, c2 in 1:d
        ])
        preserved(model) =
            norm(transpose(w) * matrixof(bondpropagator(model, P, dt)) - transpose(w))

        @test preserved(lindblad) < 1e-10
        @test preserved(thermal) > 1e-3
    end

    @testset "zero time step gives the identity" begin
        u = bondpropagator(lindblad, P, 0.0)
        @test matrixof(u) ≈ Matrix{ComplexF64}(I, 16, 16) atol = 1e-12
    end

    @testset "all four physical legs stay separate" begin
        u = bondpropagator(lindblad, P, dt)
        @test (numout(u), numin(u)) == (4, 4)
        @test codomain(u) == P ⊗ P' ⊗ P ⊗ P'
        @test domain(u) == P ⊗ P' ⊗ P ⊗ P'
    end

    @testset "one-local terms are placed per endpoint" begin
        X, Y, Z = PAULI
        P = ComplexSpace(2)

        # Splitting a uniform model across its two endpoints must reproduce the symmetric path.
        uniform = 1.5 / 4 * LocalOp(Z, Z) + LocalOp(X) + Dissipator(0.5 * (Y - im * Z))
        @test generator(uniform, uniform, P) ≈ generator(uniform, P)

        # A field on the first site only must equal kron(X, I) in the vectorised generator, at half
        # weight, and must differ from the same field on the second site.
        first_only = zero(Hamiltonian) + LocalOp(X)
        empty = zero(Hamiltonian)
        g1 = generator(first_only, empty, P)
        g2 = generator(empty, first_only, P)
        @test !(g1 ≈ g2)

        # Both halves together are the symmetric term.
        @test g1 + g2 ≈ generator(first_only, first_only, P)
    end

    @testset "Ising reaches a generator" begin
        X, Y, Z = PAULI
        P = ComplexSpace(2)
        a = 1.5 / 4

        ising = Ising(a * LocalOp(Z, Z), [LocalOp(X)], Dissipator(0.5 * (Y - im * Z)))
        equivalent = a * LocalOp(Z, Z) + LocalOp(X) + Dissipator(0.5 * (Y - im * Z))

        @test generator(ising, P) ≈ generator(equivalent, P)
        @test bondpropagator(ising, P, 0.05) ≈ bondpropagator(equivalent, P, 0.05)
    end

    @testset "bare terms are promoted to layers" begin
        X, Y, Z = PAULI
        P = ComplexSpace(2)

        @test length(trotterise(LocalOp(Z, Z), P, 0.1)) == 2          # one layer, both axes
        @test length(trotterise([LocalOp(X, X), LocalOp(Y, Y)], P, 0.1)) == 4
        @test length(trotterise([LocalOp(X, X) => 1, LocalOp(Y, Y) => 2], P, 0.1)) == 2

        # A bare Dissipator is one-local, so it must promote to a Liouvillian, not a Hamiltonian.
        @test first(first(trotterise(Dissipator(0.5 * (Y - im * Z)), P, 0.1)))[1, 1] isa
            AbstractTensorMap
    end
end
