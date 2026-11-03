@testsetup module ObservablesSetup
using Reexport
@reexport using TimeEvolutionPEPO
@reexport using TensorKit
@reexport using LinearAlgebra
@reexport using Logging

export quiet, rdm_from_matrix

quiet(f) = with_logger(f, ConsoleLogger(stderr, Logging.Warn))

"""Two-site reduced density matrix built from a 4x4 matrix, in the layout `partialtrace` returns."""
function rdm_from_matrix(m::AbstractMatrix)
    P = ComplexSpace(2)
    return TensorMap(reshape(ComplexF64.(m), 2, 2, 2, 2), P ⊗ P ← P ⊗ P)
end
end # ObservablesSetup

@testitem "Observables" setup = [ObservablesSetup] begin
    X, Y, Z = PAULI

    @testset "expval orders operators by site" begin
        up, down = (0.0, 0.0, 1.0), (0.0, 0.0, -1.0)
        dm = quiet(() -> DensityMatrix(productpepo([up down; up down], 1), 4))

        @test real(expval(partialtrace(dm, (1, 1)), Z)) ≈ 1 atol = 1e-10
        @test real(expval(partialtrace(dm, (1, 2)), Z)) ≈ -1 atol = 1e-10

        pair = partialtrace(dm, (1, 1), (1, 2))
        @test real(expval(pair, Z, one(Z))) ≈ 1 atol = 1e-10
        @test real(expval(pair, one(Z), Z)) ≈ -1 atol = 1e-10
        @test real(expval(pair, Z, Z)) ≈ -1 atol = 1e-10
        @test real(expval(pair, one(Z), one(Z))) ≈ 1 atol = 1e-10

        # Non-adjacent indices go through the general contraction.
        @test real(expval(partialtrace(dm, (1, 1), (2, 2)), Z, one(Z))) ≈ 1 atol = 1e-10
        @test_throws ArgumentError expval(pair, Z)
    end

    @testset "negativity" begin
        # A product state is separable, a Bell state has negativity 1/2.
        product = kron([1 0; 0 0], [1 0; 0 0])
        @test negativity(rdm_from_matrix(product)) ≈ 0 atol = 1e-10

        bell = zeros(ComplexF64, 4, 4)
        for i in (1, 4), j in (1, 4)
            bell[i, j] = 0.5
        end
        @test negativity(rdm_from_matrix(bell)) ≈ 0.5 atol = 1e-10

        # A Werner state is separable up to p = 1/3 and entangled beyond it.
        werner(p) = p * bell + (1 - p) * Matrix{ComplexF64}(I, 4, 4) / 4
        @test negativity(rdm_from_matrix(werner(0.2))) ≈ 0 atol = 1e-10
        @test negativity(rdm_from_matrix(werner(0.9))) > 0.3

        dm = quiet(() -> DensityMatrix(productpepo(fill((0, 0, -1), 2, 2), 1), 4))
        @test negativity(partialtrace(dm, (1, 1), (1, 2))) ≈ 0 atol = 1e-8
        @test_throws ArgumentError negativity(partialtrace(dm, (1, 1)))
    end

    @testset "correlation length and correlator of a product state" begin
        dm = quiet(() -> DensityMatrix(productpepo(fill((0.3, 0.0, -0.7), 2, 2), 2), 6))

        # A product state has no correlations, but the correlation length does not come out exactly
        # zero: the second transfer eigenvalue is ~1e-20 rather than 0, and -1/log|λ₂/λ₁| turns that
        # into ~1/46. What matters is that it is far below one lattice spacing, and that it grows once
        # an interacting evolution has built correlations.
        product_length = quiet(() -> correlationlength(dm))
        @test product_length < 0.05

        zz = real(correlator(dm, Z, (1, 1), Z, (1, 2)))
        z1 = real(expval(partialtrace(dm, (1, 1)), Z))
        z2 = real(expval(partialtrace(dm, (1, 2)), Z))
        @test zz ≈ z1 * z2 atol = 1e-8

        # An interacting evolution correlates neighbouring sites, which the connected correlator must
        # register. The correlation length is not used for this: after a short evolution it is still
        # below one lattice spacing, where it sits near the numerical floor described above and also
        # fluctuates with the random CTMRG initialisation.
        sim = quiet(
            () -> Simulation(
                productpepo(fill((0, 0, -1), 2, 2), 4),
                1.5 / 4 * LocalOp(Z, Z) + Dissipator(0.5 * (Y - im * Z)); timestep = 0.05
            ),
        )
        quiet(() -> simulate!(sim; numsteps = 5, verbose = false))
        evolved = quiet(() -> DensityMatrix(quantumstate(sim), 8))

        function connected(A, B)
            return real(correlator(evolved, A, (1, 1), B, (1, 2))) -
                real(expval(partialtrace(evolved, (1, 1)), A)) *
                real(expval(partialtrace(evolved, (1, 2)), B))
        end

        # The transverse channel correlates, but the connected ZZ correlator stays zero: the
        # Hamiltonian (J/4) sum ZZ commutes with every Z_i, so the Z sector closes and no ZZ
        # correlation is generated from a Z-diagonal initial state.
        @test abs(connected(X, X)) > 1e-4
        @test abs(connected(Z, Z)) < 1e-10
    end

    @testset "purified density matrix reproduces X'X" begin
        # With X = sqrt(rho) on every site, the purified state is rho itself.
        rho = bloch((0.2, 0.0, -0.6))
        root = sqrt(Hermitian(rho))

        purified = quiet(() -> PurifiedDensityMatrix(productpepo(fill(Matrix(root), 2, 2), 1), 4))
        @test isconverged(purified)

        m = reshape(convert(Array, partialtrace(purified, (1, 1))), 2, 2)
        @test m ≈ rho atol = 1e-8
        @test real(expval(partialtrace(purified, (1, 1)), Z)) ≈ -0.6 atol = 1e-8
        @test quiet(() -> correlationlength(purified)) < 0.05
    end
end
