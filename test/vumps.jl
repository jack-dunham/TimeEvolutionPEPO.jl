# function statmechmpo(β, h, D)
#     M = zeros(D, D, D, D)
#     for i in 1:D
#         M[i, i, i, i] = 1
#     end
#     X = zeros(D, D)
#     for j in 1:D, i in 1:D
#         X[i, j] = exp(-β * h(i, j))
#     end
#     Xsq = sqrt(X)
#     @tensor M1[a, b, c, d] :=
#         M[a', b', c', d'] * Xsq[c', c] * Xsq[d', d] * Xsq[a, a'] * Xsq[b, b']
#
#     # For computing energy: M2 is a tensor across 2 nearest neighbor sites in the lattice, whose
#     # expectation value in the converged fixed point of the transfer matrix represents the energy
#     Y = zeros(D, D)
#     for j in 1:D, i in 1:D
#         Y[i, j] = h(i, j) * exp(-β * h(i, j))
#     end
#     @tensor M2[a, b1, b2, c, d2, d1] :=
#         M[a', b1', c1, d1'] *
#         Xsq[a, a'] *
#         Xsq[b1, b1'] *
#         Xsq[d1', d1] *
#         Y[c1, c2] *
#         M[c2, b2', c', d2'] *
#         Xsq[b2, b2'] *
#         Xsq[d2', d2] *
#         Xsq[c', c]
#
#     Z0 = [1.0 0.0; 0.0 -1.0]
#     Z = zeros(D, D)
#     for j in 1:D, i in 1:D
#         Z[i, j] = Z0[i, j] * exp(-β * h(i, j))
#     end
#     @tensor M3[a, b1, b2, c, d2, d1] :=
#         M[a', b1', c1, d1'] *
#         Xsq[a, a'] *
#         Xsq[b1, b1'] *
#         Xsq[d1', d1] *
#         Z[c1, c2] *
#         M[c2, b2', c', d2'] *
#         Xsq[b2, b2'] *
#         Xsq[d2', d2] *
#         Xsq[c', c]
#
#     return M1, M2, M3
# end
#
# function classicalisingmpo(β; J=1.0, h=0.0)
#     return statmechmpo(β, (s1, s2) -> -J * (-1)^(s1 != s2) - h / 2 * (s1 == 1 + s2 == 1), 2)
# end

function statmechmpo_alt(β, h, D)
    δ1 = zeros(D, D, D, D)
    for i in 1:D
        δ1[i, i, i, i] = 1 # kronecker delta
    end

    δ2 = zeros(D, D, D, D, D)
    for i in 1:D
        δ2[i, i, i, i, i] = 1 # kronecker delta
    end

    X = zeros(D, D)
    for j in 1:D, i in 1:D
        X[i, j] = exp(-β * h(i, j))
    end
    Xsq = sqrt(X)

    Z = [1.0, -1.0]

    @tensor M1[a, b, c, d] :=
        δ1[a', b', c', d'] * Xsq[c', c] * Xsq[d', d] * Xsq[a, a'] * Xsq[b, b']

    @tensor M2[a, b, c, d] :=
        δ2[a', b', c', d', e'] * Xsq[c', c] * Xsq[d', d] * Xsq[a, a'] * Xsq[b, b'] * Z[e']

    return M1, M2
end

function classicalisingmpo_alt(β; J=1.0, h=0.0)
    return statmechmpo_alt(
        β, (s1, s2) -> -J * (-1)^(s1 != s2) - h / 2 * (s1 == 1 + s2 == 1), 2
    )
end
#----------

@testset "_AC & _CA" begin
    C = TensorMap(randn, ComplexF64, ℂ^5, ℂ^5)
    A = TensorMap(randn, ComplexF64, ℂ^2 * ℂ^5, ℂ^5)
    @tensor CA[s a; b] := C[a; x] * A[s x; b]
    @tensor AC[s a; b] := A[s a; x] * C[x; b]
    @test CA ≈ PEPOKit._CA(C, A)
    @test AC ≈ PEPOKit._AC(A, C)
end

@testset "vumps" begin
    S = ℂ
    tA = TensorMap(randn, ComplexF64, S^2 * S^4, S^2 * S^4)
    tM = TensorMap(
        randn,
        ComplexF64,
        S^2 * S^2 * S^2 * S^2 * S^2 * S^2,
        S^2 * S^2 * S^2 * S^2 * S^2 * S^2,
    )
    aA = convert(Array, tA)
    aM = convert(Array, tM)

    tAL, tC, tλ = PEPOKit.leftgauge(tA)
    aAL, aC, aλ = leftgauge(aA)

    tAR, tC, tλ = PEPOKit.rightgauge(tAL)
    aAR, aC, aλ = rightgauge(aAL)
    @testset "gauging" begin
        @testset "left" begin
            @test aAL ≈ convert(Array, tAL) # Test AL correct
            @test aC ≈ convert(Array, tC) # Test C is correct
            @test aλ ≈ tλ # Test λ is correct
        end
        @testset "right" begin
            @test aAR ≈ convert(Array, tAR) # Test AL correct
            @test aC ≈ convert(Array, tC) # Test C is correct
            @test aλ ≈ tλ # Test λ is correct
        end
    end
    order_A = ((1,), (2, 3), (4,))
    order_FL = ((1,), (2, 3, 4, 5), (6,))
    order_M = ((1, 2, 7, 8), (3, 9), (4, 5, 10, 11), (6, 12))
    tFL, tλL, _ = PEPOKit.leftenv(tAL, tM) # this is (2,2) (6,6)
    aFL, aλL, _ = leftenv(_merge(aAL, order_A), _merge(aM, order_M)) # this is rank 3, rank 4
    @testset "environments" begin
        tFL_arr = _merge(convert(Array, tFL), order_FL)
        @test tλL ≈ aλL
        @info "tλL = $(tλL) \n aλL = $(aλL)"
        # @test aFL ≈ tFL_arr
    end
end
