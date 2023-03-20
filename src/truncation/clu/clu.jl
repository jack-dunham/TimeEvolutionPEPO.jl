abstract type AbstractClusterBoundary <: AbstractBoundary end

struct CLU{Alg<:AbstractClusterBoundary} <: AbstractTruncationScheme
    gauge::Int
    ξ::Float64
end

struct NTU <: AbstractClusterBoundary
    T::AbsTen{2,2}
    B::AbsTen{2,2}
    LC::AbsTen{1,1}
    RC::AbsTen{1,1}
end

# function ntucorner(hs; perm=(1, 2, 3, 4, 5, 6, 7, 8))
#     pp = _permute(hs, perm)
#     t = _ntucorner(pp)
#     return t
# end
#
# # function _ntucorner(p1, p2)
# #     @tensoropt begin
# #         t[w1 s1; w2 s2] := p1[w1 s1 k; e n b] * p2[e n b; w2 s2 k]
# #     end
# #     return t
# # end
# function _ntucorner(hs)
#     @tensoropt begin
#         t[w1 s1; w2 s2] := hs[w1 w2 s1 s2; e e n n]
#     end
#     return t
# end
#
# function ntumetric(A, B)
#     A_HS = hilbertschmidt(A)
#     TR = ntucorner(A_HS)
#     BR = ntucorner(A_HS; perm=(7, 8, 1, 2, 3, 4, 5, 6))
#
#     left_cap = B_HS = hilbertschmidt(B)
#     TL = ntucorner(B_HS; perm=(3, 4, 5, 6, 7, 8, 1, 2))
#     BL = ntucorner(B_HS; perm=(5, 6, 7, 8, 1, 2, 3, 4))
#     return nothing
# end

# function ntumetricfull(A, B, QA, QB)
#     @tensoropt begin
#         G[1 2; 3 4] :=
#             B[w1 s1 k1; e1 n1 b1] *
#             (B')[a_e1 n1 b1; w1 s1 k1] *
#             B[w2 s2 k2; e2 n2 b2] *
#             (B')[a_e2 n2 b2; w2 a_s2 k2] *
#             QA[e1 s3 k3; 3 s2 b3] *
#             (QA)'[2 a_s2 b3; a_e1 a_s3 k3] *
#             B[w4 s4 k4; e4 s3 b4] *
#             (B')[a_e4 a_s3 b4; w4 s4 k4] *
#             A[e2 s5 k5; e5 n5 b5] *
#             (A')[e5 n5 b5; a_e2 a_s5 k5] *
#             QB[1 s6 k6; e6 s5 b6] *
#             (QB')[a_e6 a_s5 b6; 4 a_s6 k6] *
#             A[e4 s7 k7; e7 s6 b7] *
#             (A')[e7 a_s6 b7; a_e4 s7 k7] *
#             A[e6 s8 k8; e8 n8 b8] *
#             (A')[e8 n8 b8; a_e6 s8 k8]
#     end
#     return G
# end

function ntuboundary(P)
    for y in axes(P, 2)
        for x in axes(P, 1)
            T[x, y] = ntu_top(P[x, y - 1], P[x + 1, y - 1])
            B[x, y] = ntu_bottom(P[x, y + 1], P[x + 1, y + 1])

            LC[x, y] = ntu_leftcap(P[x - 1, y])
            RC[x, y] = ntu_rightcap(P[x + 2, y])
        end
        if issmmetric(P)
            break
        end
    end
end

function metric(P, boundary::NTU)
    T = boundary.T
    B = boundary.B
    LC = boundary.LC
    RC = boundary.RC
    return ntu_M.(P, circshift(P, (-1, 0)), T, B, LC, RC)
end

function _ntuboundary(A, B, style::AbstractMetricStyle=GULT())
    T = ntu_top(A, B, style)
    B = ntu_bottom(A, B, style)
    LC = ntu_leftcap(B, style)
    RC = ntu_rightcap(A, style)
    return NTU(T, B, LC, RC)
end

function ntu_M(QL, QR, T, B, LC, RC)
    @tensoropt (
        D1 => D,
        D2 => D,
        D3 => D,
        D4 => D,
        D5 => D,
        D6 => D,
        D7 => D,
        D8 => D,
        r1 => 16 * D,
        r2 => 16 * D,
        r3 => 16 * D,
        r4 => 16 * D,
        r5 => 16 * D,
        r6 => 16 * D,
        r7 => 16 * D,
        r8 => 16 * D,
        d1 => 2,
        d2 => 2,
        d3 => 2,
        d4 => 2,
    ) M[r3 r5; r2 r7] :=
        LC[r2; r1] *
        QL[r1 D5 d1; r2 D1 d2] *
        (QL')[r6 D3 d2; r5 D7 d1] *
        T[D1 D2; D3 D4] *
        B[D6 D5; D8 D7] *
        QR[r3 D6 d3; r4 D2 d4] *
        (QR')[r8 D4 d4; r7 D8 d3] *
        RC[r8; r4]
end

# Function of QL
# function ntu_left(L::AbsTen{1,1}, QL::AbsTen{3,3}, ::GULT=GULT())
#     @tensoropt (
#         D1 => D,
#         D2 => D,
#         D3 => D,
#         D4 => D,
#         r1 => 16 * D,
#         r2 => 16 * D,
#         r3 => 16 * D,
#         r4 => 16 * D,
#         d1 => 2,
#         d2 => 2,
#     ) rv[D1 r4 D4; r2 D2 D3] :=
#         L[r3; r1] * QL[r1 D1 d1; r2 D2 d2] * (QL')[r4 D4 d2; r3 D3 d1]
#     return rv
# end
# function ntu_right(QR, R)
#     return ntu_left(R, permute(QR, (4, 5, 3), (1, 2, 6)))
# end

# Function of QL
# function ntumetric(L::AbsTen{3,3}, R::AbsTen{3,3})
#     @tensoropt (
#         D1 => D,
#         D2 => D,
#         D3 => D,
#         D4 => D,
#         r1 => 16 * D,
#         r2 => 16 * D,
#         r3 => 16 * D,
#         r4 => 16 * D,
#     ) rv[r3 r1; r2 r4] := L[D1 r2 D4; r1 D2 D3] * R[r3 D1 D4; D2 r4 D3]
#     return rv
# end

# Function of A
function ntu_leftcap(B::AbsTen{3,3}, ::GULT=GULT())
    @tensoropt (
        D1 => D, D2 => D, r1 => 16 * D, r2 => 16 * D, r3 => 16 * D, d1 => 2, d2 => 2
    ) L[r3; r2] := B[r1 D1 d1; r2 D2 d2] * (B')[r3 D2 d2; r1 D1 d1]
    return L
end
function ntu_rightcap(A::AbsTen{3,3}, style::AbstractMetricStyle=GULT())
    return ntu_leftcap(permute(A, (4, 2, 3), (1, 5, 6)), style)
end

# Function of B
function ntu_TL(B::AbsTen{3,3}, ::GULT=GULT())
    @tensoropt (
        D1 => D,
        D2 => D,
        D3 => D,
        r1 => 16 * D,
        r2 => 16 * D,
        r3 => 16 * D,
        d1 => 2,
        d2 => 2,
    ) TL[D1 r3; r2 D3] := B[r1 D1 d1; r2 D2 d2] * (B')[r3 D2 d2; r1 D3 d1]
    return TL
end
function ntu_TR(A::AbsTen{3,3}, style::AbstractMetricStyle=GULT())
    rv = ntu_TL(permute(A, (5, 1, 3), (2, 4, 6)), style)
    return rv
end
function ntu_BL(B::AbsTen{3,3}, style::AbstractMetricStyle=GULT())
    rv = ntu_TL(permute(B, (2, 4, 3), (5, 1, 6)), style)
    return rv
end
function ntu_BR(A::AbsTen{3,3}, style::AbstractMetricStyle=GULT())
    rv = ntu_TL(permute(A, (4, 5, 3), (1, 2, 6)), style)
    return rv
end

# Function of A
function ntu_T(TL, TR, ::GULT=GULT())
    @tensoropt (D1 => D, D2 => D, D3 => D, D4 => D, r1 => 16 * D, r2 => 16 * D) T[
        D1 D2
        D3 D4
    ] := TL[D1 r2; r1 D3] * TR[r1 D4; D2 r2]
    return T
end

function ntu_B(BL, BR, style::AbstractMetricStyle=GULT())
    B = ntu_T(BR, BL, style)
    return B
end

function ntu_top(A, B, style::AbstractMetricStyle=GULT())
    TL = ntu_TL(B, style)
    TR = ntu_TR(A, style)
    return ntu_T(TL, TR, style)
end
function ntu_bottom(A, B, style::AbstractMetricStyle=GULT())
    BL = ntu_BL(B, style)
    BR = ntu_BR(A, style)
    return ntu_B(BL, BR, style)
end

function ntustep()
    PL = evolveleft()
    PE = evolveright(PL)

    return rv
end
