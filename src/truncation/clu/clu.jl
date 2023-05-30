abstract type AbstractNeighbourhood end

struct NearestNeighbours{L,TType<:AbsTen{2,2},CType<:AbsTen{1,1}} <: AbstractNeighbourhood
    T::OnLattice{L,TType,Matrix{TType}}
    B::OnLattice{L,TType,Matrix{TType}}
    LC::OnLattice{L,CType,Matrix{CType}}
    RC::OnLattice{L,CType,Matrix{CType}}
end

unpack(ntu::NearestNeighbours) = ntu.T, ntu.B, ntu.LC, ntu.RC

function Base.getindex(ntu::NearestNeighbours, bond::Bond)
    l = left(bond)
    t = ntu.T[l]
    b = ntu.B[l]
    lc = ntu.LC[l]
    rc = ntu.RC[l]
    return t, b, lc, rc
end

struct NeighbourhoodAlgorithm end

struct NeighbourhoodState{NN<:NearestNeighbours,MType<:ContractableTensors}
    tensors::NN
    bulk::MType
    alg::NeighbourhoodAlgorithm
end

function initenvironment(bulk::HilbertSchmidt, alg::NeighbourhoodAlgorithm; kwargs...)
    ntu = inittensors(detrace(bulk); kwargs...)
    return NeighbourhoodState(ntu, detrace(bulk), alg)
end

function inittensors(bulk; f=undef)
    el = numbertype(bulk)

    cod_l = codomain.(circshift(adjoint.(bulk), (0, 0)))
    cod_r = codomain.(circshift(adjoint.(bulk), (-1, 0)))
    dom_l = domain.(circshift(bulk, (0, 0)))
    dom_r = domain.(circshift(bulk, (-1, 0)))

    agi = (x, y) -> adjoint(getindex(x, y))

    T = @. TensorMap(f, el, agi(cod_l, 4) * agi(cod_r, 4), agi(dom_l, 4) * agi(dom_r, 4))
    B = @. TensorMap(f, el, agi(cod_r, 2) * agi(cod_l, 2), agi(dom_r, 2) * agi(dom_l, 2))
    LC = @. TensorMap(f, el, agi(cod_l, 3), agi(dom_l, 3))
    RC = @. TensorMap(f, el, agi(cod_r, 1), agi(dom_r, 1))
    return NearestNeighbours(T, B, LC, RC)
end

function calculate!(state::NeighbourhoodState)
    return ntuboundary!(state.tensors, state.bulk)
end

function ntuboundary!(ntu, bulk)
    T, B, LC, RC = unpack(ntu)

    issym = issymmetric(bulk)

    P = bulk

    for y in axes(P, 2)
        for x in axes(P, 1)
            if y == 1 || !issym
                @debug "Calculating: y = $y, issym = $issym "

                ntu_top!(T[x, y], P[x, y - 1], P[x + 1, y - 1])
                ntu_bot!(B[x, y], P[x, y + 1], P[x + 1, y + 1])
                ntu_leftcap!(LC[x, y], P[x - 1, y])
                ntu_rightcap!(RC[x, y], P[x + 2, y])

            elseif issym && y != 1
                @debug "Copying: y = $y, issym = $issym "

                copy!(T[x, y], T[x + y - 1, y])
                copy!(B[x, y], B[x + y - 1, y])
                copy!(LC[x, y], LC[x + y - 1, y])
                copy!(RC[x, y], RC[x + y - 1, y])
            else
                @debug "Unexpected branch"
            end
        end
    end

    return ntu
end

function metric(pepo, boundary::NearestNeighbours)
    T = boundary.T
    B = boundary.B
    LC = boundary.LC
    RC = boundary.RC
    return ntu_M.(pepo, circshift(pepo, (-1, 0)), T, B, LC, RC)
end

function metric(pepo, ntu::NearestNeighbours, bond::Bond)
    pl, pr = pepo[bond]
    return ntu_M(pl, pr, ntu[bond]...)
end

function _ntuboundary(A, B, style::AbstractUpdateScheme=GULT())
    T = ntu_top(A, B, style)
    B = ntu_bottom(A, B, style)
    LC = ntu_leftcap(B, style)
    RC = ntu_rightcap(A, style)
    return NTU(T, B, LC, RC)
end

function ntu_M(QL, QR, T, B, LC, RC)
    @tensoropt (
        D1 => 16 * D,
        D2 => D,
        D3 => 16 * D,
        D4 => D,
        D5 => 16 * D,
        D6 => D,
        D7 => 16 * D,
        D8 => D,
        E1 => 16 * D,
        E2 => D,
        E3 => 16 * D,
        E4 => D,
        E5 => 16 * D,
        E6 => D,
        E7 => 16 * D,
        E8 => D,
        k1 => 2,
        b1 => 2,
        k2 => 2,
        b2 => 2,
    ) M[D7 E1; D1 E7] :=
        LC[E3; D3] *
        QL[k1 b1; D1 D2 D3 D4] *
        (QL')[E1 E2 E3 E4; k1 b1] *
        T[E4 E8; D4 D8] *
        B[E6 E2; D6 D2] *
        QR[k2 b2; D5 D6 D7 D8] *
        (QR')[E5 E6 E7 E8; k2 b2] *
        RC[E5; D5]
    return M
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
function ntu_leftcap!(L, B::AbsTen{2,4}, ::GULT=GULT())
    @tensoropt (
        D2 => D, D4 => D, r1u => 16 * D, r1d => 16 * D, r3 => 16 * D, k => 2, b => 2
    ) L[r1d; r1u] = B[k b; r1u D2 r3 D4] * (B')[r1d D2 r3 D4; k b]
    return L
end
function ntu_rightcap!(R, A::AbsTen{2,4}, style::AbstractUpdateScheme=GULT())
    return ntu_leftcap!(R, permutedom(A, (3, 2, 1, 4)), style)
end

# Function of B
function ntu_TL(B::AbsTen{2,4}, ::GULT=GULT())
    @tensoropt (
        D2u => D,
        D2d => D,
        D4 => D,
        r1u => 16 * D,
        r1d => 16 * D,
        r3 => 16 * D,
        k => 2,
        b => 2,
    ) TL[r1d D2d; r1u D2u] := B[k b; r1u D2u r3 D4] * (B')[r1d D2d r3 D4; k b]
    return TL
end
function ntu_TR(A::AbsTen{2,4}, style::AbstractUpdateScheme=GULT())
    rv = ntu_TL(permutedom(A, (2, 3, 4, 1)), style)
    return rv
end
function ntu_BL(B::AbsTen{2,4}, style::AbstractUpdateScheme=GULT())
    rv = ntu_TL(permutedom(B, (4, 1, 2, 3)), style)
    return rv
end
function ntu_BR(A::AbsTen{2,4}, style::AbstractUpdateScheme=GULT())
    rv = ntu_TL(permutedom(A, (3, 4, 1, 2)), style)
    return rv
end

# Function of A
function ntu_T!(T, TL, TR, ::GULT=GULT())
    @tensoropt (Dld => D, Drd => D, Dlu => D, Dru => D, rd => 16 * D, ru => 16 * D) T[
        Dld Drd
        Dlu Dru
    ] = TL[rd Dld; ru Dlu] * TR[Drd rd; Dru ru]
    return T
end

function ntu_B!(B, BL, BR, style::AbstractUpdateScheme=GULT())
    # Swapped
    B = ntu_T!(B, BR, BL, style)
    return B
end

function ntu_top!(Top, A, B, style::AbstractUpdateScheme=GULT())
    TL = ntu_TL(B, style)
    TR = ntu_TR(A, style)
    return ntu_T!(Top, TL, TR, style)
end
function ntu_bot!(Bot, A, B, style::AbstractUpdateScheme=GULT())
    BL = ntu_BL(B, style)
    BR = ntu_BR(A, style)
    return ntu_B!(Bot, BL, BR, style)
end
