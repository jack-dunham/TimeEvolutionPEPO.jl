abstract type AbstractNeighbourhood end

struct NTUTensors{L,TType<:AbsTen{2,2},CType<:AbsTen{1,1}} <: AbstractNeighbourhood
    T::Lattice{L,TType,Matrix{TType}}
    B::Lattice{L,TType,Matrix{TType}}
    LC::Lattice{L,CType,Matrix{CType}}
    RC::Lattice{L,CType,Matrix{CType}}
end

unpack(ntu::NTUTensors) = ntu.T, ntu.B, ntu.LC, ntu.RC

function Base.getindex(ntu::NTUTensors, bond::Bond)
    l = bond.left
    t = ntu.T[l]
    b = ntu.B[l]
    lc = ntu.LC[l]
    rc = ntu.RC[l]
    return t, b, lc, rc
end

struct NeighbourhoodState{NN<:NTUTensors,MType<:ITC.AbstractNetwork}
    tensors::NN
    network::MType
end

function ntu_initenvironment(network::HilbertSchmidt; kwargs...)
    ntu = ntu_inittensors(network; kwargs...)
    return NeighbourhoodState(ntu, network)
end

function ntu_inittensors(network; f=undef)
    el = ITC.numbertype(network)

    network_top = getfield.(network, :top)
    network_bot = getfield.(network, :bot)

    cod_l = codomain.(circshift(adjoint.(network_bot), (0, 0)))
    cod_r = codomain.(circshift(adjoint.(network_bot), (-1, 0)))
    dom_l = domain.(circshift(network_top, (0, 0)))
    dom_r = domain.(circshift(network_top, (-1, 0)))

    agi = (x, y) -> adjoint(getindex(x, y))

    T = @. TensorMap(f, el, agi(cod_l, 4) * agi(cod_r, 4), agi(dom_l, 4) * agi(dom_r, 4))
    B = @. TensorMap(f, el, agi(cod_r, 2) * agi(cod_l, 2), agi(dom_r, 2) * agi(dom_l, 2))
    LC = @. TensorMap(f, el, agi(cod_l, 3), agi(dom_l, 3))
    RC = @. TensorMap(f, el, agi(cod_r, 1), agi(dom_r, 1))

    return NTUTensors(T, B, LC, RC)
end

function ITC.calculate!(state::NeighbourhoodState)
    return ntuboundary!(state.tensors, state.network)
end

function ntuboundary!(ntu, network)
    T, B, LC, RC = unpack(ntu)

    issym = issymmetric(network.data)

    P = network

    for y in axes(P, 2)
        for x in axes(P, 1)
            if y == 1 || !issym
                #@debug "Calculating: y = $y, issym = $issym "

                ntu_top!(T[x, y], P[x, y - 1], P[x + 1, y - 1])
                ntu_bot!(B[x, y], P[x, y + 1], P[x + 1, y + 1])
                ntu_leftcap!(LC[x, y], P[x - 1, y])
                ntu_rightcap!(RC[x, y], P[x + 2, y])

            elseif issym && y != 1
                #@debug "Copying: y = $y, issym = $issym "

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

function metric(pepo, boundary::NTUTensors)
    T = boundary.T
    B = boundary.B
    LC = boundary.LC
    RC = boundary.RC
    return ntu_M.(pepo, circshift(pepo, (-1, 0)), T, B, LC, RC)
end

function _metric(left, right, ntu::NTUTensors, bond::Bond)
    return ntu_M(left, right, ntu[bond]...)
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
ntu_leftcap!(L, B::TensorPair, style::GULT=GULT()) = ntu_leftcap!(L, B.top, B.bot, style)
function ntu_leftcap!(L, Bt::AbsTen{2,4}, Bb::AbsTen{2,4}, ::GULT=GULT())
    @tensoropt (
        D2 => D, D4 => D, r1u => 16 * D, r1d => 16 * D, r3 => 16 * D, k => 2, b => 2
    ) L[r1d; r1u] = Bt[k b; r1u D2 r3 D4] * (Bb')[r1d D2 r3 D4; k b]
    return L
end
function ntu_rightcap!(R, A::AbsTen{2,4}, style::AbstractUpdateScheme=GULT())
    return ntu_leftcap!(R, permutedom(A, (3, 2, 1, 4)), style)
end

# Function of B
ntu_TL(B::ITC.TensorPair, style::GULT=GULT()) = ntu_TL(B.top, B.bot, style)
function ntu_TL(Bt::AbsTen{2,4}, Bb::AbsTen{2,4}, style::GULT=GULT())
    @tensoropt (
        D2u => D,
        D2d => D,
        D4 => D,
        r1u => 16 * D,
        r1d => 16 * D,
        r3 => 16 * D,
        k => 2,
        b => 2,
    ) TL[r1d D2d; r1u D2u] := Bt[k b; r1u D2u r3 D4] * (Bb')[r1d D2d r3 D4; k b]
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
