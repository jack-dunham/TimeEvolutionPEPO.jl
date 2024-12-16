struct NTUTensors{L,TType<:AbsTen{2,2},CType<:AbsTen{1,1}}
    T::Lattice{L,TType,Matrix{TType}}
    B::Lattice{L,TType,Matrix{TType}}
    LC::Lattice{L,CType,Matrix{CType}}
    RC::Lattice{L,CType,Matrix{CType}}
end

unpack(ntu::NTUTensors) = ntu.T, ntu.B, ntu.LC, ntu.RC

function Base.getindex(ntu::NTUTensors, bond::Bond)
    l = first(bond)
    t = ntu.T[l]
    b = ntu.B[l]
    lc = ntu.LC[l]
    rc = ntu.RC[l]
    return t, b, lc, rc
end

struct NTUState{NN<:NTUTensors,MType<:ITC.AbstractNetwork}
    tensors::NN
    network::MType
end

initneighbourhood(network) = initneighbourhood(undef, network)

function initneighbourhood(f, network::HilbertSchmidt)
    ntu = initialize_neighbourhood_tensors(f, network)
    return NTUState(ntu, network)
end

function initialize_neighbourhood_tensors(f, network)
    el = scalartype(network)

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

calculate_neighbourhood!(state::NTUState) = ntuboundary!(state.tensors, state.network)

function ntuboundary!(ntu, network)
    T, B, LC, RC = unpack(ntu)

    issym = ischeckerboard(network.data)

    P = network

    println(space.(T))

    for y in axes(P, 2)
        for x in axes(P, 1)
            ntu_top!(T[x, y], P[x, y - 1], P[x + 1, y - 1])
            ntu_bot!(B[x, y], P[x, y + 1], P[x + 1, y + 1])
            ntu_leftcap!(LC[x, y], P[x - 1, y])
            ntu_rightcap!(RC[x, y], P[x + 2, y])
        end
        if issym
            copysymmetric!(T)
            copysymmetric!(B)
            copysymmetric!(LC)
            copysymmetric!(RC)
            break
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

function _ntuboundary(A, B)
    T = ntu_top(A, B)
    B = ntu_bottom(A, B)
    LC = ntu_leftcap(B)
    RC = ntu_rightcap(A)
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
ntu_leftcap!(L, B::TensorPair) = ntu_leftcap!(L, B.top, B.bot)
function ntu_leftcap!(L, Bt::AbsTen{2,4}, Bb::AbsTen{2,4})
    @tensoropt (
        D2 => D, D4 => D, r1u => 16 * D, r1d => 16 * D, r3 => 16 * D, k => 2, b => 2
    ) L[r1d; r1u] = Bt[k b; r1u D2 r3 D4] * (Bb')[r1d D2 r3 D4; k b]
    return L
end
function ntu_rightcap!(R, A::AbsTen{2,4})
    return ntu_leftcap!(R, permutedom(A, (3, 2, 1, 4)))
end

# Function of B
ntu_TL(B::ITC.TensorPair) = ntu_TL(B.top, B.bot)
function ntu_TL(Bt::AbsTen{2,4}, Bb::AbsTen{2,4})
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
function ntu_TR(A::AbsTen{2,4})
    rv = ntu_TL(permutedom(A, (2, 3, 4, 1)))
    return rv
end
function ntu_BL(B::AbsTen{2,4})
    rv = ntu_TL(permutedom(B, (4, 1, 2, 3)))
    return rv
end
function ntu_BR(A::AbsTen{2,4})
    rv = ntu_TL(permutedom(A, (3, 4, 1, 2)))
    return rv
end

# Function of A
function ntu_T!(T, TL, TR)
    @tensoropt (Dld => D, Drd => D, Dlu => D, Dru => D, rd => 16 * D, ru => 16 * D) T[
        Dld Drd
        Dlu Dru
    ] = TL[rd Dld; ru Dlu] * TR[Drd rd; Dru ru]
    return T
end

function ntu_B!(B, BL, BR)
    # Swapped
    B = ntu_T!(B, BR, BL)
    return B
end

function ntu_top!(Top, A, B)
    TL = ntu_TL(B)
    TR = ntu_TR(A)
    return ntu_T!(Top, TL, TR)
end
function ntu_bot!(Bot, A, B)
    BL = ntu_BL(B)
    BR = ntu_BR(A)
    return ntu_B!(Bot, BL, BR)
end

# function ntu_total_join(QL, QR, LL, RR, TL, TR,  BL, BR)
#     TensorKit.TensorOperations.@tensoropt_verbose (k1 => 2,
#         b1 => 2,
#         k2 => 2,
#         b2 => 2,
#         k3 => 2,
#         b3 => 2,
#         k4 => 2,
#         b4 => 2,
#         k5 => 2,
#         b5 => 2,
#         k6 => 2,
#         b6 => 2,
#         k7 => 2,
#         b7 => 2,
#         k8 => 2,
#         b8 => 2,
#         D => D,
#         D2 => D,
#         D3 => D,
#         D4 => D,
#         D5 => D,
#         D6 => D,
#         D8 => D,
#         F => D,
#         F2 => D,
#         F3 => D,
#         F4 => D,
#         F5 => D,
#         F6 => D,
#         F8 => D,
#         X1 => D,
#         X2 => D,
#         X3 => D,
#         X4 => D,
#         X5 => D,
#         X6 => D,
#         X7 => D,
#         X8 => D,
#         X9 => D,
#         X10 => D,
#         X11 => D,
#         X12 => D,
#         X13 => D,
#         X14 => D,
#         T1 => D,
#         T3 => D,
#         T5 => D,
#         T7 => D,
#     ) M =
#         LL[k1 b1; D3 X1 X2 X3] *
#         (LL')[F3 X1 X2 X3; k1 b1] *
#         TL[k2 b2; T1 D4 X4 X5] *
#         (TL')[T3 F4 X4 X5; k2 b2] *
#         TR[k3 b3; X6 D8 T1 X7]* 
#         (TR')[X6 F8 T3 X7; k3 b3] *
#         QL[k4 b4; D D2 D3 D4] *
#         (QL')[F F2 F3 F4; k4 b4] *
#         QR[k5 b5; D5 D6 D D8] *
#         (QR')[F5 F6 F F8; k5 b5] *
#         BL[k6 b6; T5 X8 X9 D2]*
#         (BL')[T7 X8 X9 F2; k6 b6]*
#         BR[k7 b7; X10 X11 T5 D6] *
#         (BR')[ X10 X11 T7 F6; k7 b7]*
#         RR[k8 b8; X12 X13 D5 X14] *
#         (RR')[ X12 X13 F5 X14; k8 b8]
#     return M
# end
function ntu_total_low(pair, lattice, bond)
    QL, QR = pair
    LL, _ = lattice[prevbond(bond)]
    _, RR = lattice[nextbond(bond)]

    i1, i2 = bond

    incr = CartesianIndex(0, 1)

    tbond = Bond(i1 - incr, i2 - incr)
    bbond = Bond(i1 + incr, i2 + incr)

    TL, TR = lattice[tbond]
    BL, BR = lattice[bbond]

    return ntu_total_low(QL, QR, LL, RR, TL, TR, BL, BR)
end
function ntu_total_low(lattice, bond)
    QL, QR = lattice[bond]
    LL, _ = lattice[prevbond(bond)]
    _, RR = lattice[nextbond(bond)]

    i1, i2 = bond

    incr = CartesianIndex(0, 1)

    tbond = Bond(i1 - incr, i2 - incr)
    bbond = Bond(i1 + incr, i2 + incr)

    TL, TR = lattice[tbond]
    BL, BR = lattice[bbond]

    return ntu_total_low(QL, QR, LL, RR, TL, TR, BL, BR)
end

function ntu_total(pair, lattice, bond)
    QL, QR = pair
    LL, _ = lattice[prevbond(bond)]
    _, RR = lattice[nextbond(bond)]

    i1, i2 = bond

    incr = CartesianIndex(0, 1)

    tbond = Bond(i1 - incr, i2 - incr)
    bbond = Bond(i1 + incr, i2 + incr)

    TL, TR = lattice[tbond]
    BL, BR = lattice[bbond]

    return ntu_total(QL, QR, LL, RR, TL, TR, BL, BR)
end

function ntu_total_infidelity(pair, lattice, bond; trunc=notrunc())

    QL, QR = pair
    LL, _ = lattice[prevbond(bond)]
    _, RR = lattice[nextbond(bond)]

    i1, i2 = bond

    incr = CartesianIndex(0, 1)

    tbond = Bond(i1 - incr, i2 - incr)
    bbond = Bond(i1 + incr, i2 + incr)

    TL, TR = lattice[tbond]
    BL, BR = lattice[bbond]

    denom2 = ntu_total_join(QL, QR, LL, RR, TL, TR, BL, BR)

    rv = function(A)
        u,s,v = tsvd(A; trunc=trunc)

        numerator = ntu_total_join_svd(QL, QR, LL, RR, TL, TR, BL, BR, u, s, v)
        denom1 = ntu_total_join_both_svd(QL, QR, LL, RR, TL, TR, BL, BR, u, s, v)

        return 1 - real(numerator / sqrt(denom1 * denom2))
    end

    return rv
end

function ntu_total_wtg(pair, lattice, bond, iso)

    LL, _ = lattice[prevbond(bond)]
    _, RR = lattice[nextbond(bond)]

    i1, i2 = bond

    incr = CartesianIndex(0, 1)

    tbond = Bond(i1 - incr, i2 - incr)
    bbond = Bond(i1 + incr, i2 + incr)

    TL, TR = lattice[tbond]
    BL, BR = lattice[bbond]

    QL, QR = map(copy, lattice[bond])

    truncateto!(QL => QR, pair, iso.u => iso.v; axis=1)

    yiso = ntu_total(QL, QR, LL, RR, TL, TR, BL, BR)

    return yiso
end

function ntu_total_low(QL, QR, LL, RR, TL, TR, BL, BR)
    @tensoropt (
        k1 => 2,
        b1 => 2,
        k2 => 2,
        b2 => 2,
        k3 => 2,
        b3 => 2,
        k4 => 2,
        b4 => 2,
        k5 => 2,
        b5 => 2,
        k6 => 2,
        b6 => 2,
        k7 => 2,
        b7 => 2,
        k8 => 2,
        b8 => 2,
        D1 => D,
        D2 => D,
        D3 => D,
        D4 => D,
        D5 => D,
        D6 => D,
        D7 => D,
        D8 => D,
        F1 => D,
        F2 => D,
        F3 => D,
        F4 => D,
        F5 => D,
        F6 => D,
        F7 => D,
        F8 => D,
        X1 => D,
        X2 => D,
        X3 => D,
        X4 => D,
        X5 => D,
        X6 => D,
        X7 => D,
        X8 => D,
        X9 => D,
        X10 => D,
        X11 => D,
        X12 => D,
        X13 => D,
        X14 => D,
        X15 => D,
        X16 => D,
        X17 => D,
        X18 => D,
    ) M[D7 F1; D1 F7] :=
        LL[k1 b1; D3 X1 X2 X3] *
        (LL')[F3 X1 X2 X3; k1 b1] *
        TL[k2 b2; X4 D4 X5 X6] *
        (TL')[X4 F4 X5 X6; k2 b2] *
        TR[k3 b3; X7 D8 X8 X9] *
        (TR')[X7 F8 X8 X9; k3 b3] *
        QL[k4 b4; D1 D2 D3 D4] *
        (QL')[F1 F2 F3 F4; k4 b4] *
        QR[k5 b5; D5 D6 D7 D8] *
        (QR')[F5 F6 F7 F8; k5 b5] *
        BL[k6 b6; X10 X11 X12 D2] *
        (BL')[X10 X11 X12 F2; k6 b6] *
        BR[k7 b7; X13 X14 X15 D6] *
        (BR')[X13 X14 X15 F6; k7 b7] *
        RR[k8 b8; X16 X17 D5 X18] *
        (RR')[X16 X17 F5 X18; k8 b8]
    return M
end

# function ntu_total_low_join(QL, QR, LL, RR, TL, TR,  BL, BR)
#     TensorKit.TensorOperations.@tensoropt_verbose (k1 => 2,
#         b1 => 2,
#         k2 => 2,
#         b2 => 2,
#         k3 => 2,
#         b3 => 2,
#         k4 => 2,
#         b4 => 2,
#         k5 => 2,
#         b5 => 2,
#         k6 => 2,
#         b6 => 2,
#         k7 => 2,
#         b7 => 2,
#         k8 => 2,
#         b8 => 2,
#         D1 => D,
#         D2 => D,
#         D3 => D,
#         D4 => D,
#         D5 => D,
#         D6 => D,
#         D7 => D,
#         D8 => D,
#         F2 => D,
#         F3 => D,
#         F4 => D,
#         F5 => D,
#         F6 => D,
#         F8 => D,
#         F => D,
#         X1 => D,
#         X2 => D,
#         X3 => D,
#         X4 => D,
#         X5 => D,
#         X6 => D,
#         X7 => D,
#         X8 => D,
#         X9 => D,
#         X10 => D,
#         X11 => D,
#         X12 => D,
#         X13 => D,
#         X14 => D,
#         X15 => D,
#         X16 => D,
#         X17 => D,
#         X18 => D,
#     ) M[D7 ; D1] :=
#         LL[k1 b1; D3 X1 X2 X3] *
#         (LL')[F3 X1 X2 X3; k1 b1] *
#         TL[k2 b2; X4 D4 X5 X6] *
#         (TL')[X4 F4 X5 X6; k2 b2] *
#         TR[k3 b3; X7 D8 X8 X9]* 
#         (TR')[X7 F8 X8 X9; k3 b3] *
#         QL[k4 b4; D1 D2 D3 D4] *
#         (QL')[F F2 F3 F4; k4 b4] *
#         QR[k5 b5; D5 D6 D7 D8] *
#         (QR')[F5 F6 F F8; k5 b5] *
#         BL[k6 b6; X10 X11 X12 D2]*
#         (BL')[X10 X11 X12 F2; k6 b6]*
#         BR[k7 b7; X13 X14 X15 D6] *
#         (BR')[ X13 X14 X15 F6; k7 b7]*
#         RR[k8 b8; X16 X17 D5 X18] *
#         (RR')[ X16 X17 F5 X18; k8 b8]
#     return M
# end

function ntu_total(QL, QR, LL, RR, TL, TR,  BL, BR)
    @tensoropt M[D7 F1 ; D1 F7] :=
        LL[k1 b1; D3 X1 X2 X3] *
        (LL')[F3 X1 X2 X3; k1 b1] *
        TL[k2 b2; T1 D4 X4 X5] *
        (TL')[T3 F4 X4 X5; k2 b2] *
        TR[k3 b3; X6 D8 T1 X7]*
        (TR')[X6 F8 T3 X7; k3 b3] *
        QL[k4 b4; D1 D2 D3 D4] *
        (QL')[F1 F2 F3 F4; k4 b4] *
        QR[k5 b5; D5 D6 D7 D8] *
        (QR')[F5 F6 F7 F8; k5 b5] *
        BL[k6 b6; T5 X8 X9 D2]*
        (BL')[T7 X8 X9 F2; k6 b6]*
        BR[k7 b7; X10 X11 T5 D6] *
        (BR')[ X10 X11 T7 F6; k7 b7]*
        RR[k8 b8; X12 X13 D5 X14] *
        (RR')[ X12 X13 F5 X14; k8 b8]
    return M
end

function ntu_total_no_opt(QL, QR, LL, RR, TL, TR,  BL, BR)
    @tensor M[D7 F1 ; D1 F7] :=
        LL[k1 b1; D3 X1 X2 X3] *
        (LL')[F3 X1 X2 X3; k1 b1] *
        TL[k2 b2; T1 D4 X4 X5] *
        (TL')[T3 F4 X4 X5; k2 b2] *
        TR[k3 b3; X6 D8 T1 X7]*
        (TR')[X6 F8 T3 X7; k3 b3] *
        QL[k4 b4; D1 D2 D3 D4] *
        (QL')[F1 F2 F3 F4; k4 b4] *
        QR[k5 b5; D5 D6 D7 D8] *
        (QR')[F5 F6 F7 F8; k5 b5] *
        BL[k6 b6; T5 X8 X9 D2]*
        (BL')[T7 X8 X9 F2; k6 b6]*
        BR[k7 b7; X10 X11 T5 D6] *
        (BR')[ X10 X11 T7 F6; k7 b7]*
        RR[k8 b8; X12 X13 D5 X14] *
        (RR')[ X12 X13 F5 X14; k8 b8]
    return M
end


# function ntu_total(QL, QR, LL, RR, TL, TR,  BL, BR)
#     @tensoropt (k1 => 2,
#         b1 => 2,
#         k2 => 2,
#         b2 => 2,
#         k3 => 2,
#         b3 => 2,
#         k4 => 2,
#         b4 => 2,
#         k5 => 2,
#         b5 => 2,
#         k6 => 2,
#         b6 => 2,
#         k7 => 2,
#         b7 => 2,
#         k8 => 2,
#         b8 => 2,
#         D1 => D,
#         D2 => D,
#         D3 => D,
#         D4 => D,
#         D5 => D,
#         D6 => D,
#         D7 => D,
#         D8 => D,
#         F1 => D,
#         F2 => D,
#         F3 => D,
#         F4 => D,
#         F5 => D,
#         F6 => D,
#         F7 => D,
#         F8 => D,
#         X1 => D,
#         X2 => D,
#         X3 => D,
#         X4 => D,
#         X5 => D,
#         X6 => D,
#         X7 => D,
#         X8 => D,
#         X9 => D,
#         X10 => D,
#         X11 => D,
#         X12 => D,
#         X13 => D,
#         X14 => D,
#         T1 => D,
#         T3 => D,
#         T5 => D,
#         T7 => D,
#     ) M[D7 F1 ; D1 F7] :=
#         LL[k1 b1; D3 X1 X2 X3] *
#         (LL')[F3 X1 X2 X3; k1 b1] *
#         TL[k2 b2; T1 D4 X4 X5] *
#         (TL')[T3 F4 X4 X5; k2 b2] *
#         TR[k3 b3; X6 D8 T1 X7]*
#         (TR')[X6 F8 T3 X7; k3 b3] *
#         QL[k4 b4; D1 D2 D3 D4] *
#         (QL')[F1 F2 F3 F4; k4 b4] *
#         QR[k5 b5; D5 D6 D7 D8] *
#         (QR')[F5 F6 F7 F8; k5 b5] *
#         BL[k6 b6; T5 X8 X9 D2]*
#         (BL')[T7 X8 X9 F2; k6 b6]*
#         BR[k7 b7; X10 X11 T5 D6] *
#         (BR')[ X10 X11 T7 F6; k7 b7]*
#         RR[k8 b8; X12 X13 D5 X14] *
#         (RR')[ X12 X13 F5 X14; k8 b8]
#     return M
# end

function ntu_total_join_svd(QL, QR, LL, RR, TL, TR,  BL, BR, u, s, v)
    @tensoropt (k1 => 2,
        b1 => 2,
        k2 => 2,
        b2 => 2,
        k3 => 2,
        b3 => 2,
        k4 => 2,
        b4 => 2,
        k5 => 2,
        b5 => 2,
        k6 => 2,
        b6 => 2,
        k7 => 2,
        b7 => 2,
        k8 => 2,
        b8 => 2,
        DA => D,
        DX => D,
        DXX => D,
        DB => D,
        D2 => D,
        D3 => D,
        D4 => D,
        D5 => D,
        D6 => D,
        D8 => D,
        F => D,
        F2 => D,
        F3 => D,
        F4 => D,
        F5 => D,
        F6 => D,
        F8 => D,
        X1 => D,
        X2 => D,
        X3 => D,
        X4 => D,
        X5 => D,
        X6 => D,
        X7 => D,
        X8 => D,
        X9 => D,
        X10 => D,
        X11 => D,
        X12 => D,
        X13 => D,
        X14 => D,
        T1 => D,
        T3 => D,
        T5 => D,
        T7 => D,
    ) M = LL[k1 b1; D3 X1 X2 X3] *
        (LL')[F3 X1 X2 X3; k1 b1] *
        TL[k2 b2; T1 D4 X4 X5] *
        (TL')[T3 F4 X4 X5; k2 b2] *
        TR[k3 b3; X6 D8 T1 X7]*
        (TR')[X6 F8 T3 X7; k3 b3] *
        QL[k4 b4; DA D2 D3 D4] *
        (QL')[F F2 F3 F4; k4 b4] *
        QR[k5 b5; D5 D6 DB D8] *
        (QR')[F5 F6 F F8; k5 b5] *
        BL[k6 b6; T5 X8 X9 D2]*
        (BL')[T7 X8 X9 F2; k6 b6]*
        BR[k7 b7; X10 X11 T5 D6] *
        (BR')[ X10 X11 T7 F6; k7 b7]*
        RR[k8 b8; X12 X13 D5 X14] *
        (RR')[ X12 X13 F5 X14; k8 b8]*
        u[DA; DX] * s[DX;DXX] * v[DXX; DB]
    return M
end

function ntu_total_join_both_svd(QL, QR, LL, RR, TL, TR,  BL, BR, u, s, v)
    @tensoropt (k1 => 2,
        b1 => 2,
        k2 => 2,
        b2 => 2,
        k3 => 2,
        b3 => 2,
        k4 => 2,
        b4 => 2,
        k5 => 2,
        b5 => 2,
        k6 => 2,
        b6 => 2,
        k7 => 2,
        b7 => 2,
        k8 => 2,
        b8 => 2,
        DA => D,
        DX => D,
        DXX => D,
        DB => D,
        D2 => D,
        D3 => D,
        D4 => D,
        D5 => D,
        D6 => D,
        D8 => D,
        FA => D,
        FX => D,
        FXX => D,
        FB => D,
        F2 => D,
        F3 => D,
        F4 => D,
        F5 => D,
        F6 => D,
        F8 => D,
        X1 => D,
        X2 => D,
        X3 => D,
        X4 => D,
        X5 => D,
        X6 => D,
        X7 => D,
        X8 => D,
        X9 => D,
        X10 => D,
        X11 => D,
        X12 => D,
        X13 => D,
        X14 => D,
        T1 => D,
        T3 => D,
        T5 => D,
        T7 => D,
    ) M = LL[k1 b1; D3 X1 X2 X3] *
        (LL')[F3 X1 X2 X3; k1 b1] *
        TL[k2 b2; T1 D4 X4 X5] *
        (TL')[T3 F4 X4 X5; k2 b2] *
        TR[k3 b3; X6 D8 T1 X7]*
        (TR')[X6 F8 T3 X7; k3 b3] *
        QL[k4 b4; DA D2 D3 D4] *
        (QL')[FA F2 F3 F4; k4 b4] *
        QR[k5 b5; D5 D6 DB D8] *
        (QR')[F5 F6 FB F8; k5 b5] *
        BL[k6 b6; T5 X8 X9 D2]*
        (BL')[T7 X8 X9 F2; k6 b6]*
        BR[k7 b7; X10 X11 T5 D6] *
        (BR')[ X10 X11 T7 F6; k7 b7]*
        RR[k8 b8; X12 X13 D5 X14] *
        (RR')[ X12 X13 F5 X14; k8 b8]*
        u[DA; DX] * s[DX;DXX] * v[DXX; DB]*
        (u')[FX; FA] * (s')[FXX;FX] * (v')[FB; FXX]
    return M
end
function ntu_total_join(QL, QR, LL, RR, TL, TR,  BL, BR)
    @tensoropt (k1 => 2,
        b1 => 2,
        k2 => 2,
        b2 => 2,
        k3 => 2,
        b3 => 2,
        k4 => 2,
        b4 => 2,
        k5 => 2,
        b5 => 2,
        k6 => 2,
        b6 => 2,
        k7 => 2,
        b7 => 2,
        k8 => 2,
        b8 => 2,
        D => D,
        D2 => D,
        D3 => D,
        D4 => D,
        D5 => D,
        D6 => D,
        D8 => D,
        F => D,
        F2 => D,
        F3 => D,
        F4 => D,
        F5 => D,
        F6 => D,
        F8 => D,
        X1 => D,
        X2 => D,
        X3 => D,
        X4 => D,
        X5 => D,
        X6 => D,
        X7 => D,
        X8 => D,
        X9 => D,
        X10 => D,
        X11 => D,
        X12 => D,
        X13 => D,
        X14 => D,
        T1 => D,
        T3 => D,
        T5 => D,
        T7 => D,
    ) M = LL[k1 b1; D3 X1 X2 X3] *
        (LL')[F3 X1 X2 X3; k1 b1] *
        TL[k2 b2; T1 D4 X4 X5] *
        (TL')[T3 F4 X4 X5; k2 b2] *
        TR[k3 b3; X6 D8 T1 X7]*
        (TR')[X6 F8 T3 X7; k3 b3] *
        QL[k4 b4; D D2 D3 D4] *
        (QL')[F F2 F3 F4; k4 b4] *
        QR[k5 b5; D5 D6 D D8] *
        (QR')[F5 F6 F F8; k5 b5] *
        BL[k6 b6; T5 X8 X9 D2]*
        (BL')[T7 X8 X9 F2; k6 b6]*
        BR[k7 b7; X10 X11 T5 D6] *
        (BR')[ X10 X11 T7 F6; k7 b7]*
        RR[k8 b8; X12 X13 D5 X14] *
        (RR')[ X12 X13 F5 X14; k8 b8]
    return M
end
# function ntu_total(QL, QR, ML, MR, LL, RR, MLL, MRR, TL, TR, MTL, MTR, BL, BR, MBL, MBR)
#     TensorKit.@tensoropt (
#         D1 => D,
#         D2 => D,
#         D3 => D,
#         D4 => D,
#         D5 => D,
#         D6 => D,
#         D7 => D,
#         D8 => D,
#         E1 => D,
#         E2 => D,
#         E3 => D,
#         E4 => D,
#         E5 => D,
#         E6 => D,
#         E7 => D,
#         E8 => D,
#         F1 => D,
#         F2 => D,
#         F3 => D,
#         F4 => D,
#         F5 => D,
#         F6 => D,
#         F7 => D,
#         F8 => D,
#         G1 => D,
#         G2 => D,
#         G3 => D,
#         G4 => D,
#         G5 => D,
#         G6 => D,
#         G7 => D,
#         G8 => D,
#         T1 => D,
#         T2 => D,
#         T3 => D,
#         T4 => D,
#         T5 => D,
#         T6 => D,
#         T7 => D,
#         T8 => D,
#         X1 => D,
#         X2 => D,
#         X3 => D,
#         X4 => D,
#         X5 => D,
#         X6 => D,
#         X7 => D,
#         X8 => D,
#         X9 => D,
#         X10 => D,
#         X11 => D,
#         X12 => D,
#         X13 => D,
#         X14 => D,
#         Y1 => D,
#         Y2 => D,
#         Y3 => D,
#         Y4 => D,
#         Y5 => D,
#         Y6 => D,
#         Y7 => D,
#         Y8 => D,
#         Y9 => D,
#         Y10 => D,
#         Y11 => D,
#         Y12 => D,
#         Y13 => D,
#         Y14 => D,
#         k1 => 2,
#         b1 => 2,
#         k2 => 2,
#         b2 => 2,
#         k3 => 2,
#         b3 => 2,
#         k4 => 2,
#         b4 => 2,
#         k5 => 2,
#         b5 => 2,
#         k6 => 2,
#         b6 => 2,
#         kk1 => 2,
#         kk2 => 2,
#         kk3 => 2,
#         kk4 => 2,
#         kk5 => 2,
#         kk6 => 2,
#         kk7 => 2,
#         kk8 => 2,
#         kk9 => 2,
#         kk10 => 2,
#         kk11 => 2,
#         kk12 => 2,
#         kk13 => 2,
#         kk14 => 2,
#         kk15 => 2,
#         kk16 => 2,
#         kk17 => 2,
#         kk18 => 2,
#         bb1 => 2,
#         bb2 => 2,
#         bb3 => 2,
#         bb4 => 2,
#         bb5 => 2,
#         bb6 => 2,
#         bb7 => 2,
#         bb8 => 2,
#         bb9 => 2,
#         bb10 => 2,
#         bb11 => 2,
#         bb12 => 2,
#         bb13 => 2,
#         bb14 => 2,
#         bb15 => 2,
#         bb16 => 2,
#         bb17 => 2,
#         bb18 => 2,
#     ) M[D7 E1 F7 G1; D1 E7 F1 G7] :=
#         LL[kk1 bb1; D3 X1 X2 X3] * MLL[kk2 bb2 kk1 bb1; E3 Y1 Y2 Y3] *
#         (LL')[F3 X1 X2 X3; kk3 bb3] * (MLL')[G3 Y1 Y2 Y3 ;kk2 bb2 kk3 bb3] *
#         TL[kk4 bb4; T1 D4 X4 X5] * MTL[kk5 bb5 kk4 bb4;T2 E4 Y4 Y5] *
#         (TL')[T3 F4 X4 X5; kk6 bb6] * (MTL')[T4 G4 Y4 Y5 ;kk5 bb5 kk6 bb6] *
#         TR[kk7 bb7; X6 D8 T1 X7] * MTR[kk8 bb8 kk7 bb7; Y6 E8 T2 Y7 ] *
#         (TR')[X6 F8 T3 X7; kk9 bb9] * (MTR')[Y6 G8 T4 Y7; kk8 bb8 kk9 bb9] *
#         QL[k1 b1; D1 D2 D3 D4] * ML[k2 b2 k1 b1; E1 E2 E3 E4] *
#         (QL')[F1 F2 F3 F4; k3 b3] * (ML')[G1 G2 G3 G4 ;k2 b2 k3 b3] *
#         QR[k4 b4; D5 D6 D7 D8] * MR[k5 b5 k4 b4; E5 E6 E7 E8] *
#         (QR')[F5 F6 F7 F8; k6 b6] * (MR')[G5 G6 G7 G8; k5 b5 k6 b6] *
#         BL[kk10 bb10; T5 X8 X9 D2] * MBL[kk11 bb11 kk10 bb10; T6 Y8 Y9 E2] *
#         (BL')[T7 X8 X9 F2; kk12 bb12] * (MBL')[ T8 Y8 Y9 G2 ;kk11 bb11 kk12 bb12] *
#         BR[kk13 bb13; X10 X11 T5 D6] * MBR[kk14 bb14 kk13 bb13; Y10 Y11 T6 E6] *
#         (BR')[ X10 X11 T7 F6; kk15 bb15] * (MBR')[ Y10 Y11 T8 G6; kk14 bb14 kk15 bb15] *
#         RR[kk16 bb16; X12 X13 D5 X14] * MRR[kk17 bb17 kk16 bb16;Y12 Y13 E5 Y14] *
#         (RR')[ X12 X13 F5 X14; kk18 bb18] * (MRR')[Y12 Y13 G5 Y14 ; kk17 bb17 kk18 bb18]
#     return M
# end
