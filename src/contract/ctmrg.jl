function onelocalcontract(ctmrg::CornerMethodRuntime, bulk::AbstractUnitCell)
    cor = corners(ctmrg)
    edg = edges(ctmrg)

    el = bulk[1, 1]

    T = similar(el, codomain(el)[1], codomain(el)[2])

    rv = similar(bulk, typeof(T))

    for y in axes(bulk, 2)
        for x in axes(bulk, 1)
            cs = cor[x, y]
            es = getindex.(edg[x, y], 1)
            rv[x, y] = onelocalcontract(_reorderargs(bulk[x, y], cs..., es...)...)
        end
    end

    return rv
end

# Util function for convenience
function _reorderargs(M, C1, C2, C3, C4, T1, T2, T3, T4)
    return C1, T1, C2, T4, M, T2, C4, T3, C3
end

# Contract a tensor representing a trace. That is M_11::AbsTen{0,4}
function tracecontract(C1_00, T1_10, C2_20, T4_01, M_11, T2_21, C4_02, T3_12, C3_22)
    @tensoropt rv =
        C1_00[x1 x3] *
        T1_10[D4; x2 x1] *
        C2_20[x2 x4] *
        T4_01[D3; x5 x3] *
        M_11[D1 D2 D3 D4] *
        T2_21[D1; x4 x6] *
        C4_02[x7 x5] *
        T3_12[D2; x7 x8] *
        C3_22[x8 x6]
    return rv
end

function onelocalcontract(C1_00, T1_10, C2_20, T4_01, M_11, T2_21, C4_02, T3_12, C3_22)
    # Need to specify output structure otherwise @tensoropt will incorrectly guess 
    # index placement.
    physpace = codomain(M_11)[1]
    onelocal = similar(M_11, physpace, physpace)
    return onelocalcontract!(
        onelocal, C1_00, T1_10, C2_20, T4_01, M_11, T2_21, C4_02, T3_12, C3_22
    )
end
function onelocalcontract!(
    dst, C1_00, T1_10, C2_20, T4_01, M_11, T2_21, C4_02, T3_12, C3_22
)
    @tensoropt dst[k; b] =
        C1_00[x1 x3] *
        T1_10[D4; x2 x1] *
        C2_20[x2 x4] *
        T4_01[D3; x5 x3] *
        M_11[k b; D1 D2 D3 D4] *
        T2_21[D1; x4 x6] *
        C4_02[x7 x5] *
        T3_12[D2; x7 x8] *
        C3_22[x8 x6]
    return dst
end
function _metric(left, right, ctmrg::CornerMethodRuntime, bond)
    return ctmrg_trunctensors(left, right, ctmrg, bond)
end
function ctmrg_trunctensors(left, right, ctmrg, bond)
    l, r = bond

    i1 = l[1]:r[1]
    i2 = l[2]

    cs = getboundary(corners(ctmrg), i1, i2)

    @debug "" l[1] r[1] l[2]

    t1s, t2s, t3s, t4s = map(x -> tuple(x...), getboundary(edges(ctmrg), i1, i2))
    # t1s, t2s, t3s, t4s = edge[l[1]:r[1], l[2]]

    return truncmetriccontract(
        cs..., t1s..., t2s..., t3s..., t4s..., left, right, CTMRG(; bonddim=1)
    )
end

function truncmetriccontract(
    C1, C2, C3, C4, T1_1, T1_2, T2, T3_1, T3_2, T4, M1, M2, ::CTMRG
)
    sp = domain(M1)[1] * domain(M1)[1]
    dst = similar(M1, sp, sp)
    return truncmetriccontract!(dst, C1, T1_1, T1_2, C2, T4, M1, M2, T2, C4, T3_1, T3_2, C3)
end

function truncmetriccontract!(
    dst, C1_00, T1_10, T1_20, C2_30, T4_01, M_11, M_21, T2_31, C4_02, T3_12, T3_22, C3_32
)
    @tensoropt (
        k1 => 2,
        k2 => 2,
        b1 => 2,
        b2 => 2,
        x1 => D,
        x2 => D,
        x3 => D,
        x4 => D,
        x5 => D,
        x6 => D,
        x7 => D,
        x8 => D,
        x9 => D,
        x10 => D,
        D1 => D,
        D2 => D,
        D3 => D,
        D4 => D,
        D5 => D,
        D6 => D,
        D7 => D,
        D8 => D,
        E1 => D,
        E2 => D,
        E3 => D,
        E4 => D,
        E5 => D,
        E6 => D,
        E7 => D,
        E8 => D,
    ) dst[D7 E1; D1 E7] =
        C1_00[x1 x4] *
        T1_10[D4 E4; x2 x1] *
        T1_20[D8 E8; x3 x2] *
        C2_30[x3 x5] *
        T4_01[D3 E3; x6 x4] *
        M_11[k1 b1; D1 D2 D3 D4] *
        (M_11')[E1 E2 E3 E4; k1 b1] *
        M_21[k2 b2; D5 D6 D7 D8] *
        (M_21')[E5 E6 E7 E8; k2 b2] *
        T2_31[D5 E5; x5 x7] *
        C4_02[x8 x6] *
        T3_12[D2 E2; x8 x9] *
        T3_22[D6 E6; x9 x10] *
        C3_32[x10 x7]
    return dst
end
