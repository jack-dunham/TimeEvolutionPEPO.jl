function get_truncmetric_tensors(fp::FixedPoints, bond::Bond)
    l, r = bond
    return fp.left[l], fp.right[r]
end

function get_truncmetric_tensors(mps::MPS, bond::Bond)
    uind, vind = bond

    AC = ITC.getcentral(mps)
    AR = ITC.getright(mps)

    ac_u = AC[uind]
    ac_d = AC[uind + CartesianIndex(0, 1)]

    ar_u = AR[vind]
    ar_d = AR[vind + CartesianIndex(0, 1)]

    return ac_u, ac_d, ar_u, ar_d
end

# function tracecontract(vumps::VUMPSRuntime, network)
#     AC = ITC.getcentral(vumps.mps)
#     FL = vumps.fixedpoints.left
#     FR = vumps.fixedpoints.right
#     return tracecontract.(FL, FR, AC, circshift(AC, (0, -1)), network)
# end
function onelocalcontract(vumps::VUMPSRuntime, network)
    AC = ITC.getcentral(vumps.mps)
    FL = vumps.fixedpoints.left
    FR = vumps.fixedpoints.right

    return onelocalcontract.(FL, FR, AC, circshift(AC, (0, -1)), network)
end

function _metric(left, right, vumps::VUMPSRuntime, bond::Bond)
    fs = get_truncmetric_tensors(vumps.fixedpoints, bond)

    as = get_truncmetric_tensors(vumps.mps, bond)

    return truncmetriccontract(fs..., as..., left, right)
end

function truncmetric!(dst, vumps::VUMPSRuntime, network::ITC.AbstractNetwork)
    mps = vumps.mps

    AC = ITC.getcentral(mps)
    AR = ITC.getright(mps)

    FL = vumps.fixedpoints.left
    FR = vumps.fixedpoints.right

    # for x in axes(network, 1)
    #     for y in axes(network, 2)
    #         truncmetriccontract!(dst[x,y],FL[x,y],AC[x,y],AC[x,y+1],AR[x+1,y],AR[x+1,y+1],network[x,y],network[x+1,y],FR[x+1,y])
    #     end
    # end
    #
    # return dst

    return truncmetriccontract!.(
        dst,
        FL,
        circshift(FR, (-1, 0)),
        AC,
        circshift(AC, (0, -1)),
        circshift(AR, (-1, 0)),
        circshift(AR, (-1, -1)),
        network,
        circshift(network, (-1, 0)),
    )
end

function tracecontract(FL_00, FR_00, AT_00, AB_01, M_00)
    @tensoropt rv =
        FL_00[D3; x1 x3] *
        AT_00[D4; x2 x1] *
        FR_00[D1; x2 x4] *
        M_00[D1 D2 D3 D4] *
        (AB_01')[x4 x3; D2]
    return rv
end
function onelocalcontract(FL_00, FR_00, AT_00, AB_01, M_00)
    c = codomain(M_00)[1]
    dst = similar(M_00, c, c)
    return onelocalcontract!(dst, FL_00, FR_00, AT_00, AB_01, M_00)
end
function onelocalcontract!(dst, FL_00, FR_00, AT_00, AB_01, M_00)
    @tensoropt dst[k; b] =
        FL_00[D3; x1 x3] *
        AT_00[D4; x2 x1] *
        FR_00[D1; x2 x4] *
        M_00[k b; D1 D2 D3 D4] *
        (AB_01')[x4 x3; D2]
    return dst
end

function truncmetriccontract(FL_00, FR_10, AC_00, AC_01, AR_10, AR_11, M_00, M_10)
    sp = domain(M_00)[1] * domain(M_00)[1]
    dst = similar(FL_00, sp, sp)
    return truncmetriccontract!(dst, FL_00, FR_10, AC_00, AC_01, AR_10, AR_11, M_00, M_10)
end

function truncmetriccontract!(dst, FL_00, FR_10, AC_00, AC_01, AR_10, AR_11, M_00, M_10)
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
        AC_00[D4 E4; x2 x1] *
        AR_10[D8 E8; x3 x2] *
        (AC_01')[x5 x4; D2 E2] *
        (AR_11')[x6 x5; D6 E6] *
        M_00[k1 b1; D1 D2 D3 D4] *
        (M_00')[E1 E2 E3 E4; k1 b1] *
        M_10[k2 b2; D5 D6 D7 D8] *
        (M_10')[E5 E6 E7 E8; k2 b2] *
        FL_00[D3 E3; x1 x4] *
        FR_10[D5 E5; x3 x6]

    return dst
end
