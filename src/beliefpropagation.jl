struct MessageTensors{T}
    east::T
    south::T
    west::T
    north::T
end

struct VidalIsometries{T<:AbsTen{2,1}}
    east::T
    south::T
    west::T
    north::T
end
struct VidalWeights{T<:AbsTen{1,1}}
    east::T
    south::T
    west::T
    north::T
end

const AbstractTensorsOnBond = Union{MessageTensors,VidalIsometries,VidalWeights}

unpack(mts::AbstractTensorsOnBond) = (mts.east, mts.south, mts.west, mts.north)
Base.getindex(mts::AbstractTensorsOnBond, i::Int) = unpack(mts)[i]

function setvidal!(truncators, u, s, v, bond)
    vidals = truncators.isometries
    weights = truncators.weights
    ax = getaxis(bond)

    src_iso, snk_iso = vidals[bond]
    src_w, snk_w = weights[bond]

    if ax == 1
        copy!(src_iso.east, u * sqrt(s))
        copy!(src_w.east, s)
        copy!(snk_iso.west, _transpose(sqrt(s) * v))
        copy!(snk_w.west, s)
    elseif ax == 2
        copy!(src_iso.south, u * sqrt(s))
        copy!(src_w.south, s)
        copy!(snk_iso.north, _transpose(sqrt(s) * v))
        copy!(snk_w.north, s)
    end

    return truncators
end

function getmessages(mts::Lattice{G,<:MessageTensors}, bond::Bond) where {G}
    m1, m2 = mts[bond]

    axis = getaxis(bond)

    if axis == 1
        return m1.east, m2.west
    elseif axis == 2
        return m1.south, m2.north
    end
end

function update_message_tensor!(Mw_new, Te, Ms, Mw, Mn)
    if Mw_new === Mw
        Mw = copy(Mw)
    end
    @tensoropt (
        k => 2,
        b => 2,
        e1 => D,
        e2 => D,
        w1 => D,
        w2 => D,
        s1 => D,
        s2 => D,
        n1 => D,
        n2 => D,
    ) Mw_new[e1; e2] =
        Te[k b; e1 s1 w1 n1] *
        (Te')[e2 s2 w2 n2; k b] *
        Ms[s1; s2] *
        Mw[w1; w2] *
        Mn[n1; n2]
    rmul!(Mw_new, 1 / tr(Mw_new))
    return Mw_new
end
function remessage!(mts, p)
    updatewest!(mts[x + 1, y], mts[x, y], p[x, y])
    updatenorth!(mts[x, y + 1], mts[x, y], p[x, y])
    updateeast!(mts[x - 1, y], mts[x, y], p[x, y])
    updatesouth!(mts[x, y - 1], mts[x, y], p[x, y])
    return mts
end

# function update_message_tensor_opt!(Mw_new, Te, Ms, Mw, Mn, P)
#     if Mw_new === Mw
#         Mw = copy(Mw)
#     end
#     @tensoropt (
#         k => 2,
#         b => 2,
#         kk => 2,
#         bb => 2,
#         kkk => 2,
#         bbb => 2,
#         e1 => D,
#         e2 => D,
#         w1 => D,
#         w2 => D,
#         s1 => D,
#         s2 => D,
#         n1 => D,
#         n2 => D,
#         ee1 => D,
#         ee2 => D,
#         ww1 => D,
#         ww2 => D,
#         ss1 => D,
#         ss2 => D,
#         nn1 => D,
#         nn2 => D,
#     ) Mw_new[e1 ee1; e2 ee2] =
#         Te[k b; e1 s1 w1 n1] *
#         P[kk bb k b; ee1 ss1 ww1 nn1] *
#         (P')[ee2 ss2 ww2 nn2; kk bb kkk bbb] *
#         (Te')[e2 s2 w2 n2; kkk bbb] *
#         Ms[s1 ss1; s2 ss2] *
#         Mw[w1 ww1; w2 ww2] *
#         Mn[n1 nn1; n2 nn2]
#     rmul!(Mw_new, 1 / tr(Mw_new))
#     return Mw_new
# end

function update_message_tensor_opt!(Mw_new, Te, Ms, Mw, Mn, P)
    if Mw_new === Mw
        Mw = copy(Mw)
    end
    @tensoropt (
        k => 2,
        b => 2,
        kk => 2,
        bb => 2,
        kkk => 2,
        bbb => 2,
        e1 => D,
        e2 => D,
        w1 => D,
        w2 => D,
        s1 => D,
        s2 => D,
        n1 => D,
        n2 => D,
        ee1 => D,
        ee2 => D,
        ww1 => D,
        ww2 => D,
        ss1 => D,
        ss2 => D,
        nn1 => D,
        nn2 => D,
    ) Mw_new[e1 ee1; e2 ee2] =
        Te[k b; e1 s1 w1 n1] *
        P[kk bb; k b ee1 ss1 ww1 nn1] *
        conj(Te[kkk bbb; e2 s2 w2 n2] * P[kk bb kkk bbb; ee2 ss2 ww2 nn2]) *
        Ms[s1 ss1; s2 ss2] *
        Mw[w1 ww1; w2 ww2] *
        Mn[n1 nn1; n2 nn2]
    rmul!(Mw_new, 1 / tr(Mw_new))
    return Mw_new
end
function vidal_gauge_test!(Te, P, isoms, weights, dir)
    Ws, Ww, Wn = sqrt_weights(weights, dir)
    Us, Uw, Un = get_matrices_except_dir(isoms, dir)

    Us = Us * Ws
    Uw = Uw * _transpose(Ww)
    Un = Un * _transpose(Wn)

    @tensoropt (
        k => 2,
        b => 2,
        kk => 2,
        bb => 2,
        kkk => 2,
        bbb => 2,
        e1 => D,
        e2 => D,
        w1 => D,
        w2 => D,
        s1 => D,
        s2 => D,
        n1 => D,
        n2 => D,
        ee1 => D,
        ee2 => D,
        ww1 => D,
        ww2 => D,
        ss1 => D,
        ss2 => D,
        nn1 => D,
        nn2 => D,
    ) out[e1 ee1; e2 ee2] :=
        Te[k b; e1 s1 w1 n1] *
        P[kk bb; k b ee1 ss1 ww1 nn1] *
        conj(
            Te[kkk bbb; e2 s2 w2 n2] *
            P[kk bb kkk bbb; ee2 ss2 ww2 nn2] *
            Us[s2 ss2; sx] *
            Uw[w2 ww2; wx] *
            Un[n2 nn2; nx],
        ) *
        Us[s1 ss1; sx] *
        Uw[w1 ww1; wx] *
        Un[n1 nn1; nx]
    rmul!(out, 1 / tr(out))
    return out
end

# inits Me
function initmessagetensor(Te)
    c = domain(Te)[1] # east bond
    Me = TensorMap(rand, eltype(Te), c, c)
    return normalize(Me * Me')
end
function initmessagetensor(Te, P)
    c = domain(Te)[1] # east bond
    p = domain(P)[1] # east bond
    Me = TensorMap(rand, eltype(Te), c * p, c * p)
    return normalize(Me * Me')
end

function initmessages(T)
    p = (2, 3, 4, 1) # anti clockwise rotation

    Te = T
    Me = initmessagetensor(Te)

    Ts = permutedom(Te, p)
    Ms = initmessagetensor(Ts)

    Tw = permutedom(Ts, p)
    Mw = initmessagetensor(Tw)

    Tn = permutedom(Tw, p)
    Mn = initmessagetensor(Tn)

    return MessageTensors(Me, Ms, Mw, Mn)
end
function initmessages(T, P)
    p = (2, 3, 4, 1) # anti clockwise rotation

    Te = T
    Pe = P
    Me = initmessagetensor(Te, Pe)

    Ts = permutedom(Te, p)
    Ps = permutedom(Pe, p)
    Ms = initmessagetensor(Ts, Ps)

    Tw = permutedom(Ts, p)
    Pw = permutedom(Ps, p)
    Mw = initmessagetensor(Tw, Pw)

    Tn = permutedom(Tw, p)
    Pn = permutedom(Pw, p)
    Mn = initmessagetensor(Tn, Pn)

    return MessageTensors(Me, Ms, Mw, Mn)
end

function updatemessage!(mts_dst, mts, tensor, op, dir::Symbol)
    if dir === :west
        perm = (1, 2, 3, 4)
    elseif dir === :north
        perm = (2, 3, 4, 1)
    elseif dir === :east
        perm = (3, 4, 1, 2)
    elseif dir === :south
        perm = (4, 1, 2, 3)
    end
    return updatemessage!(mts_dst, mts, tensor, op, perm)
end

function updatemessage!(mts_dst, mts, tensor, op, perm::NTuple{4,Int})
    return update_message_tensor_opt!(
        mts_dst[perm[3]],
        permutedom(tensor, perm),
        mts[perm[2]],
        mts[perm[3]],
        mts[perm[4]],
        permutedom(op, perm),
    )
end
function updatemessage!(mts_dst, mts, tensor, dir::Symbol)
    if dir === :west
        perm = (1, 2, 3, 4)
    elseif dir === :north
        perm = (2, 3, 4, 1)
    elseif dir === :east
        perm = (3, 4, 1, 2)
    elseif dir === :south
        perm = (4, 1, 2, 3)
    end
    return updatemessage!(mts_dst, mts, tensor, perm)
end

function updatemessage!(mts_dst, mts, tensor, perm::NTuple{4,Int})
    return update_message_tensor!(
        mts_dst[perm[3]], permutedom(tensor, perm), mts[perm[2]], mts[perm[3]], mts[perm[4]]
    )
end

# function updatewest!(mts_dst, mts, p) # x + 1 <- x
#     update_message_tensor_opt!(
#         mts_dst[3], getsite(mts, 1), mts[2], mts[3], mts[4], permutedom(p, perm)
#     )
#     return mts_dst
# end
# function updatenorth!(mts_dst, mts, p)# y + 1 <- x
#     perm = (2, 3, 4, 1) # anti clockwise rotation
#     update_message_tensor_opt!(
#         mts_dst[4], getsite(mts, 2), mts[3], mts[4], mts[1], permutedom(p, perm)
#     )
#     return mts_dst
# end
# function updateeast!(mts_dst, mts, p) # x - 1 <-x
#     perm = (3, 4, 1, 2) # anti clockwise rotation
#     update_message_tensor_opt!(
#         mts_dst[1], getsite(mts, 3), mts[4], mts[1], mts[2], permutedom(p, perm)
#     )
#     return mts_dst
# end
# function updatesouth!(mts_dst, mts, p)# y - 1<y
#     perm = (4, 1, 2, 3) # anti clockwise rotation
#     update_message_tensor_opt!(
#         mts_dst[perm[3]], getsite(mts, 4), mts[1], mts[2], mts[3], permutedom(p, perm)
#     )
#     return mts_dst
# end

# function beliefprop(tensors, p; numiter=20)
#     mts = initmessages.(tensors, p)
#     for _ in 1:numiter
#         for x in axes(tensors, 1)
#             for y in axes(tensors, 2)
#                 updatewest!(mts[x + 1, y], mts[x, y], p[x, y])
#                 updatenorth!(mts[x, y + 1], mts[x, y], p[x, y])
#                 updateeast!(mts[x - 1, y], mts[x, y], p[x, y])
#                 updatesouth!(mts[x, y - 1], mts[x, y], p[x, y])
#             end
#         end
#     end
#     return mts
# end
function beliefpropagation!(mts, pepo, map; kwargs...)
    return _beliefpropagation!(mts, getstate(pepo), map.data, kwargs...)
end

function _beliefpropagation!(mts, tensors, p; maxiter=20, tol=10.0 * eps())
    numiter = 0
    C = Inf

    # mts = initmessages.(tensors, p)
    while numiter < maxiter && real(C) ≥ tol
        mts_old = deepcopy(mts)
        for x in axes(tensors, 1)
            for y in axes(tensors, 2)
                # um! =
                #     (x, y, dir) -> updatemessage!(
                #         mts[x + 1, y], mts[x, y], tensors[x, y], p[x, y], dir
                #     )
                #
                # um!(x + 1, y, :west)
                # um!(x, y + 1, :north)
                # um!(x - 1, y, :east)
                # um!(x, y - 1, :south)
                updatemessage!(mts[x + 1, y], mts[x, y], tensors[x, y], p[x, y], :west)
                updatemessage!(mts[x, y + 1], mts[x, y], tensors[x, y], p[x, y], :north)
                updatemessage!(mts[x - 1, y], mts[x, y], tensors[x, y], p[x, y], :east)
                updatemessage!(mts[x, y - 1], mts[x, y], tensors[x, y], p[x, y], :south)
            end
        end
        numiter += 1
        C = bperror(mts, mts_old)
        @info C
    end
    println(tsvd(mts[1, 1].east)[2])
    return mts
end
function _beliefpropagation!(mts, tensors; maxiter=20, tol=10.0 * eps())
    numiter = 0
    C = Inf

    # mts = initmessages.(tensors, p)
    while numiter < maxiter && real(C) ≥ tol
        mts_old = deepcopy(mts)
        for x in axes(tensors, 1)
            for y in axes(tensors, 2)
                # um! =
                #     (x, y, dir) -> updatemessage!(
                #         mts[x + 1, y], mts[x, y], tensors[x, y], p[x, y], dir
                #     )
                #
                # um!(x + 1, y, :west)
                # um!(x, y + 1, :north)
                # um!(x - 1, y, :east)
                # um!(x, y - 1, :south)
                updatemessage!(mts[x + 1, y], mts[x, y], tensors[x, y], :west)
                updatemessage!(mts[x, y + 1], mts[x, y], tensors[x, y], :north)
                updatemessage!(mts[x - 1, y], mts[x, y], tensors[x, y], :east)
                updatemessage!(mts[x, y - 1], mts[x, y], tensors[x, y], :south)
            end
        end
        numiter += 1
        C = bperror(mts, mts_old)
        @info C
    end
    println(tsvd(mts[1, 1].east)[2].data)
    return mts
end

function beliefpropagation(tensors, p; numiter=20)
    mts = initmessages.(tensors, p)
    return beliefpropagation!(mts, tensors, p)
end

function bperror(mts_new, mts)
    nnorm = (A, B) -> tr(A / tr(A) - B / tr(B))
    cumsum = 0.0
    for (mn, mo) in zip(mts_new, mts)
        for i in 1:4
            cumsum += nnorm(mn[i], mo[i])
        end
    end
    return (1 / (2 * length(mts))) * cumsum
end

# function vidalupdate!(pepo, mts, ops; trunc=trunc)
#     beliefpropagation!(mts, pepo, ops)
#
#     bonds = [getbonds(pepo, 1)..., getbonds(pepo, 2)...]
#
#     for bond in bonds
#         u, s, v = vidalisometries(pepo, mts, bond; trunc=trunc)
#         setvidal!(vidals, u * sqrt(s), sqrt(s) * v, bond)
#     end
#
#     pepo_dst = pepo.evolved_tensors
#     pepo_src = getstate(pepo)
#
#     vidaltruncation!(pepo_dst, pepo_src)
#     copy!(pepo_src, pepo_dst)
#
#     return pepo
# end

function vidalisometries(tensors, mts, bond; trunc=notrunc())
    left = bond.left
    right = bond.right

    t1, t2 = tensors[bond]
    m1, m2 = getmessages(mts, bond)

    m1_sqrt = sqrt(m1)# <-m1<-   -'>m2-'>
    m2_sqrt = _transpose(sqrt(m2))

    u, s, v = tsvd!(m1_sqrt * m2_sqrt; trunc=trunc)# <= u <- s <- v <=

    @info "" (u.data)

    return pinv(m1_sqrt) * u, s, v * pinv(m2_sqrt)
end

function bp_tensor(T; tol=1e-12, maxiter=10)
    p = (2, 3, 4, 1) # anti clockwise rotation

    Te = T
    Me = initmessagetensor(Te)

    Ts = permutedom(Te, p)
    Ms = initmessagetensor(Ts)

    Tw = permutedom(Ts, p)
    Mw = initmessagetensor(Tw)

    Tn = permutedom(Tw, p)
    Mn = initmessagetensor(Tn)

    Me_new = similar(Me)
    Ms_new = similar(Ms)
    Mw_new = similar(Mw)
    Mn_new = similar(Mn)

    cost_func = Inf
    numiter = 0

    while numiter < maxiter
        # 3, 1 2 3 4
        update_message_tensor!(Mw_new, Te, Ms, Mw, Mn) # x -> x + 1
        # 4, 2 3 4 1
        update_message_tensor!(Mn_new, Ts, Mw, Mn, Me) # y -> y + 1
        # 1, 3 4 1 2
        update_message_tensor!(Me_new, Tw, Mn, Me, Ms) # x -> x - 1
        # 2, 4 1 2 3
        update_message_tensor!(Ms_new, Tn, Me, Ms, Mw) # y -> y - 1

        rmul!(Me_new, 1 / tr(Me_new))
        rmul!(Ms_new, 1 / tr(Ms_new))
        rmul!(Mw_new, 1 / tr(Mw_new))
        rmul!(Mn_new, 1 / tr(Mn_new))

        cost_func = real(
            1 / 8 * (
                tr(Me / tr(Me) - Me_new / tr(Me_new)) +
                tr(Ms / tr(Ms) - Ms_new / tr(Ms_new)) +
                tr(Mw / tr(Mw) - Mw_new / tr(Mw_new)) +
                tr(Mn / tr(Me) - Mn_new / tr(Mn_new))
            ),
        )

        copy!(Me, Me_new)
        copy!(Ms, Ms_new)
        copy!(Mw, Mw_new)
        copy!(Mn, Mn_new)

        numiter += 1
    end
    return Me, Ms, Mw, Mn, cost_func, numiter
end
