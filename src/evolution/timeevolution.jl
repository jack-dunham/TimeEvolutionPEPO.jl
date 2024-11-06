# function newevolution(
#     pepo::AbstractPEPO, map::Vector{T}, _
# ) where {T<:Union{OneBondStepper,TwoBondStepper}}
#     xind = findfirst(s -> bondaxis(s) == 1, map)
#     yind = findfirst(s -> bondaxis(s) == 2, map)
#
#     xpepo = newevolution(pepo, map[xind])
#     ypepo = newevolution(changeaxis(pepo), map[yind])
#
#     return xpepo, ypepo
# end

newevolution(pepo, steps, method) = newevolution(eltype(steps), pepo, steps, method)
newevolution(::Type{<:AxialStepper}, pepo, steps, method) = AxialPEPO(pepo)
newevolution(::Type{<:AbstractTimeStepper}, pepo, steps, method) = pepo

newevolution(::Type{<:LocalStepper}, pepo, steps, method::TEBD{<:Union{FET, NTU}}) = AxialPEPO(pepo)

function newevolution(pepo::AbstractPEPO, map::LocalStepper)
    u, v = tensors(map)
    evolved = broadcast(tensors(pepo), u, v) do p, u, v
        D1, D2, D3, D4 = domain(p)

        TL = promote_type(eltype(p), eltype(u))
        TR = promote_type(eltype(p), eltype(v))

        left = TensorMap(undef, TL, codomain(p), fuse(D1 * domain(u)) * D2 * D3 * D4)
        right = TensorMap(undef, TR, codomain(p), D1 * D2 * fuse(D3 * domain(v))' * D4)

        return left, right
    end

    firstbond = Bond(CartesianIndex(1, 1), CartesianIndex(2, 1))

    return LocalUpdatablePEPO(pepo, evolved, firstbond, false)
end

function newevolution(pepo::AbstractPEPO, map::TwoBondStepper)
    evolved = broadcast(tensors(pepo), tensors(map)) do p, t
        D1, D2, D3, D4 = domain(p)

        east = domain(t)[1]
        west = domain(t)[2]

        T = promote_type(eltype(p), eltype(t))

        tensor = TensorMap(
            undef, T, codomain(p), fuse(D1 * east) * D2 * fuse(D3 * west)' * D4
        )

        return tensor
    end

    return GlobalUpdatablePEPO(pepo, evolved, false)
end

function newevolution(pepo::AbstractPEPO, map::FullStepper)
    evolvedx = broadcast(tensors(pepo), tensors(map)) do p, t
        D1, D2, D3, D4 = domain(p)
        η1, η2, η3, η4 = domain(t)

        T = promote_type(eltype(p), eltype(t))

        tensor = TensorMap(
            undef,
            T,
            codomain(p),
            fuse(D1 * η1) * fuse(D2 * η2) * fuse(D3 * η3)' * fuse(D4 * η4)',
        )

        return tensor
    end

    evolvedy = broadcast(tensors(pepo), tensors(map)) do p, t
        D1, D2, D3, D4 = domain(p)
        η1, η2, η3, η4 = domain(t)

        T = promote_type(eltype(p), eltype(t))

        tensor = TensorMap(
            undef,
            T,
            codomain(p),
            fuse(D2 * η2) * fuse(D1 * η1) * fuse(D4 * η4)' * fuse(D3 * η3)',
        )

        return tensor
    end

    pepox = GlobalUpdatablePEPO(pepo, evolvedx, false)
    pepoy = GlobalUpdatablePEPO(changeaxis(pepo), evolvedy, false)
    return pepox, pepoy
end

function newevolution(pepo::AbstractPEPO, map::NTuple{2,<:SplitStepper}, method)
    evolvedx = broadcast(tensors(pepo), tensors(map[1])) do p, t
        D1, D2, D3, D4 = domain(p)
        ηE, ηW = domain(t)

        dk, db, _, _ = codomain(t)

        T = promote_type(eltype(p), eltype(t))

        tensor = TensorMap(undef, T, dk * db, fuse(D1 * ηE) * D2 * fuse(D3 * ηW)' * D4)

        return tensor
    end

    basey = broadcast(tensors(pepo), tensors(map[1])) do p, t
        D1, D2, D3, D4 = domain(p)

        T = eltype(p)

        dk, db, _, _ = codomain(t)

        tensor = TensorMap(undef, T, dk * db, D2 * D1 * D4 * D3)

        return tensor
    end

    evolvedy = broadcast(tensors(pepo), tensors(map[2])) do p, t
        D1, D2, D3, D4 = domain(p)
        ηS, ηN = domain(t)

        T = promote_type(eltype(p), eltype(t))

        tensor = TensorMap(undef, T, codomain(p), fuse(D2 * ηS) * D1 * fuse(D4 * ηN)' * D3)

        return tensor
    end

    newpepoy = PEPO(basey, changeaxis(getweight(pepo)))

    pepox = GlobalUpdatablePEPO(pepo, evolvedx, false)
    pepoy = GlobalUpdatablePEPO(newpepoy, evolvedy, false)
    return pepox, pepoy
end

# # One full cycle of the trotter evolution. That is, t -> t + τ
# function timestep!(pepo, truncator, sequence, truncalg; kwargs...)
#     # Do forward staircase on horizontal then vertical, switch inbetween, i.e does the 
#     # sequence: H <-> V
#
#     axis = prevaxis = 1
#
#     for stepper in sequence
#         axis = bondaxis(stepper)
#
#         # @info "Applying operator on axis $axis..."
#
#         # If we are now applying on a new axis then we must write the data from the old
#         # axis to the new one.
#         axis != prevaxis && setaxis!(pepo, axis)
#
#         # Update the state
#         trotterstep!(pepo, truncator, stepper, truncalg; kwargs...)
#
#         prevaxis = axis
#     end
#
#     # We must convert back to axis 1 for the next iteration
#     axis == 2 && setaxis!(pepo, 1)
#
#     return pepo
# end

# Non-trotterised evolution
function timestep!(pepo, truncator, sequence, truncalg; kwargs...)
    # us = updatestyle(pepo)
    # ts = truncstyle(pepo)

    timestep!(pepo, first(truncator), sequence, truncalg; kwargs...)

    return pepo
end

# Parallel Semi Parallel
function timestep!(pepo::AxialPEPO, truncator, stepper, truncalg; kwargs...)

    axis = stepper.axis

    pepo.currentaxis == axis || setaxis!(pepo, axis)

    evolvedtensors = evolve(pepo, stepper)

    infidelity = 0

    buffer = tensors(pepo)

    if truncalg isa FET
        trunc = newtruncators(evolvedtensors, truncalg)
        renormalize!(trunc)
    end

    # mts = beliefpropagation(evolvedtensors(pepox))

    isomh = broadcast(bonds(buffer, 1)) do bond
        if truncalg isa NTU
            trunc = pair -> ntu_total(pair, evolvedtensors, bond)
        end

        (u, v), s, _ = parallelisometries(
            evolvedtensors, trunc, bond, truncalg; D=bondspace(pepo, bond)
        )

        realbond = axis == 2 ? transpose(bond) : bond

        setweight!(pepo, s, realbond; copy=false)

        return (u, v)
    end

    isou = map(iso -> first(iso), isomh)
    isov = map(iso -> last(iso), circshift(isomh, (1, 0)))

    broadcast(buffer, evolvedtensors, isou, isov) do dst, src, u, v
        truncateto!(dst, src, u, v)
    end

    # size(pepo) == (1,1) && minimalcanonicalform!(tensors(pepo))

    quicknormalize!(pepo.parent)
    hermitianize!(pepo.parent)

    return pepo, infidelity
end

function itrtimestep_deprec!(
    pepo,
    truncator,
    stepper::Vector{<:FullStepper{<:Any,<:SparseTensorMap}},
    ::Number;
    kwargs...,
)
    stepper = only(stepper)

    infidelity = 0

    buff = currentbuffer(pepo)

    weightsbuff = deepcopy(weights(pepo))
    oldweights = deepcopy(weights(pepo))

    isoms_data = broadcast(buff, tensors(stepper)) do dst, src
        return stepstitchisometries(storagetype(dst), domain(dst), domain(src))
    end

    isoms = Isometries(isoms_data)
    isomsbuff = similar(isoms)

    local isoe
    local isow

    local ison
    local isos

    schedule = :sequential
    # schedule = :parallel

    for i in 1:15

        isomh = broadcast(bonds(buff, 1)) do bond
            reducedpair = localevolve(buff[bond], stepper, isoms, bond)

            foreach(normalize!, reducedpair)

            D = bondspace(pepo, bond)
            (u, v), s, _ = simpleupdate(reducedpair, weights(pepo), bond; trunc=truncspace(D))


            if schedule === :sequential
                setindex!(isoms, u => v, bond)
                setweight!(pepo, s, bond; copy=true)
            elseif schedule === :parallel
                setindex!(isomsbuff, u => v, bond)
                weightsbuff[bond] = s
            end

            return (u, v)
        end

        isomv = broadcast(bonds(buff, 2)) do bond
            reducedpair = localevolve(buff[bond], stepper, isoms, bond)

            foreach(normalize!, reducedpair)

            D = bondspace(pepo, bond)
            (u, v), s, _ = simpleupdate(reducedpair, weights(pepo), bond; trunc=truncspace(D))


            if schedule === :sequential
                setindex!(isoms, u => v, bond)
                setweight!(pepo, s, bond; copy=true)
            elseif schedule === :parallel
                setindex!(isomsbuff, u => v, bond)
                weightsbuff[bond] = s
            end

            return (u, v)
        end

        # foreach(bonditer(buff)) do bond
        #     s = weightsbuff[bond]
        #     setweight!(pepo, s, bond; copy=false)
        # end

        # isoe = broadcast(first, isomh)
        # isow = broadcast(last, circshift(isomh, (1, 0)))
        #
        # isos = broadcast(first, permutedims(isomv))
        # ison = broadcast(last, permutedims(circshift(isomv, (1, 0))))
        #
        #
        # isoms = SquareLattice(
        #     broadcast(tuple, isoe, isos, isow, ison)
        # )


        if schedule === :sequential
            tol = tolerance!(weightsbuff, weights(pepo), 1.0)
        else
            elcopy!(isoms, isomsbuff)
            tol = tolerance!(weights(pepo), weightsbuff, 1.0)
        end

        @warn "Got $tol after $i"

        if tol < 1e-6 && i > 0
            break
        end

        # temppepo = deepcopy(pepo)
        #
        # broadcast(
        #     tensors(temppepo), tensors(stepper), isoe, isos, isow, ison
        # ) do dst, op, e, s, w, n
        #     spten = truncateto(dst, dst, op, e, s, w, n)
        #     mergeinto!(dst, spten)
        # end
        #
        # gauge1, gauge2 = simple_update_gauge_isometries!(temppepo)
        #
        # ge = broadcast(first, gauge1)
        # gw = broadcast(last, circshift(gauge1, (1, 0)))
        #
        # gs = broadcast(first, permutedims(gauge2))
        # gn = broadcast(last, permutedims(circshift(gauge2, (1, 0))))
        #
        # gisoms = SquareLattice(
        #     broadcast(tuple, ge, gs, gw, gn)
        # )

        # isoms = broadcast(isoms, gisoms) do iso, g
        #     e1, s1, w1, n1 = iso
        #     e2, s2, w2, n2 = g
        #
        #     e = e1 * e2
        #     s = s1 * s2
        #
        #     w = w2 * w1
        #     n = n2 * n1
        #
        #     return (e,s,w,n)
        # end
        #
        # foreach(bonditer(buff)) do bond
        #     s = weights(temppepo)[bond]
        #     setweight!(pepo, s, bond; copy=true)
        # end
    end

    # broadcast(
    #     buff, tensors(stepper), isoe, isos, isow, ison
    # ) do dst, op, e, s, w, n
    #     spten = truncateto(dst, dst, op, e, s, w, n)
    #     mergeinto!(dst, spten)
    # end

    for I in eachindex(buff)

        dst = buff[I]
        op = tensors(stepper)[I]
        e, s, w, n = isoms[I]

        spten = truncateto(dst, dst, op, e, s, w, n)

        mergeinto!(dst, spten)
    end

    size(buff) == (1, 1) && minimalcanonicalform!(buff)

    size(buff) != (1, 1) && simple_update_gauge!(pepo; maxiter=100)

    pepo.state = 0

    # minimalcanonicalform!(tensors(pepo))

    quicknormalize!(pepo.parent)
    hermitianize!(pepo.parent)

    return pepo, infidelity
end

function timestep!(
    pepo, truncator, stepper::Vector{<:FullStepper{<:Any,<:TensorMap}}, ::Number; kwargs...
)
    evolve!(pepo, only(stepper))

    # simple_update_gauge!(pepo; maxiter=100)

    # truncate_bond_matrices!(pepo)

    # quicknormalize!(pepo.parent)
    # hermitianize!(pepo.parent)

    # return pepo

    # ξ = 500.0
    ξ = Inf
    # truncalg = NTU(; ξ=ξ, rankreduce=false, optalg=EVB(; gauge=NoGauge()))
    truncalg = NTU(; ξ=ξ, rankreduce=false, optalg=EVB())
    # truncalg = FET(; alg=VUMPS(bonddim=16, tol=1e-8), ξ=ξ, rankreduce=false, optalg=EVB())
    # truncalg = NTU(; ξ=ξ, rankreduce=false, optalg=DZM(gauge=NoGauge()))
    # truncalg = NTU(; ξ=ξ, rankreduce=true, optalg=DZM())
    # truncalg = SU()

    # println("xi = $(ξ)")

    # evx = evolvedtensors(pepox)[1]
    # evy = evolvedtensors(pepoy)[1]
    #

    infidelity = 0

    buff = currentbuffer(pepo)
    ybuff = permutetranspose(buff)

    # truncx = newtruncators(buff, truncalg)
    # renormalize!(truncx)
    #
    # truncy = newtruncators(ybuff, truncalg)
    # renormalize!(truncy)

    reducedbuffer = broadcast(baretruncation!, similar.(tensors(pepo)), buff)

    xisoms = stitchisometries.(tensors(pepo), buff)
    yisoms = stitchisometries.(permutetranspose(tensors(pepo)), ybuff)

    weightsbuff = deepcopy(weights(pepo))

    tol = Inf

    maxiter = 1

    for numiter in 1:maxiter
        yreducedbuffer = permutetranspose(reducedbuffer)

        # truncx = newtruncators(reducedbuffer, truncalg)
        # renormalize!(truncx)
        # truncx = reducedbuffer
        # truncx = SimpleUpdateState()

        isomh = broadcast(bonds(buff, 1)) do bond
            # reducedpair = parallel2sequential(buff, xisoms, bond)
            # truncx = ntu_total_low(reducedpair, reducedbuffer, bond)

            reducedpair = buff[bond]
            truncx = ntu_total_low(reducedpair, buff, bond)

            # truncx = ntu_total_low(reducedpair, reducedbuffer, bond)

            (u, v), s, _ = parallelisometries(
                reducedpair, truncx, bond, truncalg; D=bondspace(pepo, bond)
            )

            # D = bondspace(pepo, bond)
            # (u, v), s, _ = simpleupdate(reducedpair, weights(pepo), bond; trunc=truncspace(D))

            setweight!(pepo, s, bond; copy=false)

            return (u, v)
        end

        # truncy = newtruncators(yreducedbuffer, truncalg)
        # renormalize!(truncy)
        # truncy = yreducedbuffer
        # truncy = SimpleUpdateState()

        isomv = broadcast(bonds(ybuff, 1)) do fictbond
            # reducedpair = parallel2sequential(ybuff, yisoms, fictbond)
            # truncy = ntu_total_low(reducedpair, yreducedbuffer, fictbond)

            reducedpair = ybuff[fictbond]
            truncy = ntu_total_low(reducedpair, ybuff, fictbond)
            # ql0, qr0 = parallel2sequential(ybuff, yisoms, fictbond)
            #
            # ql = permutedom(ql0, (2,1,4,3))
            # qr = permutedom(qr0, (2,1,4,3))

            # truncy = ntu_total_low(reducedpair,yreducedbuffer, fictbond)

            realbond = transpose(fictbond)

            (u, v), s, _ = parallelisometries(
                reducedpair, truncy, fictbond, truncalg; D=bondspace(pepo, realbond)
            )

            # D = bondspace(pepo, realbond)
            # (u, v), s, _ = simpleupdate(ql => qr, weights(pepo), realbond; trunc=truncspace(D))

            setweight!(pepo, s, realbond; copy=false)

            return (u, v)
        end

        isoe = broadcast(first, isomh)
        isow = broadcast(last, circshift(isomh, (1, 0)))

        isos = broadcast(first, isomv)
        ison = broadcast(last, circshift(isomv, (1, 0)))

        xisoms = SquareLattice(
            broadcast(tuple, isoe, permutedims(isos), isow, permutedims(ison))
        )
        yisoms = SquareLattice(
            broadcast(tuple, permutedims(isoe), isos, permutedims(isow), ison)
        )

        tol = tolerance!(weightsbuff, weights(pepo), 1.0)

        # if numiter == maxiter && tol > 1e-10
        #     xisoms = stitchisometries.(tensors(pepo), buff)
        #     yisoms = stitchisometries.(permutetranspose(tensors(pepo)), ybuff)
        # end

        foreach(
            reducedbuffer, buff, isoe, permutedims(isos), isow, permutedims(ison)
        ) do dst, src, e, s, w, n
            truncateto!(dst, src, e, s, w, n)
        end

        #
        # if tol < 1e-8
        #     @warn "isometries converged to $tol after $numiter"
        #     break
        # end
        #
        #
        # minimalcanonicalform!(reducedbuffer)

    end

    @warn "isometries converged to $tol after"

    copy!.(tensors(pepo), reducedbuffer)

    # foreach(
    #     tensors(pepo), buff, isoe, permutedims(isos), isow, permutedims(ison)
    # ) do dst, src, e, s, w, n
    #     truncateto!(dst, src, e, s, w, n)
    # end

    pepo.state = 0

    # simple_update_gauge!(pepo; maxiter=100)

    # minimalcanonicalform!(tensors(pepo))

    quicknormalize!(pepo.parent)
    hermitianize!(pepo.parent)

    return pepo, infidelity
end

function timestep!(pepo, truncator, stepper::NTuple{2,<:SplitStepper}, ::Number; kwargs...)
    _, infidelity = halftimestep!(pepo, truncator, stepper, 1)
    halftimestep!(pepo, truncator, stepper, 2)
    #
    return pepo, infidelity
end
# split method
function halftimestep!(
    allpepo, truncator, stepper::NTuple{2,<:SplitStepper}, dir; kwargs...
)
    pepo = allpepo[dir]
    otherpepo = allpepo[3 - dir]
    dst = tensors(otherpepo)
    stepper = stepper[dir]

    # truncalg = NTU(; rankreduce=false, optalg=EVB())
    truncalg = SU()
    trunc = SimpleUpdateState()
    # truncalg = FET(; alg=VUMPS(bonddim=16), rankreduce=false, optalg=EVB())

    evolve!(pepo, stepper)

    # trunc = newtruncators(evolvedtensors(pepo), truncalg)
    # renormalize!(trunc)

    bond = uniquebonds(pepo)

    infidelity = similar(bond, Float64)

    isom = broadcast(enumerate(bond)) do ebond
        i, bond = ebond
        iso, _ = isometries(pepo, trunc, bond, truncalg)

        infidelity[i] = 0.0

        setweight!(pepo, iso.s, bond; copy=false)

        # w = getweights(pepo).horizontal
        # w[left(bond)] = iso.s

        return iso
    end

    isoe = map(iso -> iso.u, isom)
    isow = map(iso -> iso.v, circshift(isom, (1, 0)))

    broadcast(dst, currentbuffer(pepo), isoe, isow) do dst, src, e, w
        truncateto_split!(dst, src, e, w)
    end

    quicknormalize!(pepo)
    hermitianize!(pepo)

    return allpepo, infidelity
end

# quick bp
function _timestep!(
    pepo, truncators, stepper::NTuple{N,FullStepper}, ::Number; kwargs...
) where {N}
    pepo = pepo[1]

    dstpepo = tensors(pepo)
    srcpepo = currentbuffer(pepo)

    site = dstpepo[1]

    infidelity = similar(dstpepo, Float64)
    bondh = Bond(CartesianIndex(1, 1), CartesianIndex(2, 1))
    bondv = Bond(CartesianIndex(1, 1), CartesianIndex(1, 2))

    # mts = beliefpropagation(srcpepo)
    op = tensors(stepper[1])
    mts = beliefpropagation(dstpepo, op)

    # gauge

    sitetemp = srcpepo[1]

    isoh = vidalisometries(pepo, mts, bondh; axis=1)
    isov = vidalisometries(pepo, mts, bondv; axis=2)

    wh = sqrt(isoh.s)
    wv = sqrt(isov.s)

    # gauge the tensor
    @tensor siteprime[k b; ee ss ww nn] :=
        op[1, 1][k b ik ib; e2 s2 w2 n2] *
        site[ik ib; e1 s1 w1 n1] *
        isoh.u[e1 e2; ee] *
        isov.u[s1 s2; ss] *
        isoh.v[ww; w1 w2] *
        isov.v[nn; n1 n2]

    # absorb the weights for horizontal
    @tensor begin
        left[k b; e ss ww nn] := siteprime[k b; e s w n] * wv[s; ss] * wh[ww; w] * wv[nn; n]
        right[k b; ee ss w nn] :=
            siteprime[k b; e s w n] * wv[s; ss] * wh[e; ee] * wv[nn; n]
    end

    D = domain(site)[1]

    _, u0, v0, _ = rankreduce(left, right)
    # _, u0, v0, _ = rankreduce(siteprime, siteprime)
    u, s, v, _ = safetsvd!(u0 * v0; trunc=truncspace(D))

    maxnormalize!(s)

    uh = pinv(u0) * u * sqrt(s)
    vh = sqrt(s) * v * pinv(v0)

    @tensor begin
        up[k b; s ee nn ww] := siteprime[k b; e s w n] * wh[ww; w] * wv[nn; n] * wh[e; ee]
        down[k b; ss ee n ww] := siteprime[k b; e s w n] * wv[s; ss] * wh[ww; w] * wh[e; ee]
    end

    D = domain(site)[2]

    _, u0, v0, _ = rankreduce(up, down)
    # _, u0, v0, _ = rankreduce(permutedom(siteprime,(2,1,4,3)), permutedom(siteprime,(2,1,4,3)))
    u, s, v, _ = safetsvd!(u0 * v0; trunc=truncspace(D))

    maxnormalize!(s)

    uv = pinv(u0) * u * sqrt(s)
    vv = sqrt(s) * v * pinv(v0)

    truncateto!(site, siteprime, uh, uv, vh, vv)

    if ischeckerboard(pepo)
        # If our unitcell is symmetric, then copy the updated bond to all
        # other columns with an appropriate shift
        deepcopysymmetric!(dstpepo)
    end

    quicknormalize!(pepo)
    hermitianize!(pepo)

    minimalcanonicalform!(site)

    return pepo, infidelity
end

# belief propagation
function _bptimestep!(
    pepo, truncators, stepper::NTuple{N,FullStepper}, ::Number; kwargs...
) where {N}
    pepox, pepoy = pepo
    stepper = stepper[1]

    dstpepo = tensors(pepox)
    srcpepo = currentbuffer(pepox)

    mts = beliefpropagation(dstpepo, tensors(stepper))

    bondh = bonds(pepox, 1)
    bondv = bonds(pepox, 2)

    infidelity = similar(bondh, Float64)

    # gauge

    isomh = broadcast(enumerate(bondh)) do ebond
        i, bond = ebond
        D = bondspace(pepox, bond)
        # trunc=truncspace(D)
        trunc = notrunc()
        iso = vidalisometries(pepox, mts, bond; trunc=trunc, axis=1)

        infidelity[i] = 0.0

        w = getweight(pepox).horizontal
        w[left(bond)] = iso.s

        println(iso.s)
        # setweights!(pepox, iso.s, bond; copy=false)

        return iso
    end

    changeaxis!(getweight(pepoy), getweight(pepox))

    isomv = broadcast(bondv) do bond
        D = bondspace(pepoy, bond)
        # trunc=truncspace(D)
        trunc = notrunc()
        iso = vidalisometries(pepox, mts, bond; trunc=trunc, axis=2)

        w = getweight(pepox).vertical
        w[left(bond)] = iso.s

        # setweights!(pepoy, iso.s, bond; copy=false)

        return iso
    end

    changeaxis!(getweight(pepoy), getweight(pepox))

    isoe = map(iso -> iso.u, isomh)
    isow = map(iso -> iso.v, circshift(isomh, (1, 0)))

    isos = map(iso -> iso.u, isomv)
    ison = map(iso -> iso.v, circshift(isomv, (1, 0)))

    foreach(srcpepo, dstpepo, isoe, isos, isow, ison) do src, dst, e, s, w, n
        step = tensors(stepper)[1]
        truncateto!(dst, deepcopy(dst), step, e, s, w, n)
    end
    #=

    # Write into the permuted network
    broadcast(evolvedtensors(pepoy), permutedims(evolvedtensors(pepox))) do pdst, psrc
        permute!(pdst, psrc, ((1, 2), (4, 3, 6, 5)))
    end

    # truncate

    truncalg = SU()
    truncator = SimpleUpdateState()

    infidelity = similar(bondv, Float64)

    isomh = broadcast(uniquebonds(pepox)) do bond

        println(typeof(getweights(pepox)))

        absorb_evolved_weights!(pepox, bond; inverse=false)
        iso, _ = isometries(pepox,  bond)
        absorb_evolved_weights!(pepox, bond; inverse=true)

        # setweights!(pepo, iso.s, bond; copy=false)

        return iso
    end

    isomv = broadcast(uniquebonds(pepoy)) do bond

        absorb_evolved_weights!(pepoy, bond; inverse=false)
        iso, _ = isometries(pepoy, bond)
        absorb_evolved_weights!(pepoy, bond; inverse=true)

        # setweights!(pepo, iso.s, bond; copy=false)

        return iso
    end

    foreach(isomh, uniquebonds(pepox)) do iso, bond
        w = getweights(pepox).horizontal
        w[left(bond)] = iso.s
    end
    foreach(isomv, uniquebonds(pepoy)) do iso, bond
        w = getweights(pepox).vertical
        w[left(bond)] = iso.s
    end

    isoe = map(iso -> iso.u, isomh)
    isow = map(iso -> iso.v, circshift(isomh, (1, 0)))

    isos = map(iso -> iso.u, isomv)
    ison = map(iso -> iso.v, circshift(isomv, (1, 0)))

    foreach(dstpepo, srcpepo, isoe, isos, isow, ison) do dst, src, e, s, w, n
        truncateto!(dst, src,  e, s, w, n)
    end

    =#
    if ischeckerboard(pepox)
        # If our unitcell is symmetric, then copy the updated bond to all
        # other columns with an appropriate shift
        deepcopysymmetric!(dstpepo)
    end

    quicknormalize!(pepox)

    hermitianize!(pepox)

    quicknormalize!(pepoy)

    hermitianize!(pepoy)

    return pepo, infidelity
end

function trotterstep!(pepo, truncator, stepper; kwargs...)
    reversed = isreversed(stepper)

    updatestep!(pepo, truncator, stepper; isreversed=reversed, kwargs...)

    return pepo
end

function _trotterstep!(pepo, truncator, stepper; kwargs...)
    reverse = isreversed(stepper)

    in = 1 + reverse
    out = 2 - reverse

    # H/V
    updatestep!(pepo[in], truncator[in], stepper[in]; reverse=reverse, kwargs...)

    # <->
    changeaxis!(pepo[out], pepo[in])

    # V/H
    updatestep!(pepo[out], truncator[out], stepper[out]; reverse=reverse, kwargs...)

    return pepo
end

# Single site update step
function _ssquickupdate!(pepo::GlobalUpdatablePEPO, truncator; truncalg)
    onlybond = bonds(pepo)[1, 1]

    src_pepo = currentbuffer(pepo)
    dst_pepo = tensors(pepo)

    src_site = src_pepo[1, 1]
    dst_site = dst_pepo[1, 1]

    renormalize!(truncator)

    iso = isometries(pepo, truncator, onlybond, truncalg)

    setweight!(pepo, iso.s, onlybond)

    temp = partialtruncate(src_site, iso)
    rv = partialtruncate(iso, temp)

    truncalg.gauge isa MCF && minimalcanonicalform!(rv)

    copy!(dst_site, rv)

    return pepo
end

function ssquickupdate!(pepo::GlobalUpdatablePEPO, truncator; truncalg)
    onlybond = bonds(pepo)[1, 1]

    src_pepo = currentbuffer(pepo)
    dst_pepo = tensors(pepo)

    src_site = src_pepo[1, 1]
    dst_site = dst_pepo[1, 1]

    D = bondspace(pepo, onlybond)

    w = getweight(pepo)

    src_site, _ = absorbweights!(src_site, deepcopy(src_site), w, onlybond; inverse=false)
    P1, R, L, _ = rankreduce(src_site, src_site)

    u, s, v = tsvd!(R * L; trunc=truncspace(D), alg=TensorKit.SVD())

    maxnormalize!(s)

    #
    # temp = _partialtruncate(P1, u * sqrt(s))
    #
    # _, R, L, P2 = rankreduce(src_site, temp)
    #
    # u,s,v = tsvd!(R * L; trunc=truncspace(D))
    #
    # rv = _partialtruncate(sqrt(s) * v, P2)

    l = pinv(R) * u * sqrt(s)
    r = sqrt(s) * v * pinv(L)

    src_site, _ = absorbweights!(src_site, deepcopy(src_site), w, onlybond; inverse=true)

    truncateto!(dst_site, src_site, l, r)

    # truncalg.gauge isa MCF && minimalcanonicalform!(dst_site)

    setweight!(pepo, s, onlybond)

    # copy!(dst_site, rv)

    return pepo
end

function quickupdate!(pepo::GlobalUpdatablePEPO, truncator; truncalg)
    # if truncalg.rankreduce == true
    #     throw(ErrorException("Cannot do rank reduction when updating in parallel"))
    # end

    bonds = uniquebonds(pepo)

    srcpepo = currentbuffer(pepo)
    dstpepo = tensors(pepo)

    renormalize!(truncator)

    infidelity = similar(bonds, Float64)

    # Parallelisable
    isoms = broadcast(enumerate(bonds)) do ebond
        i, bond = ebond
        iso, infid = isometries(pepo, truncator, bond, truncalg)

        infidelity[i] = infid

        setweight!(pepo, iso.s, bond; copy=false)

        return iso
    end

    us = map(iso -> iso.u, isoms)
    vs = map(iso -> iso.v, circshift(isoms, (1, 0)))

    foreach(dstpepo, srcpepo, us, vs) do dst, src, u, v
        truncateto!(dst, src, u, v)
    end

    setisevolved!(pepo, false)
    # BP
    #=
    mts = beliefpropagation(dstpepo)

    bondh = getbonds(pepo, 1)
    bondv = getbonds(pepo, 2)

    isomh = broadcast(bondh) do bond
        iso = vidalisometries(pepo, mts, bond; trunc=notrunc(), axis=1)

        setweights!(pepo, iso.s, bond; copy=true)

        return iso
    end

    isomv = broadcast(bondv) do bond
        iso = vidalisometries(pepo, mts, bond; trunc=notrunc(), axis=2)

        setweights!(pepo, iso.s, bond; copy=true)

        return iso
    end

    isoe = map(iso -> iso.u, isomh)
    isow = map(iso -> iso.v, circshift(isomh, (1, 0)))

    isos = map(iso -> iso.u, isomv)
    ison = map(iso -> iso.v, circshift(isomv, (1, 0)))

    foreach(dstpepo, isoe, isos, isow, ison) do dst, e, s, w, n
        truncateto!(dst, deepcopy(dst), e, s, w, n)
    end
    =#

    if ischeckerboard(pepo)
        # If our unitcell is symmetric, then copy the updated bond to all
        # other columns with an appropriate shift
        deepcopysymmetric!(dstpepo)
    end

    truncalg.gauge isa MCF && minimalcanonicalform!(dstpepo)

    return pepo, infidelity
end

function bpupdate!(pepo::GlobalUpdatablePEPO, truncator, stepper; truncalg)
    # if truncalg.rankreduce == true
    #     throw(ErrorException("Cannot do rank reduction when updating in parallel"))
    # end

    dstpepo = tensors(pepo)
    srcpepo = currentbuffer(pepo)

    # mts = beliefpropagation(srcpepo)
    mts = beliefpropagation(dstpepo, tensors(stepper))
    mten = tensors(stepper)[1]
    # @tensor tno[kkk bbb k b; e s w n] := mten[kkk bbb kk bb; s n] * mten[kk bb k b; e w]
    # mts = beliefpropagation(latticetensors(pepo), [tno tno; tno tno])

    bondh = bonds(pepo, 1)
    bondv = bonds(pepo, 2)

    infidelity = similar(bondh, Float64)

    isomh = broadcast(enumerate(bondh)) do ebond
        i, bond = ebond
        D = bondspace(pepo, bond)
        iso = vidalisometries(pepo, mts, bond; trunc=truncspace(D), axis=1)

        infidelity[i] = 0.0

        setweight!(pepo, iso.s, bond; copy=false)

        # w = getweights(pepo).horizontal
        # w[left(bond)] = iso.s

        return iso
    end

    isomv = broadcast(bondv) do bond
        D = bondspace(pepo, bond)
        iso = vidalisometries(pepo, mts, bond; trunc=truncspace(D), axis=2)

        setweight!(pepo, iso.s, bond; copy=false)

        # w = getweights(pepo).vertical
        # w[left(bond)] = iso.s

        return iso
    end

    isoe = map(iso -> iso.u, isomh)
    isow = map(iso -> iso.v, circshift(isomh, (1, 0)))

    isos = map(iso -> iso.u, isomv)
    ison = map(iso -> iso.v, circshift(isomv, (1, 0)))

    # foreach(dstpepo, srcpepo, isoe, isos, isow, ison) do dst, src, e, s, w, n
    #     step = currentstepper(stepper)[1]
    #     oldsrc = deepcopy(src)
    #     # u = one(e)
    #     # v = one(w)
    #     # oldo = overlap(u, v, oldsrc, oldsrc, u, v)
    #     truncateto!(src, oldsrc, e, s, w, n)
    #     # truncateto!(dst, src, e, s, w, n)
    #     # newo = overlap(u, v, src, src, u, v)
    #     # @warn "" oldo - newo
    # end

    # dummypepo = deepcopy(pepo)
    # isoms = broadcast(enumerate(bondh)) do ebond
    #     i, bond = ebond
    #
    #     left, right = evolvedtensors(dummypepo)[bond]
    #     println(space(left))
    #     absorbweights!(left, right, getweights(pepo), bond; inverse=false)
    #     iso, infid = isometries(dummypepo, bond; xi=truncalg.ξ)
    #     # setweights!(pepo, iso.s, bond)
    #
    #     return iso
    # end
    #
    # us = map(iso -> iso.u, isoms)
    # vs = map(iso -> iso.v, circshift(isoms, (1, 0)))
    #
    # foreach(dstpepo, srcpepo, us, vs) do dst, src, u, v
    #     truncateto!(dst, src, u, v)
    # end

    # foreach(dstpepo, srcpepo, isoe, isos, isow, ison) do dst, src, e, s, w, n
    #     truncateto!(dst, src, e, s, w, n)
    # end

    foreach(dstpepo, isoe, isos, isow, ison) do dst, e, s, w, n
        step = tensors(stepper)[1]
        truncateto!(dst, deepcopy(dst), step, e, s, w, n)
    end

    # foreach(dstpepo, isoe, isow) do dst, e, w
    #     step = currentstepper(stepper)[1]
    #     truncateto!(dst, deepcopy(dst), step, e, w)
    # end

    # quickupdate!(pepo, truncator; truncalg=truncalg)

    # weights = getweights(pepo)
    #
    # isoms = broadcast(enumerate(bondh)) do ebond
    #     i, bond = ebond
    #     iso, infid = isometries(pepo, truncator, bond, truncalg)
    #
    #     infidelity[i] = infid
    #
    #     setweights!(pepo, iso.s, bond; copy=false)
    #
    #     return iso
    # end
    #
    # us = map(iso -> iso.u, isoms)
    # vs = map(iso -> iso.v, circshift(isoms, (1, 0)))
    #
    # foreach(dstpepo, srcpepo, us, vs) do dst, src, u, v
    #     truncateto!(dst, src, u, v)
    # end

    if ischeckerboard(pepo)
        # If our unitcell is symmetric, then copy the updated bond to all
        # other columns with an appropriate shift
        deepcopysymmetric!(dstpepo)
    end

    truncalg.gauge isa MCF && minimalcanonicalform!(dstpepo)

    return pepo, infidelity
end

# function updatestep!(pepo::GlobalUpdatablePEPO, truncator, stepperlist::Vector{<:AbstractTimeStepper}; truncalg, reverse)
#
#     # Apply the operator trotter decomposition
#     for stepper in stepperlist
#         updatestep!(pepo, truncator, stepper; truncalg=truncalg, reverse=false)
#     end
#
#     return pepo
# end

function updatestep!(
    pepo::GlobalUpdatablePEPO, truncator, stepper::AbstractTimeStepper; truncalg, isreversed
)
    # Evolve, all sites

    bonds = bonds(pepo)

    if isreversed
        bonds = Base.reverse(bonds)
    end

    # beliefpropagation(latticetensors(pepo), currentstepper(stepper))
    #
    pepo = evolve!(pepo, stepper)

    # hermitianize!.(evolvedtensors(pepo))
    # normalize!.(evolvedtensors(pepo))

    _, infid = quickupdate!(pepo, truncator; truncalg=truncalg)

    # _, infid = bpupdate!(pepo, truncator, stepper; truncalg=truncalg)

    # symmetric = issymmetric(pepo)
    #     interpepo = copyevolved(pepo)
    #
    #     # Calculate the truncator for the first time.
    #     # truncator = renormalize(interpepo, truncator)
    #
    #     for col in eachcol(bonds)
    #         for bond in col
    #             # Tensors have changed bond dimensions so need to construct new env
    #             truncator = renormalize(interpepo, truncator)
    #
    #             left_site, right_site = interpepo[bond]
    #
    #             iso = isometries(interpepo, truncator, bond, truncalg)
    #
    #             setweights!(pepo, iso.s, bond)
    #
    #             interstate = evolvedtensors(interpepo)
    #
    #             #  = A = ->  = A -
    #             interstate[bond.left] = partialtruncate(left_site, iso)
    #             #  = B = ->  - B =
    #             interstate[bond.right] = partialtruncate(iso, right_site)
    #
    #             if symmetric
    #                 # If our unitcell is symmetric, then copy the updated bond to all
    #                 # other columns with an appropriate shift
    #                 copysymmetric!(interstate)
    #             end
    #         end
    #         if symmetric
    #             break
    #         end
    #     end
    #
    #     truncateinto!(pepo, interpepo)

    hermitianize!(pepo)
    quicknormalize!(pepo)

    return pepo
end

function deepcopysymmetric!(array::AbstractMatrix)
    for c in axes(array, 2)
        if c == 1
            continue
        end
        # This is safe as the array views reference different parts of the same array
        # @views circshift!(array[:, c], array[:, 1], -(c - 1))
        @views foreach(copy!, array[:, c], circshift(array[:, 1], -(c - 1)))
    end
    return array
end
function copysymmetric!(array::AbstractMatrix)
    for c in axes(array, 2)
        if c == 1
            continue
        end
        # This is safe as the array views reference different parts of the same array
        @views circshift!(array[:, c], copy.(array[:, 1]), -(c - 1))
    end
    return array
end

function updatestep!(pepo::LocalUpdatablePEPO, truncator, map; isreversed=false, kwargs...)
    symmetric = ischeckerboard(pepo)
    bonds = bonds(pepo)

    if isreversed
        bonds = Base.reverse(bonds)
    end

    # renormalize!(truncator)

    for col in eachcol(bonds)
        for bond in col

            # truncator = renormalize(latticetensors(pepo), truncator)
            renormalize!(truncator)
            # Update the states lying on the bond
            updatebond!(pepo, truncator, map, bond; kwargs...)

            # quicknormalize!(pepo)
            #
            # hermitianize!(pepo)

            if symmetric
                deepcopysymmetric!(tensors(pepo))
            end
        end

        if symmetric
            break
        end
    end

    # Quick normalization for algorithm stability. Does not ensure tr(ρ) = 1
    quicknormalize!(pepo)

    hermitianize!(pepo)

    return pepo
end

function updatebond!(
    pepo::LocalUpdatablePEPO, truncator, stepper, bond; truncalg, kwargs...
)
    # Evolve the sites on the bond given by `bond` according to `stepper`.
    evolve!(pepo, stepper, bond)

    # Find the truncation isometries.
    isoms, infid = isometries(pepo, truncator, bond, truncalg; kwargs...)

    # Truncate the pepo according to the truncation isometries
    truncate!(pepo, isoms)

    return pepo, infid
end

function evolve!(pepo::LocalUpdatablePEPO, stepper::LocalStepper, bond)
    # Apply the time stepper, updating the evolved tensors on `bond`
    setcurrentbond!(pepo, bond)
    applystepper!(pepo, stepper, bond)
    # Set the current bond of the pepo to `bond`
    setisevolved!(pepo, true)

    return pepo
end

function evolve!(pepo::GlobalUpdatablePEPO, stepper)
    applystepper!(pepo, stepper)
    setisevolved!(pepo, true)
    return pepo
end

function applybothtrotter!(pepo::GlobalUpdatablePEPO, stepper; kwargs...)
    setisevolved!(pepo, false)

    pin = tensors(pepo)
    pev = currentbuffer(pepo)

    applybothtrotter!(pev, pin, stepper; kwargs...)

    return pepo
end
function applybothtrotter!(pev, pin, gates::LocalStepper; kwargs...)
    return _applybothtrotter!.(pev, pin, gates.u, gates.v; kwargs...)
end
# function applybothtrotter!(pev, pin, gates::SimpleGates)
#     return _applybothtrotter!.(pev, pin, gates.east, gates.west, gates.loc)
# end
function applybothtrotter!(pev, pin, mpo::TwoBondStepper; kwargs...)
    return _applybothtrotter!.(pev, pin, mpo.t)
end

function _applybothtrotter!(pev, pin, op_l, op_r; reverse=false)
    if reverse
        @tensoropt begin
            p_out[ook oob; e r s w l n] :=
                op_r[ook oob ook oob; l] * pin[ik ib; e s w n] * op_l[ok ob ik ib; r]
        end
    else
        @tensoropt begin
            p_out[ook oob; e r s w l n] :=
                op_r[ok ob ik ib; l] * pin[ik ib; e s w n] * op_l[ook oob ok ob; r]
        end
    end
    mergeinto!(pev, p_out)
    return pev
end
function _applybothtrotter!(pev, pin, op_l, op_r, loc)
    # west -> easts -> loc
    @tensoropt begin
        p_out[ook oob; e r s w l n] :=
            op_r[ok ob ik ib; l] * pin[ik ib; e s w n] * op_l[ook oob ok ob; r]
    end
    mergeinto!(pev, loc * p_out)
    return pev
end
function _applybothtrotter!(pev, pin, mpo)
    # west -> easts -> loc
    @tensoropt begin
        p_out[ok ob; e ee s w ww n] := mpo[ok ob ik ib; ee ww] * pin[ik ib; e s w n]
    end

    mergeinto!(pev, p_out)
    return pev
end

function applytrotter!(pepo::LocalUpdatablePEPO, operator, bond)
    # Get the current left and right pepo tensors
    setisevolved!(pepo, false)

    p_l, p_r = pepo[bond]
    op_l, op_r = operator[bond]

    # Get the evolved pepo tensors such that we can write into them
    pe_l, pe_r = evolvedpair(pepo)

    @tensoropt begin
        q_l[ok ob; e x s w n] := op_l[ok ob ik ib; x] * p_l[ik ib; e s w n]
        q_r[ok ob; e s w x n] := op_r[ok ob ik ib; x] * p_r[ik ib; e s w n]
    end

    mergeinto!(pe_l, q_l)
    mergeinto!(pe_r, q_r)

    return pepo
end
