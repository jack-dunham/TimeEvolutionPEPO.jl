# This file defines the top level `timestep` function

newevolution(pepo, steps, method) = newevolution(eltype(steps), pepo, steps, method)
newevolution(::Type{<:AxialStepper}, pepo, steps, method) = AxialPEPO(pepo)
newevolution(::Type{<:AbstractTimeStepper}, pepo, steps, method) = pepo
function newevolution(::Type{<:LocalStepper}, pepo, steps, method::TEBD{<:Union{FET,NTU}})
    return AxialPEPO(pepo)
end

# Default
function timestep!(pepo, truncator, stepper, truncalg; kwargs...)
    timestep!(pepo, first(truncator), stepper, truncalg; kwargs...)

    return pepo
end
function _timestep!(pepo::PEPO, truncator, stepper::LocalStepper, truncalg::SU; kwargs...)
    size(pepo) < (2, 2) && throw(ArgumentError(""))

    axis = stepper.axis

    for bond in bonds(pepo, axis)
        env = environment!(pepo, SimpleUpdateState(), bond)

        pair = evolve(pepo, stepper, bond)

        newpair, newweight = updatetensors(pair, env, truncalg, bond)

        pepo[bond] = newpair

        setweight!(pepo, newweight, bond; copy=false)

        # Normalizes each tensor using `norm`; not the same as normalizing the quantum
        # state. This is for algorithm stability reasons only
        quicknormalize!(pepo)
        hermitianize!(pepo)
    end

    return pepo
end

function timestep!(pepo!::PEPO, truncator, stepper::FullStepper, truncalg::SU; kwargs...)
    isoms = newisometries(pepo!, stepper)

    newpepo = evolve(pepo!, stepper)

    simple_update_gauge_parallel!(newpepo)

    truncate_bond_matrices!(pepo!, newpepo, isoms)

    quicknormalize!(pepo!)
    hermitianize!(pepo!)

    return pepo!
end

# function timestep!(
#     pepo!::AxialPEPO, truncator, stepper::AxialStepper, truncalg::SU; kwargs...
# )
#     axis = stepper.axis
#
#     println(axis)
#
#     # Evolve and then truncate each bond in sequence
#     if pepo!.currentaxis != axis
#         setaxis!(pepo!, stepper.axis)
#     end
#
#     isoms = newisometries(tensors(pepo!), stepper)
#
#     newpepo = evolve(pepo!, stepper)
#
#     simple_update_gauge_parallel!(newpepo)
#
#     truncate_bond_matrices!(pepo!, newpepo, isoms)
#
#     quicknormalize!(pepo!)
#     hermitianize!(pepo!)
#
#     return pepo!
# end

function timestep!(
    pepo::AxialPEPO, truncator, stepper::LocalStepper, truncalg::Union{NTU,FET}; kwargs...
)
    axis = stepper.axis

    # Evolve and then truncate each bond in sequence
    if pepo.currentaxis != axis
        setaxis!(pepo, axis)
    end

    infidelity = 0

    axisbonds = bonds(pepo, 1)

    checker = true

    if checker
        altbonds = circshift(axisbonds[:, 2], 1)
        axisbonds = axisbonds[:, 1]
    end

    for (i, bond) in enumerate(axisbonds)
        D = bondspace(pepo, bond)

        # Evolved the pair of tensors connected by `bond`, leaving the original pair
        # unmutated.
        pair = evolvemerge(pepo, stepper, bond)


        if truncalg.rankreduce
            init_weights = environment!(pepo.parent, SimpleUpdateState(), bond)
            if axis == 2
                init_weights = permutedims(init_weights)
            end

            pair, initiso = updatetensors(pair, init_weights, SU(), bond)

            env = environment!(pepo, truncator[axis], bond)
            (u, v), s, _ = metric_isometries_from_simple_update!(
                pair, initiso, env, bond;
                D=D, optalg=truncalg.optalg, xi=truncalg.ξ
            )

        else
            env = environment!(pepo, truncator[axis], bond)
            (u, v), s, _ = isometries(pair, env, bond, truncalg; D=D)
        end

        oldA, oldB = pepo[bond]

        a1, a2, a3, a4 = domain(oldA)
        b1, b2, b3, b4 = domain(oldB)

        Du = domain(u)
        Dv = codomain(v)

        if bondaxis(bond) == 1
            domA = Du * a2 * a3 * a4
            domB = b1 * b2 * Dv' * b4
        elseif bondaxis(bond) == 2
            domA = a1 * Du * a3 * a4
            domB = b1 * b2 * b3 * Dv'
        end

        newA = similar(oldA, codomain(oldA), domA)
        newB = similar(oldB, codomain(oldB), domB)

        newpair = newA => newB

        truncateto!(newpair, pair, u => v; axis=bondaxis(bond))

        pepo[bond] = newpair

        if checker
            altbond = altbonds[i]

            pepo[altbond] = mapreduce(copy, =>, newpair)

            setweight!(pepo, s, altbond; copy=false)
        end

        setweight!(pepo, s, bond; copy=false)

        normalize!(pepo)
        hermitianize!(pepo)
    end

    # if axis == 1
    #     simple_update_gauge!(pepo.parent; reset = true)
    # end

    return pepo, infidelity
end

function timestep!(
    pepo::AxialPEPO, truncator, stepper::AxialStepper, truncalg::SU; kwargs...
)
    axis = stepper.axis

    # Evolve and then truncate each bond in sequence
    if pepo.currentaxis != axis
        setaxis!(pepo, axis)
    end

    infidelity = similar(ITC.getdata(pepo))

    inweights = axis == 1 ? deepcopy(weights(pepo)) : permutedims(weights(pepo))

    evpepo = evolve(pepo, stepper)

    axial = true

    if !axial
        if 1 in size(evpepo)
            simple_update_gauge_parallel!(evpepo; maxiter=500, tol=1e-10)
        else
            simple_update_gauge!(evpepo)
        end
        inweights = weights(evpepo)
    else
        inweights = weights(pepo)
    end

    isoms = map(bonds(pepo, 1)) do bond
        D = bondspace(pepo, bond)

        # Evolved the pair of tensors connected by `bond`, leaving the original pair
        # unmutated.

        pair = evpepo[bond]

        iso, s, _ = isometries(pair, inweights, bond, truncalg; D=D, axial=axial)

        setweight!(pepo, s, bond; copy=false)

        return iso
    end

    isoe = broadcast(first, isoms)
    isow = broadcast(last, circshift(isoms, (1, 0)))

    broadcast(pepo, evpepo, isoe, isow) do dst, src, e, w
        truncateto!(dst, src, e, w)
    end

    quicknormalize!(pepo)
    hermitianize!(pepo)

    return pepo, infidelity
end

function timestep!(
    pepo::AxialPEPO, truncator, stepper::AxialStepper, truncalg::NTU; kwargs...
)
    axis = stepper.axis

    if pepo.currentaxis != axis
        setaxis!(pepo, stepper.axis)
    end

    infidelity = similar(pepo)

    evpepo = evolve(pepo, stepper)

    T = tensormaptype(spacetype(pepo), 1, 1, scalartype(pepo))

    isomu = similar(getaxis(pepo, axis), T)
    isomv = similar(getaxis(pepo, axis), T)

    # This can happen in parallel
    Threads.@threads for bond in bonds(pepo, 1)
    # isoms = map(bonds(pepo, 1)) do bond
        D = bondspace(pepo, bond)

        env = environment!(evpepo, truncator[axis], bond)

        pair = evpepo[bond]

        (u, v), s, _ = isometries(pair, env, bond, truncalg; D=D, parallel=true)

        setweight!(pepo, s, bond; copy=false)

        isomu[first(bond)] = u
        isomv[last(bond)] = v

        # return u => v
    end

    # isomu = broadcast(first, isoms)
    # isomv = broadcast(last, circshift(isoms,(1,0)))

    broadcast(pepo, evpepo, isomu, isomv) do dst, src, u, v
        truncateto!(dst, src, u, v)
    end

    quicknormalize!(pepo)
    hermitianize!(pepo)

    return pepo, infidelity
end
function timestep!(pepo!::PEPO, truncator!::Isometries, stepper, truncalg::IterativeSU)
    iterative_simple_update_prototype!(
        pepo!,
        truncator!,
        stepper;
        maxiter=truncalg.maxiter,
        tol=truncalg.tol,
        recycle=truncalg.recycle,
        sequential=truncalg.sequential,
        gauge=truncalg.gauge,
        xi=truncalg.xi,
    )

    return pepo!
end
function timestep!(pepo!::PEPO, truncator!::Isometries, stepper, truncalg::IterativeSUTest)
    iterative_simple_update_prototype!(
        pepo!,
        truncator!,
        stepper;
        maxiter=truncalg.maxiter,
        tol=truncalg.tol,
        recycle=truncalg.recycle,
        sequential=truncalg.sequential,
        gauge=truncalg.gauge,
        xi=truncalg.xi,
        check_fidelity=true,
    )

    return pepo!
end

## TODO: Old methods below

# Parallel Semi Parallel
function _timestep!(pepo::AxialPEPO, truncator, stepper, truncalg; kwargs...)
    axis = stepper.axis

    pepo.currentaxis == axis || setaxis!(pepo, axis)

    evolvedtensors = evolve(pepo, stepper)

    infidelity = 0

    buffer = axisbuffer(pepo)

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

# function itrtimestep_deprec!(
#     pepo,
#     truncator,
#     stepper::Vector{<:FullStepper{<:Any,<:SparseTensorMap}},
#     ::Number;
#     kwargs...,
# )
#     stepper = only(stepper)
#
#     infidelity = 0
#
#     buff = currentbuffer(pepo)
#
#     weightsbuff = deepcopy(weights(pepo))
#     oldweights = deepcopy(weights(pepo))
#
#     isoms_data = broadcast(buff, tensors(stepper)) do dst, src
#         return stepstitchisometries(storagetype(dst), domain(dst), domain(src))
#     end
#
#     isoms = Isometries(isoms_data)
#     isomsbuff = similar(isoms)
#
#     local isoe
#     local isow
#
#     local ison
#     local isos
#
#     schedule = :sequential
#     # schedule = :parallel
#
#     for i in 1:15
#
#         isomh = broadcast(bonds(buff, 1)) do bond
#             reducedpair = localevolve(buff[bond], stepper, isoms, bond)
#
#             foreach(normalize!, reducedpair)
#
#             D = bondspace(pepo, bond)
#             (u, v), s, _ = simpleupdate(reducedpair, weights(pepo), bond; trunc=truncspace(D))
#
#
#             if schedule === :sequential
#                 setindex!(isoms, u => v, bond)
#                 setweight!(pepo, s, bond; copy=true)
#             elseif schedule === :parallel
#                 setindex!(isomsbuff, u => v, bond)
#                 weightsbuff[bond] = s
#             end
#
#             return (u, v)
#         end
#
#         isomv = broadcast(bonds(buff, 2)) do bond
#             reducedpair = localevolve(buff[bond], stepper, isoms, bond)
#
#             foreach(normalize!, reducedpair)
#
#             D = bondspace(pepo, bond)
#             (u, v), s, _ = simpleupdate(reducedpair, weights(pepo), bond; trunc=truncspace(D))
#
#
#             if schedule === :sequential
#                 setindex!(isoms, u => v, bond)
#                 setweight!(pepo, s, bond; copy=true)
#             elseif schedule === :parallel
#                 setindex!(isomsbuff, u => v, bond)
#                 weightsbuff[bond] = s
#             end
#
#             return (u, v)
#         end
#
#         # foreach(bonditer(buff)) do bond
#         #     s = weightsbuff[bond]
#         #     setweight!(pepo, s, bond; copy=false)
#         # end
#
#         # isoe = broadcast(first, isomh)
#         # isow = broadcast(last, circshift(isomh, (1, 0)))
#         #
#         # isos = broadcast(first, permutedims(isomv))
#         # ison = broadcast(last, permutedims(circshift(isomv, (1, 0))))
#         #
#         #
#         # isoms = SquareLattice(
#         #     broadcast(tuple, isoe, isos, isow, ison)
#         # )
#
#
#         if schedule === :sequential
#             tol = tolerance!(weightsbuff, weights(pepo), 1.0)
#         else
#             elcopy!(isoms, isomsbuff)
#             tol = tolerance!(weights(pepo), weightsbuff, 1.0)
#         end
#
#         @warn "Got $tol after $i"
#
#         if tol < 1e-6 && i > 0
#             break
#         end
#
#         # temppepo = deepcopy(pepo)
#         #
#         # broadcast(
#         #     tensors(temppepo), tensors(stepper), isoe, isos, isow, ison
#         # ) do dst, op, e, s, w, n
#         #     spten = truncateto(dst, dst, op, e, s, w, n)
#         #     mergeinto!(dst, spten)
#         # end
#         #
#         # gauge1, gauge2 = simple_update_gauge_isometries!(temppepo)
#         #
#         # ge = broadcast(first, gauge1)
#         # gw = broadcast(last, circshift(gauge1, (1, 0)))
#         #
#         # gs = broadcast(first, permutedims(gauge2))
#         # gn = broadcast(last, permutedims(circshift(gauge2, (1, 0))))
#         #
#         # gisoms = SquareLattice(
#         #     broadcast(tuple, ge, gs, gw, gn)
#         # )
#
#         # isoms = broadcast(isoms, gisoms) do iso, g
#         #     e1, s1, w1, n1 = iso
#         #     e2, s2, w2, n2 = g
#         #
#         #     e = e1 * e2
#         #     s = s1 * s2
#         #
#         #     w = w2 * w1
#         #     n = n2 * n1
#         #
#         #     return (e,s,w,n)
#         # end
#         #
#         # foreach(bonditer(buff)) do bond
#         #     s = weights(temppepo)[bond]
#         #     setweight!(pepo, s, bond; copy=true)
#         # end
#     end
#
#     # broadcast(
#     #     buff, tensors(stepper), isoe, isos, isow, ison
#     # ) do dst, op, e, s, w, n
#     #     spten = truncateto(dst, dst, op, e, s, w, n)
#     #     mergeinto!(dst, spten)
#     # end
#
#     for I in eachindex(buff)
#
#         dst = buff[I]
#         op = tensors(stepper)[I]
#         e, s, w, n = isoms[I]
#
#         spten = truncateto(dst, dst, op, e, s, w, n)
#
#         mergeinto!(dst, spten)
#     end
#
#     size(buff) == (1, 1) && minimalcanonicalform!(buff)
#
#     size(buff) != (1, 1) && simple_update_gauge!(pepo; maxiter=100)
#
#     pepo.state = 0
#
#     # minimalcanonicalform!(tensors(pepo))
#
#     quicknormalize!(pepo.parent)
#     hermitianize!(pepo.parent)
#
#     return pepo, infidelity
# end

function _timestep!(
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

    reducedbuffer = broadcast(baretruncation!, similar.(axisbuffer(pepo)), buff)

    xisoms = stitchisometries.(axisbuffer(pepo), buff)
    yisoms = stitchisometries.(permutetranspose(axisbuffer(pepo)), ybuff)

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

    copy!.(axisbuffer(pepo), reducedbuffer)

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
    dst = axisbuffer(otherpepo)
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

    dstpepo = axisbuffer(pepo)
    srcpepo = currentbuffer(pepo)

    site = dstpepo[1]

    infidelity = similar(dstpepo, Float64)
    bondh = Bond(CartesianIndex(1, 1), CartesianIndex(2, 1))
    bondv = Bond(CartesianIndex(1, 1), CartesianIndex(1, 2))

    # mts = beliefpropagation(srcpepo)
    op = axisbuffer(stepper[1])
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

    dstpepo = axisbuffer(pepox)
    srcpepo = currentbuffer(pepox)

    mts = beliefpropagation(dstpepo, axisbuffer(stepper))

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
        step = axisbuffer(stepper)[1]
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
