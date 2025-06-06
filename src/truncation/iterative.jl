@kwdef mutable struct IterativeSU <: AbstractTruncationScheme
    maxiter::Int = 50
    tol::Float64 = 1e-8
    sequential::Bool = false
    recycle::Bool = true
    gauge::Int = 0
    zeta::Float64 = Inf
end
@kwdef mutable struct IterativeSUTest <: AbstractTruncationScheme
    maxiter::Int = 50
    tol::Float64 = 1e-8
    sequential::Bool = false
    recycle::Bool = true
    gauge::Int = 0
    zeta::Float64 = Inf
end

struct Isometries{G,N,U<:AbsTen{N},V<:TenAbs{N}}
    e::UnitCell{G,U,Matrix{U}}
    s::UnitCell{G,U,Matrix{U}}
    w::UnitCell{G,V,Matrix{V}}
    n::UnitCell{G,V,Matrix{V}}
end

Base.length(iso::Isometries) = length(iso.e)

function getmessages(iso::Isometries, bond; axis=bondaxis(bond))
    (e1, s1, w1, n1), (e2, s2, w2, n2) = iso[bond]
    if axis == 1
        return e1, w2
    elseif axis == 2
        return s1, n2
    end
end
function Base.iterate(iso::Isometries, state=1)
    i = state

    if state > length(iso)
        return nothing
    else
        return (iso.e[i], iso.s[i], iso.w[i], iso.n[i]), state + 1
    end
end

function Base.similar(isoms::Isometries)
    e, s, w, n = map(similar, unpack(isoms))
    return Isometries(e, s, w, n)
end

function newisometries(buffer, stepper)
    isomsinit = broadcast(buffer, stepper) do dst, src
        return stepstitchisometries(storagetype(dst), domain(dst), domain(src))
    end
    return Isometries(isomsinit)
end

function newsplitisometries(buffer, stepper)
    isomsinit = broadcast(buffer, getaxis(stepper)) do dst, src
        return stitchisometries(storagetype(dst), domain(dst), domain(src))
    end
    return Isometries(isomsinit)
end

function new_bp_isometries(buffer, stepper)
    isomsinit = broadcast(buffer, getaxis(stepper)) do dst, src
        return bpisometries(storagetype(dst), domain(dst), domain(src))
    end
    return Isometries(isomsinit)
end

function elcopy!(dst::Isometries, src::Isometries)
    foreach(unpack(dst), unpack(src)) do d, s
        elcopy!(d, s)
    end
    return dst
end

unpack(isoms::Isometries) = (isoms.e, isoms.s, isoms.w, isoms.n)

function Isometries(data::AbstractMatrix{Tuple{U,U,V,V}}) where {U,V}
    datal = UnitCell(data)

    E = similar(data, U)
    S = similar(data, U)
    W = similar(data, V)
    N = similar(data, V)

    for I in eachindex(datal)
        e, s, w, n = datal[I]

        E[I] = e
        S[I] = s
        W[I] = w
        N[I] = n
    end

    return Isometries(E, S, W, N)
end

function Base.setindex!(isoms::Isometries, val::Union{Pair,NTuple{2}}, bond::Bond)
    axis = bondaxis(bond)

    i1, i2 = bond
    u, v = val

    if axis == 1
        isoms.e[i1] = u
        isoms.w[i2] = v
    elseif axis == 2
        isoms.s[i1] = u
        isoms.n[i2] = v
    else
        throw(ErrorException(""))
    end

    return isoms
end

Base.getindex(isoms::Isometries, I...) = map(uv -> getindex(uv, I...), unpack(isoms))
Base.getindex(isoms::Isometries, bond::Bond) = isoms[first(bond)] => isoms[last(bond)]

# stripped down version
function iterative_simple_update!(pepo!::PEPO, isometries!, stepper, truncalg::IterativeSU)
    oldweights = deepcopy(bondweights(pepo!))

    i = 0

    while i < truncalg.maxiter
        for bond in eachbond(pepo!)
            trunc = truncspace(bondspace(pepo!, bond))

            evolvedpepo = evolve(pepo!, stepper, isometries!, bond)

            pair = evolvedpepo[bond]

            newiso, newweight = simpleupdate(pair, bondweights(pepo!), bond; trunc=trunc)

            isometries[bond] = newiso
            setweight!(pepo!, newweight, bond; copy=false)
        end

        # Check convergence of singular values 
        conv = tolerance!(oldweights, bondweights(pepo!))

        conv < truncalg.tol && break
    end

    truncate!(pepo!, stepper, isometries!)

    quicknormalize!(pepo!)
    hermitianize!(pepo!)

    return pepo!
end

function truncate!(pepo, stepper, isometries)
    buff = currentbuffer(pepo)

    for I in eachindex(buff)
        dst = buff[I]
        op = getaxis(stepper)[I]
        e, s, w, n = isometries[I]

        truncateto!(dst, copy(dst), op, e, s, w, n)

        # mergeinto!(dst, spten)
    end

    return pepo
end

#
# function timestep!(
#     pepo!::AxialPEPO, truncator!::Tuple, stepper, truncalg::IterativeSU
# )
#     if pepo.currentaxis != stepper.axis
#         setaxis!(pepo, stepper.axis)
#     end
#
#     axis = pepo!.currentaxis
#
#     iterative_simple_update!(
#         pepo!,
#         truncator![axis],
#         stepper;
#         maxiter=truncalg.maxiter,
#         tol=truncalg.tol,
#         recycle=truncalg.recycle,
#         sequential=truncalg.sequential,
#     )
#
#     return pepo!
# end
function timestep!(pepo!::AbstractPEPO, truncator!::Isometries, sequence, truncalg::NTU)
    iterative_neighbourhood_update!(pepo!, truncator!, sequence;)

    return pepo!
end

function iterative_simple_update_step(pepo, stepper, isometries, bond; kwargs...)
    pair = pepo[bond]

    reducedpair = evolve(pair, stepper, isometries, bond)

    foreach(normalize!, reducedpair)

    D = bondspace(pepo, bond)

    (u, v), s, infid = simpleupdate(
        reducedpair, bondweights(pepo), bond; trunc=truncspace(D), kwargs...
    )

    return (u, v), s, infid
end

function newtruncators(pepo, sequence, ::Union{IterativeSU,IterativeSUTest})
    rv = map(sequence.body) do stepper
        iso = newisometries(pepo, stepper)
        return iso
    end
    return (rv, rv)
end

function iterative_simple_update_prototype!(
    pepo!,
    truncator!,
    stepper;
    maxiter,
    tol,
    sequential,
    recycle,
    gauge,
    zeta,
    check_fidelity=false,
)
    mode = :none

    isunit = axis -> any(D -> dim(D) == 1, bondspace.(Ref(pepo!), bonds(pepo!, axis)))

    if mode === :quick || isunit(1) || isunit(2)
        newpepo = evolve(pepo!, stepper)

        simple_update_gauge_parallel!(newpepo)

        truncate_bond_matrices!(pepo!, newpepo, truncator!)

        quicknormalize!(pepo!)
        hermitianize!(pepo!)

        return pepo!
    end

    if mode === :bp
        buff = currentbuffer(pepo!)
        messages = sqrtbeliefpropagation(buff, stepper)
        isoms = newsplitisometries(buff, stepper)

        for bond in eachbond(buff)
            D = bondspace(buff, bond)

            (u, v), s = vidalisometries(messages, bond; trunc=truncspace(D))
            setindex!(isoms, u => v, bond)
        end
        for I in eachindex(buff)
            dst = buff[I]
            op = getaxis(stepper)[I]
            e, s, w, n = isoms[I]

            truncateto!(dst, dst, op, e, s, w, n)

            # mergeinto!(dst, spten)
        end

        quicknormalize!(pepo!)
        hermitianize!(pepo!)

        return pepo!
    end

    infidelity = 0

    isoms = recycle ? truncator! : newisometries(pepo!, stepper)

    # Buffers 
    newisoms = similar(isoms)
    newweights = deepcopy(bondweights(pepo!))

    S0 = bondentropy(newweights)

    println("Original bond entropy $S0")

    conv = Inf

    bisol = false
    bisoldone = false
    skipnext = false

    i = 0

    local ent0

    while i < 1 * maxiter
        i += 1

        bonds = eachbond(pepo!)

        if sequential
            itr = iseven(i) ? Iterators.reverse(bonds) : bonds

            for bond in itr
                (u, v), s, _ = iterative_simple_update_step(
                    pepo!, stepper, isoms, bond; zeta=Inf
                )

                setindex!(isoms, u => v, bond)
                setweight!(pepo!, s, bond; copy=false)
            end

            new = newweights
            old = bondweights(pepo!)

        else
            Threads.@threads for bond in collect(bonds)
                if check_fidelity && bond == Bond((1, 1), (1, 2))
                    evolvedpepo = evolve(deepcopy(pepo!), stepper)

                    copybuff = deepcopy.(buff)

                    let
                        foreach(eachindex(buff)) do I
                            dst = copybuff[I]
                            op = stepper[I]
                            e, s, w, n = isoms[I]

                            truncateto!(dst, copy(dst), op, e, s, w, n)

                            # mergeinto!(dst, spten)
                        end
                    end

                    evbuff = evolvedpepo

                    alg = VUMPS(; bonddim=16, tol=1e-10, maxiter=20, verbose=false)

                    overlap = UnitCell(
                        map(copybuff, evbuff) do c, e
                            return CompositeTensor(c, e')
                        end,
                    )

                    vumps = ITC.newrenormalization(overlap; alg=alg)

                    ITC.renormalize!(vumps)

                    vumps = vumps.runtime

                    mps = vumps.mps

                    AC = ITC.getcentral(mps)

                    FL = vumps.fixedpoints.left
                    FR = vumps.fixedpoints.right

                    rv = tracecontract_two_layer(
                        FL[1, 1], FR[1, 1], AC[1, 1], AC[1, 2], copybuff[1, 1], evbuff[1, 1]
                    )

                    overlap = UnitCell(
                        map(copybuff, copybuff) do c, e
                            return CompositeTensor(c, e')
                        end,
                    )

                    vumps = ITC.newrenormalization(overlap; alg=alg)
                    ITC.renormalize!(vumps)

                    vumps = vumps.runtime

                    mps = vumps.mps

                    AC = ITC.getcentral(mps)

                    FL = vumps.fixedpoints.left
                    FR = vumps.fixedpoints.right

                    rvnorm1 = tracecontract_two_layer(
                        FL[1, 1],
                        FR[1, 1],
                        AC[1, 1],
                        AC[1, 2],
                        copybuff[1, 1],
                        copybuff[1, 1],
                    )

                    overlap = UnitCell(
                        map(evbuff, evbuff) do c, e
                            return CompositeTensor(c, e')
                        end,
                    )

                    vumps = ITC.newrenormalization(overlap; alg=alg)
                    ITC.renormalize!(vumps)

                    vumps = vumps.runtime

                    mps = vumps.mps

                    AC = ITC.getcentral(mps)

                    FL = vumps.fixedpoints.left
                    FR = vumps.fixedpoints.right

                    rvnorm2 = tracecontract_two_layer(
                        FL[1, 1],
                        FR[1, 1],
                        AC[1, 1],
                        AC[1, 2],
                        currentbuffer(evolvedpepo)[1, 1],
                        currentbuffer(evolvedpepo)[1, 1],
                    )

                    @warn "overlap:  " 1 - rv / sqrt(rvnorm1 * rvnorm2)
                    @warn "overlap:  " rv / sqrt(rvnorm1)
                end
                (u, v), s, _ = iterative_simple_update_step(
                    pepo!, stepper, isoms, bond; zeta=Inf
                )

                setindex!(newisoms, u => v, bond)
                newweights[bond] = s
            end

            new = bondweights(pepo!)
            old = newweights

            elcopy!(isoms, newisoms)
        end

        if skipnext
            @warn "skipping bad solution"
            skipnext = false
            continue
        end

        # comment out this
        # skipnext = true

        oldconv = conv
        conv = tolerance!(new, old, 1.0)

        diff = abs(oldconv - conv) / max(oldconv, conv)

        @warn "Got $conv after $i; diff = $(diff)"

        S = bondentropy(new)
        println(S)

        if conv < tol || i > maxiter
            break
        end

        if bisol
            if !bisoldone
                ent1 = 0
                broadcast(S, S0) do ss, ss0
                    ent1 += abs(ss - ss0)
                end

                if ent1 < ent0
                    bisoldone = true
                    skipnext = true
                else
                    ent0 = ent1
                end
            else
                skipnext = true
            end
            @warn "Trying every other solution..."
        else
            if Inf * diff < conv
                ent0 = 0
                broadcast(S, S0) do ss, ss0
                    ent0 += abs(ss - ss0)
                end

                bisol = true
            end
        end

        gauge < 2 && continue

        wtgisometries!(isoms, pepo!, stepper)

        continue

        temppepo = deepcopy(pepo!)

        for I in eachindex(buff)
            dst = getaxis(temppepo)[I]
            op = getaxis(stepper)[I]
            e, s, w, n = isoms[I]

            truncateto!(dst, deepcopy(dst), op, e, s, w, n)

            # mergeinto!(dst, spten)
        end

        gauge1, gauge2, info = simple_update_gauge_isometries!(
            temppepo; tol=1e-10, verbose=false
        )
        @info "Gauged to $(info[1]) after $(info[2]) iterations"

        ge = broadcast(first, gauge1)
        gw = broadcast(last, circshift(gauge1, (1, 0)))
        #
        gs = broadcast(first, permutedims(gauge2))
        gn = broadcast(last, permutedims(circshift(gauge2, (1, 0))))

        gisoms = SquareLattice(broadcast(tuple, ge, gs, gw, gn))

        isoms = broadcast(eachindex(buff)) do I
            e1, s1, w1, n1 = isoms[I]
            e2, s2, w2, n2 = gisoms[I]

            e = e1 * e2
            s = s1 * s2

            w = w2 * w1
            n = n2 * n1

            return (e, s, w, n)
        end

        isoms = Isometries(UnitCell(isoms))

        foreach(eachbond(buff)) do bond
            s = bondweights(temppepo)[bond]
            setweight!(pepo!, s, bond; copy=true)
        end
    end

    # broadcast(
    #     buff, tensors(stepper), isoe, isos, isow, ison
    # ) do dst, op, e, s, w, n
    #     spten = truncateto(dst, dst, op, e, s, w, n)
    #     mergeinto!(dst, spten)
    # end

    @info "gauging..."
    # simple_update_gauge!(pepo!; maxiter=200, tol=1e-10, verbose=false, reset=false)
    #=
    for numiter in 1:1
        conv = 0

        temppepo = deepcopy(pepo!)

        # Formulate the hypthoetical state using isometries isoms
        for I in eachindex(buff)
            dst = tensors(temppepo)[I]
            op = tensors(stepper)[I]
            let (e, s, w, n) = isoms[I]
                truncateto!(dst, deepcopy(dst), op, e, s, w, n)
            end
        end

        for (i, bond) in enumerate(eachbond(pepo!))

            # Compute the environment
            env = weightenvironment(temppepo[bond], weights(pepo!), bond)

            if bondaxis(bond) == 1
                (u, _, _, _), (_, _, v, _) = isoms[bond]
            elseif bondaxis(bond) == 2
                (_, u, _, _), (_, _, _, v) = isoms[bond]
            end

            s = weights(temppepo)[bond]
            oldS = bondentropy(weights(pepo!))[i][bondaxis(bond)]
            barewtg!(u, s, v, env)

            newconv = abs(bondentropy(weights(pepo!))[i][bondaxis(bond)] - oldS)

            newconv > conv && (conv = newconv)
        end

        tolerance!(weights(pepo!), weights(temppepo), 1.0)

        println(conv)
        if conv < 1e-8
            break
        end
    end
    =#

    for I in eachindex(pepo!)
        dst = pepo![I]
        op = stepper[I]
        let (e, s, w, n) = isoms[I]
            if zeta < Inf
                @info "rescaling " xi
                e = e * sqrt(rescale!(id(domain(e)), id(domain(e)), zeta))
                s = s * sqrt(rescale!(id(domain(s)), id(domain(s)), zeta))
                w = sqrt(rescale!(id(codomain(w)), id(codomain(w)), zeta)) * w
                n = sqrt(rescale!(id(codomain(n)), id(codomain(n)), zeta)) * n
            end

            truncateto!(dst, copy(dst), op, e, s, w, n)
        end

        # mergeinto!(dst, spten)
    end

    # xi < Inf && simple_update_gauge_parallel!(pepo!; tol=1e-10)

    # for (i, p) in enumerate(tensors(pepo!))
    #     # copy!(p, 0.5 * (p + permutedom(p, (2,1,4,3))))
    #     w = weights(pepo!)[i]
    #     wnew = 0.5 * (w[1] + w[2])
    #     maxnormalize!(wnew)
    #     copy!(w[1], wnew)
    #     copy!(w[2], wnew)
    # end

    if gauge == 1
        @info "Gauge fixing"
        size(pepo!) == (1, 1) && minimalcanonicalform!(pepo!; maxiter=2000)
        # size(buff) == (1, 1) &&
        #     simple_update_gauge!(pepo!; maxiter=200, tol=1e-10, verbose=false, reset=false)
        size(pepo!) != (1, 1) && simple_update_gauge!(pepo!; maxiter=100, reset=false)
    end

    # minimalcanonicalform!(tensors(pepo))

    quicknormalize!(pepo!)
    hermitianize!(pepo!)

    return pepo!, infidelity
end

function iterative_neighbourhood_update!(pepo, truncator, stepper)
    evolvedpepo = evolve(pepo, stepper)

    # simple_update_gauge!(pepo; maxiter=100)

    # truncate_bond_matrices!(pepo)

    # quicknormalize!(pepo.parent)
    # hermitianize!(pepo.parent)

    # return pepo

    # zeta = 500.0
    zeta = Inf
    # truncalg = NTU(; zeta=zeta, rankreduce=false, optalg=EVB(; gauge=NoGauge()))
    truncalg = NTU(; zeta=zeta, rankreduce=false, optalg=EVB(; maxiter=100))
    # truncalg = FET(; alg=VUMPS(bonddim=16, tol=1e-8), zeta=zeta, rankreduce=false, optalg=EVB())
    # truncalg = NTU(; zeta=zeta, rankreduce=true, optalg=DZM(EAT=false,gauge=NoGauge()))
    # truncalg = NTU(; zeta=zeta, rankreduce=true, optalg=DZM())
    # truncalg = SimpleUpdate()

    # println("xi = $(zeta)")

    # evx = evolvedtensors(pepox)[1]
    # evy = evolvedtensors(pepoy)[1]
    #

    infidelity = 0

    buff = currentbuffer(evolvedpepo)
    ybuff = permutetranspose(buff)

    # truncx = newtruncators(buff, truncalg)
    # renormalize!(truncx)
    #
    # truncy = newtruncators(ybuff, truncalg)
    # renormalize!(truncy)

    reducedbuffer = currentbuffer(pepo)
    # reducedbuffer = broadcast(baretruncation!, similar.(tensors(pepo)), buff)

    # xisoms = Isometries(stitchisometries.(tensors(pepo), buff))
    # yisoms = stitchisometries.(permutetranspose(tensors(pepo)), ybuff)

    xisoms = truncator
    ep, sp, wp, np = map(permutedims, unpack(xisoms))
    yisoms = Isometries(sp, ep, np, wp)

    weightsbuff = deepcopy(bondweights(pepo))

    tol = Inf

    maxiter = 1

    for numiter in 1:maxiter
        normalize!.(reducedbuffer)

        yreducedbuffer = permutetranspose(reducedbuffer)

        # truncx = newtruncators(reducedbuffer, truncalg)
        # renormalize!(truncx)
        # truncx = reducedbuffer
        # truncx = SimpleUpdateState()
        local isomh

        for i in 1:1
            isomh = broadcast(bonds(buff, 1)) do bond
                # reducedpair = parallel2sequential(buff, xisoms, bond)
                # truncx = ntu_total(reducedpair, reducedbuffer, bond)
                #
                # foreach(normalize!, reducedpair)

                reducedpair = buff[bond]
                truncx = reducedpair -> ntu_total_low(reducedpair, buff, bond)
                # truncx = buff

                # truncx = ntu_total_low(reducedpair, reducedbuffer, bond)

                (u, v), s, _ = parallelisometries(
                    reducedpair, truncx, bond, truncalg; D=bondspace(pepo, bond)
                )

                # D = bondspace(pepo, bond)
                # (u, v), s, _ = simpleupdate(reducedpair, weights(pepo), bond; trunc=truncspace(D))

                setindex!(xisoms, u => v, bond)
                # setindex!(yisoms, u => v, transpose(bond))

                maxnormalize!(s)

                setweight!(pepo, s, bond; copy=true)

                return (u, v)
            end
        end

        # truncy = newtruncators(yreducedbuffer, truncalg)
        # renormalize!(truncy)
        # truncy = yreducedbuffer
        # truncy = SimpleUpdateState()

        local isomv

        for i in 1:1
            isomv = broadcast(bonds(ybuff, 1)) do fictbond
                # reducedpair = parallel2sequential(ybuff, yisoms, fictbond)
                # truncy = ntu_total(reducedpair, yreducedbuffer, fictbond)
                #
                # foreach(normalize!, reducedpair)

                reducedpair = ybuff[fictbond]
                truncy = reducedpair -> ntu_total_low(reducedpair, ybuff, fictbond)
                # truncy = ybuff
                # ql0, qr0 = parallel2sequential(ybuff, yisoms, fictbond)
                #
                # ql = permutedom(ql0, (2,1,4,3))
                # qr = permutedom(qr0, (2,1,4,3))

                # truncy = ntu_total_low(reducedpair,yreducedbuffer, fictbond)

                realbond = transpose(fictbond)

                (u, v), s, _ = parallelisometries(
                    reducedpair, truncy, fictbond, truncalg; D=bondspace(pepo, realbond)
                )

                setindex!(xisoms, u => v, realbond)
                setindex!(yisoms, u => v, fictbond)

                # D = bondspace(pepo, realbond)
                # (u, v), s, _ = simpleupdate(ql => qr, weights(pepo), realbond; trunc=truncspace(D))

                setweight!(pepo, s, realbond; copy=true)

                return (u, v)
            end
        end

        isoe = broadcast(first, isomh)
        isow = broadcast(last, circshift(isomh, (1, 0)))

        isos = broadcast(first, isomv)
        ison = broadcast(last, circshift(isomv, (1, 0)))

        xisoms = Isometries(
            SquareLattice(
                broadcast(tuple, isoe, permutedims(isos), isow, permutedims(ison))
            ),
        )
        yisoms = Isometries(
            SquareLattice(
                broadcast(tuple, isos, permutedims(isoe), ison, permutedims(isow))
            ),
        )

        tol = tolerance!(weightsbuff, bondweights(pepo), 1.0)
        @warn "isometries converged to $tol after $numiter"

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
        if tol < 1e-6
            @warn "isometries converged to $tol after $numiter"
            break
        end
        #
        #
        # simple_update_gauge!(PEPO(reducedbuffer, weights(pepo)); maxiter=100)

    end

    @warn "isometries converged to $tol after"

    copy!.(getaxis(pepo), reducedbuffer)

    # foreach(
    #     tensors(pepo), buff, isoe, permutedims(isos), isow, permutedims(ison)
    # ) do dst, src, e, s, w, n
    #     truncateto!(dst, src, e, s, w, n)
    # end

    # simple_update_gauge!(pepo; maxiter=100, reset=true)
    # foreach(tup -> map(normalize!, tup), xisoms)
    # minimalcanonicalform!(tensors(pepo))

    # simple_update_gauge!(pepo); maxiter=100)
    elcopy!(truncator, xisoms)

    quicknormalize!(pepo)
    hermitianize!(pepo)

    return pepo, infidelity
end
