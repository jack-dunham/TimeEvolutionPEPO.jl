@kwdef mutable struct IterativeSU <: AbstractTruncationScheme
    maxiter::Int = 20
    tol::Float64 = 1e-6
    sequential::Bool = true
    recycle::Bool = true
end

struct Isometries{G,N,U<:AbsTen{N},V<:TenAbs{N}}
    e::Lattice{G,U,Matrix{U}}
    s::Lattice{G,U,Matrix{U}}
    w::Lattice{G,V,Matrix{V}}
    n::Lattice{G,V,Matrix{V}}
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
    isomsinit = broadcast(buffer, tensors(stepper)) do dst, src
        return stepstitchisometries(storagetype(dst), domain(dst), domain(src))
    end
    return Isometries(isomsinit)
end

function newsplitisometries(buffer, stepper)
    isomsinit = broadcast(buffer, tensors(stepper)) do dst, src
        return stitchisometries(storagetype(dst), domain(dst), domain(src))
    end
    return Isometries(isomsinit)
end

function new_bp_isometries(buffer, stepper)
    isomsinit = broadcast(buffer, tensors(stepper)) do dst, src
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
    datal = Lattice(data)

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

function timestep!(
    pepo!::AbstractPEPO, truncator!::Isometries, stepper, truncalg::IterativeSU
)
    iterative_simple_update!(
        pepo!,
        truncator!,
        stepper;
        maxiter=truncalg.maxiter,
        tol=truncalg.tol,
        recycle=truncalg.recycle,
        sequential=truncalg.sequential,
    )

    return pepo!
end

# function timestep!(
#     pepo!::AbstractPEPO, truncator!::Isometries, sequence::Vector, truncalg,
# )
#     iterative_neighbourhood_update!(
#         pepo!,
#         truncator!,
#         only(sequence);
#     )
#
#     return pepo!
# end

function newtruncators(pepo, sequence, ::IterativeSU)
    iso = newisometries(tensors(pepo), first(sequence))
    return (iso, iso)
end

function iterative_simple_update_step(pepo, stepper, isometries, bond)
    pair = currentbuffer(pepo)[bond]

    reducedpair = localevolve(pair, stepper, isometries, bond)

    foreach(normalize!, reducedpair)

    D = bondspace(pepo, bond)

    (u, v), s, infid = simpleupdate(reducedpair, weights(pepo), bond; trunc=truncspace(D))

    return (u, v), s, infid
end

function iterative_simple_update!(
    pepo!, truncator!, stepper; maxiter, tol, sequential, recycle
)
    mode = :none
    if mode === :quick
        rv = evolve(pepo!, stepper)
        newpepo = PEPO(rv)

        simple_update_gauge!(newpepo)

        truncate_bond_matrices!(pepo!, newpepo, truncator!)

        quicknormalize!(pepo!)
        hermitianize!(pepo!)

        return pepo!
    end

    if mode === :bp
        buff = currentbuffer(pepo!)
        messages = sqrtbeliefpropagation(buff, stepper)
        isoms = newsplitisometries(buff, stepper)

        for bond in bonditer(buff)
            D = bondspace(buff, bond)
            (u, v), s = vidalisometries(messages, bond; trunc=truncspace(D))
            setindex!(isoms, u => v, bond)
        end
        for I in eachindex(buff)
            dst = buff[I]
            op = tensors(stepper)[I]
            e, s, w, n = isoms[I]

            truncateto!(dst, dst, op, e, s, w, n)

            # mergeinto!(dst, spten)
        end

        quicknormalize!(pepo!)
        hermitianize!(pepo!)

        return pepo!
    end

    infidelity = 0

    buff = currentbuffer(pepo!)

    isoms = recycle ? truncator! : newisometries(buff, stepper)

    # Buffers 
    newisoms = similar(isoms)
    newweights = deepcopy(weights(pepo!))

    for i in 1:maxiter
        bonds = bonditer(buff)

        if sequential
            itr = iseven(i) ? Iterators.reverse(bonds) : bonds

            for bond in itr
                (u, v), s, _ = iterative_simple_update_step(pepo!, stepper, isoms, bond)

                setindex!(isoms, u => v, bond)
                setweight!(pepo!, s, bond; copy=true)
            end

            new = newweights
            old = weights(pepo!)

        else
            Threads.@threads for bond in collect(bonds)
                (u, v), s, _ = iterative_simple_update_step(pepo!, stepper, isoms, bond)

                setindex!(newisoms, u => v, bond)
                newweights[bond] = s
            end

            new = weights(pepo!)
            old = newweights

            elcopy!(isoms, newisoms)
        end

        conv = tolerance!(new, old, 1.0)

        @warn "Got $conv after $i"

        if conv < tol && i > 0
            break
        end

        #=

        temppepo = deepcopy(pepo!)

        for I in eachindex(buff)
            dst = tensors(temppepo)[I]
            op = tensors(stepper)[I]
            e, s, w, n = isoms[I]

            truncateto!(dst, dst, TensorMap(op), e, s, w, n)

            # mergeinto!(dst, spten)
        end

        gauge1, gauge2 = simple_update_gauge_isometries!(temppepo)

        ge = broadcast(first, gauge1)
        gw = broadcast(last, circshift(gauge1, (1, 0)))
        #
        gs = broadcast(first, permutedims(gauge2))
        gn = broadcast(last, permutedims(circshift(gauge2, (1, 0))))

        gisoms = SquareLattice(
            broadcast(tuple, ge, gs, gw, gn)
        )

        isoms = broadcast(eachindex(buff)) do I
            e1, s1, w1, n1 = isoms[I]
            e2, s2, w2, n2 = gisoms[I]

            e = e1 * e2
            s = s1 * s2

            w = w2 * w1
            n = n2 * n1

            return (e,s,w,n)
        end

        isoms = Isometries(Lattice(isoms))

        foreach(bonditer(buff)) do bond
            s = weights(temppepo)[bond]
            setweight!(pepo!, s, bond; copy=true)
        end

        =#
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

        truncateto!(dst, dst, op, e, s, w, n)

        # mergeinto!(dst, spten)
    end

    size(buff) == (1, 1) && minimalcanonicalform!(buff)
    #
    size(buff) != (1, 1) && simple_update_gauge!(pepo!; maxiter=500, reset=true)


    # minimalcanonicalform!(tensors(pepo))

    quicknormalize!(pepo!)
    hermitianize!(pepo!)

    return pepo!, infidelity
end

function iterative_neighbourhood_update!(pepo, truncator, stepper)
    evolve!(pepo, stepper)

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
    # truncalg = NTU(; ξ=ξ, rankreduce=true, optalg=DZM(EAT=false,gauge=NoGauge()))
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

    # xisoms = Isometries(stitchisometries.(tensors(pepo), buff))
    # yisoms = stitchisometries.(permutetranspose(tensors(pepo)), ybuff)

    xisoms = truncator
    ep, sp, wp, np = map(permutedims, unpack(xisoms))
    yisoms = Isometries(sp, ep, np, wp)

    weightsbuff = deepcopy(weights(pepo))

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
                # truncx = ntu_total_low(reducedpair, reducedbuffer, bond)
                #
                # foreach(normalize!, reducedpair)

                reducedpair = buff[bond]
                truncx = ntu_total_low(reducedpair, buff, bond)

                # truncx = ntu_total_low(reducedpair, reducedbuffer, bond)

                (u, v), s, _ = parallelisometries(
                    reducedpair, truncx, bond, truncalg; D=bondspace(pepo, bond)
                )

                # D = bondspace(pepo, bond)
                # (u, v), s, _ = simpleupdate(reducedpair, weights(pepo), bond; trunc=truncspace(D))

                setindex!(xisoms, u => v, bond)
                setindex!(yisoms, u => v, transpose(bond))

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
                # truncy = ntu_total_low(reducedpair, yreducedbuffer, fictbond)
                #
                # foreach(normalize!, reducedpair)

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

        tol = tolerance!(weightsbuff, weights(pepo), 1.0)
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

    copy!.(tensors(pepo), reducedbuffer)

    # foreach(
    #     tensors(pepo), buff, isoe, permutedims(isos), isow, permutedims(ison)
    # ) do dst, src, e, s, w, n
    #     truncateto!(dst, src, e, s, w, n)
    # end

    pepo.state = 0

    # simple_update_gauge!(pepo; maxiter=100)
    # foreach(tup -> map(normalize!, tup), xisoms)
    # minimalcanonicalform!(tensors(pepo))

    # simple_update_gauge!(pepo); maxiter=100)
    elcopy!(truncator, xisoms)

    quicknormalize!(pepo.parent)
    hermitianize!(pepo.parent)

    return pepo, infidelity
end
