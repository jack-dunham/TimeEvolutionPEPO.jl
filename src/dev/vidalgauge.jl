function simple_update_gauge!(
    pepo; maxiter=100, tol=1e-12, verbose=true, buffering=false, reset=false
)

    reset && weights2identity!(pepo)

    prevweights = deepcopy(weights(pepo))

    ϵ = Inf
    numiter = 0

    if buffering
        buffer = similar.(currentbuffer(pepo))
    else
        buffer = nothing
    end

    while ϵ > tol && numiter < maxiter
        reverse = isodd(numiter)

        if 1 in size(pepo)
            simple_update_gauge_parallel_step!(pepo)
        else
            simple_update_gauge_step!(pepo, buffer; inreverse=reverse)
        end

        ϵ = tolerance!(prevweights, weights(pepo), 1.0)

        verbose && @info ϵ

        numiter += 1
    end

    return pepo
end

function simple_update_gauge_parallel!(
    pepo; maxiter=100, tol=1e-12, verbose=true, reset=false
)
    reset && weights2identity!(pepo)

    prevweights = deepcopy(weights(pepo))

    ϵ = Inf
    numiter = 0

    while ϵ > tol && numiter < maxiter
        reverse = isodd(numiter)

        simple_update_gauge_parallel_step!(pepo)

        ϵ = tolerance!(prevweights, weights(pepo), 1.0)

        verbose && @info ϵ

        numiter += 1
    end

    return pepo
end
function simple_update_gauge_isometries!(
    pepo; maxiter=100, tol=1e-12, verbose=true, buffering=false
)
    # weights2identity!(pepo)

    prevweights = deepcopy(weights(pepo))

    ϵ = Inf
    numiter = 0

    if buffering
        buffer = similar.(currentbuffer(pepo))
    else
        buffer = nothing
    end

    while ϵ > tol && numiter < maxiter
        reverse = isodd(numiter)

        if 1 in size(pepo)
            simple_update_gauge_parallel_step!(pepo)
        else
            simple_update_gauge_step!(pepo, buffer; inreverse=reverse)
        end

        ϵ = tolerance!(prevweights, weights(pepo), 1.0)

        verbose && @info ϵ

        numiter += 1
    end

    xisoms, yisoms = simple_update_gauge_parallel_isometries!(pepo)

    return xisoms, yisoms, (ϵ, numiter)
end

function simple_update_gauge_get_pair(pepo, ::Nothing, bond)
    a, b = pepo[bond]
    tensorpair = copy(a) => copy(b)

    return tensorpair
end
simple_update_gauge_get_pair(_, buffer, bond) = buffer[bond]

function simple_update_gauge_step!(pepo, buffer; inreverse=false, absorb=true)

    # Iterative calculation 
    isnothing(buffer) || copy!.(buffer, currentbuffer(pepo))

    bonds = eachbond(pepo)

    if inreverse
        itr = Iterators.reverse(bonds)
    else
        itr = bonds
    end

    itr = bonds

    for bond in itr
        pair = simple_update_gauge_get_pair(pepo, buffer, bond)

        iso, s, infid = simpleupdate!(
            pair, weights(pepo), bond; absorb=absorb, trunc=notrunc()
        )

        # if abs(infid) > 1e-14
        #     @warn infid
        # end

        newpair = mapreduce(copy, =>, pepo[bond])

        truncateto!(newpair, pair, iso; axis=bondaxis(bond))

        pepo.state[bond] = newpair

        setweight!(pepo, s, bond; copy=false)

        isnothing(buffer) || copy!(last(buffer[bond]), last(pepo[bond]))
    end

    return pepo
end

function simple_update_gauge_parallel_isometries!(pepo)
    newweights = deepcopy(weights(pepo))

    isomh, isomv = map((1:2)) do axis
        thebonds = bonds(pepo, axis)

        isom = map(thebonds) do bond

            # l, s, r, infid = simpleupdate!(axispepo, bond; trunc=notrunc())
            iso, s, infid = simpleupdate(pepo[bond], weights(pepo), bond; trunc=notrunc())

            # if abs(infid) > 1e-14
            #     @warn infid
            # end

            setweight!(newweights, s, bond; copy=false)

            return iso
        end

        return isom
    end

    # all tohether

    isoe = broadcast(first, isomh)
    isow = broadcast(last, circshift(isomh, (1, 0)))

    isos = broadcast(first, isomv)
    ison = broadcast(last, circshift(isomv, (1, 0)))

    return isomh, isomv, newweights
end

function simple_update_gauge_parallel_step!(pepo)

    isomh, isomv, newweights = simple_update_gauge_parallel_isometries!(pepo)

    ge = broadcast(first, isomh)
    gw = broadcast(last, circshift(isomh, (1, 0)))
    #
    gs = broadcast(first, permutedims(isomv))
    gn = broadcast(last, permutedims(circshift(isomv, (1, 0))))

    isoms = Isometries(SquareLattice(broadcast(tuple, ge, gs, gw, gn)))

    buff = currentbuffer(pepo)

    for I in eachindex(buff)
        dst = buff[I]
        e, s, w, n = isoms[I]

        truncateto!(dst, copy(dst), e, s, w, n)

        # mergeinto!(dst, spten)
    end

    tolerance!(weights(pepo), newweights, 1.0)

    quicknormalize!(pepo)
    hermitianize!(pepo)

    return pepo
end
function truncate_bond_matrices!(pepo)
    isomh, isomv = map(bonds(pepo)) do axisbonds
        isoms = broadcast(axisbonds) do bond
            D = bondspace(pepo, bond)

            w = weights(pepo)[bond]

            sqrtw = sqrt(w)

            su, ss, sv = tsvd!(w; trunc=truncspace(D))

            u = pinv(sqrtw) * su * sqrt(ss)
            v = sqrt(ss) * sv * pinv(sqrtw)

            setweight!(pepo, ss, bond; copy=false)

            return (u, v)
        end

        return isoms
    end

    isometries = EdgeLattice(isomh, isomv)

    srcpepo = currentbuffer(pepo)
    dstpepo = tensors(pepo)

    foreach(dstpepo, srcpepo, isometries) do dst, src, iso
        (e, s, w, n) = iso
        truncateto!(dst, src, e[1], s[1], w[2], n[2])
    end

    pepo.state = 0

    return pepo
end

function truncate_bond_matrices!(dstpepo, srcpepo, isoms)
    for bond in eachbond(tensors(dstpepo))
        D = bondspace(dstpepo, bond)

        w = weights(srcpepo)[bond]

        sqrtw = sqrt(w)

        su, ss, sv = tsvd!(w; trunc=truncspace(D))

        u = pinv(sqrtw) * su * sqrt(ss)
        v = sqrt(ss) * sv * pinv(sqrtw)

        setindex!(isoms, u => v, bond)

        setweight!(dstpepo, ss, bond; copy=false)
    end

    for I in eachindex(tensors(dstpepo))
        dst = tensors(dstpepo)[I]
        src = tensors(srcpepo)[I]
        e, s, w, n = isoms[I]
        truncateto!(dst, src, e, s, w, n)
    end

    return dstpepo
end

function _current_buffer(
    pepo::PEPO{G,PType,CType}, buffer, state::Int
) where {G,PType,CType}
    T = Lattice{G,PType,CType}

    rv = iszero(state) ? tensors(pepo) : buffer[state]

    return rv
end

function truncate!(::Axial, ::Axial, pepo, isometries, weights)
    state = pepo.state

    @assert pepo in (1, 3)

    buffers = pepo.buffer

    srcbuff = buffers[state]

    if state == 1
        dstbuff = tensors(pepo)
    elseif state == 3
        dstbuff = buffers[2]
    end

    foreach(dstbuff, srcbuff, isometries) do dst, src, iso
        u, v, = iso
        truncate!(dstbuff, srcbuff, u, v)
    end

    setweight!(pepo, weights; copy=false)

    pepo.state = state - 1

    return pepo
end

function truncate!(::Global, ::Global, pepo, isometries, weights)
    @assert pepo.state == 1

    src = only(pepo.buffer)
    dst = tensors(pepo)

    foreach(dstbuff, srcbuff, isometries) do dst, src, iso
        e, s, w, n = iso
        truncate!(dstbuff, srcbuff, e, s, w, n)
    end

    setweight!(pepo, weights; copy=false)

    pepo.state = 0

    return pepo
end

function truncate!(::Global, ::Axial, pepo, isometries, weights)
    buffer1, buffer2 = pepo.buffer.data

    state = pepo.state

    _truncate_semi_parallel!(pepo, isometries, weights, buffer1, buffer2, state)

    return pepo
end

function _truncate_semi_parallel!(
    pepo, isometries, weights::Lattice, buffer1, buffer2, state
)
    uiso, viso = unpack(isometries)

    if state == 1
        # Untruncated

        src = buffer1
        dst = buffer2

        axis = 1

        state = 2
    elseif state == 2
        # Truncated along the x-axis

        src = buffer2
        dst = tensors(pepo)

        axis = 2

        pepo.state = 0
    else
        throw(ErrorException("got state $(pepo.state)"))
    end

    setweight!(pepo, weights; axis=axis)

    foreach(permtruncate!, dst, transpose(src), uiso, viso)

    return pepo
end

function setweight!(pepo, svals; axis=axis, copy=false)
    if axis == 1
        src = svals
    elseif axis == 2
        src = transpose(svals)
    else
        throw(ErrorException(""))
    end

    pepoweights = getweight(pepo)

    if copy
        copy!.(pepoweights, src)
    else
        pepoweights .= src
    end

    return pepo
end

function evolve(pepo::PEPO, stepper::FullStepper)
    state = pepo.state

    evolved = broadcast(tensors(pepo), tensors(stepper)) do p, t
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

    src = tensors(pepo)
    apply_four_bond_stepper!(evolved, src, stepper)

    return PEPO(evolved)
end

function evolve(pepo::AbstractPEPO, stepper::FullStepper)
    state = pepo.state

    ten = tensors(pepo)
    ste = tensors(stepper)

    rv = broadcast(ten, ste) do site, op
        d1, d2, d3, d4 = domain(site)
        D1, D2, D3, D4 = domain(op)

        c = codomain(site)

        dst = similar(
            site, c, fuse(d1 * D1) * fuse(d2 * D2) * fuse(d3 * D3)' * fuse(d4 * D4)'
        )

        apply_four_bond_stepper!(dst, site, op)

        return dst
    end

    return rv
end

# function evolve(pepo::AxialPEPO, stepper::AxialStepper)
#     ten = tensors(pepo)
#     ste = tensors(stepper)
#
#     rv = broadcast(ten, ste) do site, op
#         d1, d2, d3, d4 = domain(site)
#         D1, D3 = domain(op)
#
#         dst = similar(site, codomain(site), fuse(d1 * D1) * d2 * fuse(d3 * D3)' * d4)
#
#         apply_four_bond_stepper!(dst, site, op)
#
#         return dst
#     end
#
#     return rv
# end
function localevolve(pair::Pair, stepper::FullStepper, bond)
    ops = tensors(stepper)[bond]

    rvpair = apply_four_bond_stepper_locally(pair, ops)

    return rvpair
end

function _splitu(op, η)
    D = domain(op)
    rv = similar(op, D * η, D)

    mergeinto!(rv, op)

    return rv
end

function _splitv(op, η)
    D = codomain(op)
    rv = similar(op, D, D * η')

    mergeinto!(rv, op)

    return rv
end

function evolve(pair::Pair, stepper::FullStepper, isoms, bond)
    opspair = tensors(stepper)[bond]

    isomspair = isoms[bond]

    A, B = pair
    opA, opB = opspair

    domA = domain(opA)
    domB = domain(opB)

    DA1, DA2, DA3, DA4 = domain(A)
    DB1, DB2, DB3, DB4 = domain(B)

    codA = codomain(A)
    codB = codomain(B)

    isomsA, isomsB = map(isomspair, opspair) do isoms, op
        η1, η2, η3, η4 = domain(op)

        (e, s, w, n) = isoms

        ee = _splitu(e, η1)
        ss = _splitu(s, η2)
        ww = _splitv(w, η3)
        nn = _splitv(n, η4)

        return (ee, ss, ww, nn)
    end

    if bondaxis(bond) == 1
        rvA = similar(A, codA, fuse(DA1 * domA[1]) * DA2 * DA3 * DA4)
        rvB = similar(B, codB, DB1 * DB2 * fuse(DB3 * domB[3])' * DB4)

        apply_pepso_left!(rvA, A, opA, isomsA)
        apply_pepso_right!(rvB, B, opB, isomsB)

    elseif bondaxis(bond) == 2
        rvA = similar(A, codA, DA1 * fuse(DA2 * domA[2]) * DA3 * DA4)
        rvB = similar(B, codB, DB1 * DB2 * DB3 * fuse(DB4 * domB[4])')

        apply_pepso_up!(rvA, A, opA, isomsA)
        apply_pepso_down!(rvB, B, opB, isomsB)
    end

    return rvA => rvB
end

function pepo_contract_test(out, p, op, cs, cw, cn)
    @tensoropt out[e2 ee2; ee1 e1] =
        p[k1 b1; e1 s1 w1 n1] *
        op[kx bx k1 b1; ee1 ss1 ww1 nn1] *
        (op')[ee2 ss2 ww2 nn2; kx bx k2 b2] *
        (p')[e2 s2 w2 n2; k2 b2] *
        cs[s1 ss1; ss2 s2] *
        cw[w1 ww1; ww2 w2] *
        cn[n1 nn1; nn2 n2]
    return out
end

# Be default, we don't need any additional buffers
constructbuffers(pepo, steps, truncstyle) = latticetype(pepo)[]

# Be default, we don't need any additional buffers
function constructbuffers(pepo::PEPO, steps::Vector{<:TwoBondStepper}, _)
    xind = findfirst(s -> bondaxis(s) == 1, steps)
    yind = findfirst(s -> bondaxis(s) == 2, steps)

    buffer0 = tensors(pepo)

    buffer1 = construct_axis_buffer(buffer0, steps[xind])
    buffer2 = permutetranpose(buffer0)
    buffer3 = construct_axis_buffer(buffer2, steps[yind])

    return [buffer1, buffer2, buffer3]
end

function construct_axis_buffer(lattice::Lattice, map::TwoBondStepper)
    evolved = broadcast(lattice, tensors(map)) do p, t
        D1, D2, D3, D4 = domain(p)

        east = domain(t)[1]
        west = domain(t)[2]

        T = promote_type(eltype(p), eltype(t))

        tensor = TensorMap(
            undef, T, codomain(p), fuse(D1 * east) * D2 * fuse(D3 * west)' * D4
        )

        return tensor
    end

    return evolved
end

function construct_axis_buffer(lattice::Lattice, map::FullStepper)
    evolved = broadcast(lattice, tensors(map)) do p, t
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

    return evolved
end

function constructbuffers(pepo, steps::Vector{<:FullStepper{<:Any,<:TensorMap}}, truncstyle)
    buffers = latticetype(pepo)[]

    buffer0 = tensors(pepo)

    buffer1 = construct_axis_buffer(buffer0, steps[begin])
    push!(buffers, buffer1)

    if truncstyle isa Axial
        buffer2 = broadcast(permutedims(buffer0), permutedims(buffer1)) do dst, src
            d1 = domain(dst)[1]
            d2 = domain(src)[2]
            d3 = domain(dst)[3]
            d4 = domain(src)[4]
            return similar(src, codomain(src), d2 * d1 * d4 * d3)
        end

        push!(buffers, buffer2)
    end

    return buffers
end

function stitchisometries(A, dst::ProductSpace{S,4}, src::ProductSpace{S,4}) where {S}
    D1, D2, D3, D4 = src
    d1, d2, d3, d4 = dst

    e = isometry(A, d1 * D1, d1)
    s = isometry(A, d2 * D2, d2)
    w = _transpose(isometry(A, d3 * D3, d3))
    n = _transpose(isometry(A, d4 * D4, d4))

    return (e, s, w, n)
end

function stepstitchisometries(
    A::Type, dst::ProductSpace{S,4}, src::ProductSpace{S,4}
) where {S}
    d1, d2, d3, d4 = dst
    D1, D2, D3, D4 = src

    e = isometry(A, fuse(d1 * D1), d1)
    s = isometry(A, fuse(d2 * D2), d2)
    w = _transpose(isometry(A, fuse(d3 * D3)', d3))
    n = _transpose(isometry(A, fuse(d4 * D4)', d4))

    return (e, s, w, n)
end
function stepstitchisometries(
    A::Type, dst::ProductSpace{S,4}, src::ProductSpace{S,2}
) where {S}
    d1, d2, d3, d4 = dst
    D1, D3 = src

    e = isometry(A, fuse(d1 * D1), d1)
    s = isometry(A, d2, d2)
    w = _transpose(isometry(A, fuse(d3 * D3)', d3))
    n = _transpose(isometry(A, d4, d4))

    return (e, s, w, n)
end
function bpisometries(A::Type, dst::ProductSpace{S,4}, src::ProductSpace{S,4}) where {S}
    d1, d2, d3, d4 = dst
    D1, D2, D3, D4 = src

    e = isometry(A, d1 * D1, d1 * D1)
    s = isometry(A, d2 * D2, d2 * D2)
    w = _transpose(isometry(A, d3' * D3', d3' * D3'))
    n = _transpose(isometry(A, d4' * D4', d4' * D4'))

    return (e, s, w, n)
end
function stitchisometries(dst::TenAbs{4}, src::TenAbs{4})
    return stitchisometries(storagetype(dst), domain(dst), domain(src))
end

function stitchisometries(template::AbstractLattice, src::AbstractLattice)
    return map(stitchisometries, template, src)
end
function stitchisometries(template::PermuteTranspose, src::AbstractLattice)
    isoms = map(PermutedDimsArray(template.parent, (2, 1)), src) do out, in
        dstdom = permute(domain(out), template.perm)
        srcdom = domain(src)

        return sitchisometries(dstdom, srcdom)
    end
    return isoms
end

function baretruncation!(dst, src)
    (e, s, w, n) = stitchisometries(dst, src)

    truncateto!(dst, src, e, s, w, n)

    return dst
end

function parallel2sequential(lattice, isometries, bond)
    uind, vind = bond

    uold, vold = lattice[bond]

    _, s1, w1, n1 = isometries[uind]
    e2, s2, _, n2 = isometries[vind]

    @tensoropt begin
        unew[k b; e s w n] := uold[k b; e ss ww nn] * s1[ss; s] * w1[w; ww] * n1[n; nn]
        vnew[k b; e s w n] := vold[k b; ee ss w nn] * e2[ee; e] * s2[ss; s] * n2[n; nn]
    end

    return unew => vnew
end

function parallel2sequential(lattice, stepper, isometries, bond)
    uind, vind = bond

    uold, vold = lattice[bond]

    _, s1, w1, n1 = isometries[uind]
    e2, s2, _, n2 = isometries[vind]

    @tensoropt begin
        unew[k b; e s w n] := uold[k b; e ss ww nn] * s1[ss; s] * w1[w; ww] * n1[n; nn]
        vnew[k b; e s w n] := vold[k b; ee ss w nn] * e2[ee; e] * s2[ss; s] * n2[n; nn]
    end

    return unew => vnew
end

function parallel2sequential(pair::Pair, isometries, bond)
    uind, vind = bond

    _, s1, w1, n1 = isometries[uind]
    e2, s2, _, n2 = isometries[vind]

    @tensoropt begin
        unew[k b; e s w n] := uold[k b; e ss ww nn] * s1[ss; s] * w1[w; ww] * n1[n; nn]
        vnew[k b; e s w n] := vold[k b; ee ss w nn] * e2[ee; e] * s2[ss; s] * n2[n; nn]
    end

    return unew => vnew
end
