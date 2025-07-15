function simple_update_gauge!(pepo; maxiter=10, tol=1e-10, verbose=true, reset=false)
    reset && weights2identity!(pepo)

    prevweights = deepcopy(bondweights(pepo))

    ϵ = Inf
    numiter = 0

    while ϵ > tol && numiter < maxiter
        reverse = isodd(numiter)

        # if 1 in size(pepo)
        if false
            simple_update_gauge_parallel_step!(pepo)
        else
            simple_update_gauge_step!(pepo; inreverse=reverse)
        end

        ϵ = tolerance!(prevweights, bondweights(pepo), 1.0)

        verbose && @info ϵ

        numiter += 1
    end

    return pepo
end

function simple_update_gauge_parallel!(
    pepo; maxiter=100, tol=1e-12, verbose=true, reset=false
)
    reset && weights2identity!(pepo)

    prevweights = deepcopy(bondweights(pepo))

    ϵ = Inf
    best_ϵ = Inf
    numiter = 0

    best_tensors = broadcast(copy, pepo)
    best_weights = broadcast(copy, bondweights(pepo))

    while ϵ > tol && numiter < maxiter
        simple_update_gauge_parallel_step!(pepo)

        # ϵ = sqrt(tolerance!(prevweights, bondweights(pepo), 1.0))
        ϵ = gaugedistance(pepo)

        if ϵ < best_ϵ
            copy!.(best_tensors, pepo)
            copy!.(best_weights, bondweights(pepo))
            best_ϵ = ϵ
        end

        verbose && @info ϵ

        numiter += 1
    end

    copy!.(pepo, best_tensors)
    copy!.(bondweights(pepo), best_weights)

    @info "Distance to Vidal gauge: $(gaugedistance(pepo))"

    return pepo
end
function simple_update_gauge_isometries!(pepo; maxiter=100, tol=1e-12, verbose=true)
    # weights2identity!(pepo)

    prevweights = deepcopy(bondweights(pepo))

    ϵ = Inf
    numiter = 0
    while ϵ > tol && numiter < maxiter
        reverse = isodd(numiter)

        if 1 in size(pepo)
            simple_update_gauge_parallel_step!(pepo)
        else
            simple_update_gauge_step!(pepo, buffer; inreverse=reverse)
        end

        ϵ = gaugedistance(pepo)

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

function simple_update_gauge_step!(pepo; inreverse=false, absorb=true)
    bonds = eachbond(pepo)

    if inreverse
        itr = Iterators.reverse(bonds)
    else
        itr = bonds
    end

    itr = bonds

    for bond in itr
        pair = simple_update_gauge_get_pair(pepo, pepo, bond)

        newpair, s = vidalupdate(pair, bondweights(pepo), bond; trunc=notrunc())

        pepo[bond] = newpair

        setweight!(pepo, s, bond; copy=false)
    end

    @warn gaugedistance(pepo)

    return pepo
end

function simple_update_gauge_parallel_isometries!(pepo)
    newweights = deepcopy(bondweights(pepo))

    isomh, isomv = map((1:2)) do axis
        thebonds = bonds(pepo, axis)

        isom = map(thebonds) do bond

            # l, s, r, infid = simpleupdate!(axispepo, bond; trunc=notrunc())
            iso, s, infid = simpleupdate(pepo[bond], bondweights(pepo), bond; trunc=notrunc())

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

    isoms = Isometries(UnitCell{Square}(broadcast(tuple, ge, gs, gw, gn)))

    for I in eachindex(pepo)
        dst = pepo[I]
        e, s, w, n = isoms[I]

        truncateto!(dst, copy(dst), e, s, w, n)

        # mergeinto!(dst, spten)
    end

    tolerance!(bondweights(pepo), newweights, 1.0)

    quicknormalize!(pepo)
    hermitianize!(pepo)

    return pepo
end
function truncate_bond_matrices!(pepo)
    isomh, isomv = map(bonds(pepo)) do axisbonds
        isoms = broadcast(axisbonds) do bond
            D = bondspace(pepo, bond)

            w = bondweights(pepo)[bond]

            sqrtw = sqrt(w)

            su, ss, sv = tsvd!(w; trunc=truncspace(D))

            u = pinv(sqrtw) * su * sqrt(ss)
            v = sqrt(ss) * sv * pinv(sqrtw)

            setweight!(pepo, ss, bond; copy=false)

            return (u, v)
        end

        return isoms
    end

    isometries = UnitCellBonds(isomh, isomv)

    srcpepo = currentbuffer(pepo)
    dstpepo = axisbuffer(pepo)

    foreach(dstpepo, srcpepo, isometries) do dst, src, iso
        (e, s, w, n) = iso
        truncateto!(dst, src, e[1], s[1], w[2], n[2])
    end

    pepo.state = 0

    return pepo
end

function truncate_bond_matrices!(dstpepo, srcpepo, isoms)
    for bond in eachbond(dstpepo)
        D = bondspace(dstpepo, bond)

        w = bondweights(srcpepo)[bond]

        sqrtw = sqrt(w)

        su, ss, sv = tsvd!(w; trunc=truncspace(D))

        u = pinv(sqrtw) * su * sqrt(ss)
        v = sqrt(ss) * sv * pinv(sqrtw)

        setindex!(isoms, u => v, bond)

        setweight!(dstpepo, ss, bond; copy=false)
    end

    for I in eachindex(dstpepo)
        dst = (dstpepo)[I]
        src = (srcpepo)[I]
        e, s, w, n = isoms[I]
        truncateto!(dst, src, e, s, w, n)
    end

    return dstpepo
end

function _current_buffer(
    pepo::PEPO{G,PType,CType}, buffer, state::Int
) where {G,PType,CType}
    T = UnitCell{G,PType,CType}

    rv = iszero(state) ? axisbuffer(pepo) : buffer[state]

    return rv
end

function truncate!(::Axial, ::Axial, pepo, isometries, weights)
    state = pepo.state

    @assert pepo in (1, 3)

    buffers = pepo.buffer

    srcbuff = buffers[state]

    if state == 1
        dstbuff = axisbuffer(pepo)
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
    dst = axisbuffer(pepo)

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
    pepo, isometries, weights::UnitCell, buffer1, buffer2, state
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
        dst = axisbuffer(pepo)

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

function evolve(pepo::PEPO{G}, stepper::FullStepper) where {G}
    state = pepo

    evolved = broadcast(pepo, stepper) do p, t
        D1, D2, D3, D4 = domain(p)
        η1, η2, η3, η4 = domain(t)

        T = promote_type(eltype(p), eltype(t))

        tensor = TensorMap{T}(
            undef,
            codomain(p),
            fuse(D1 * η1) * fuse(D2 * η2) * fuse(D3 * η3)' * fuse(D4 * η4)',
        )

        return tensor
    end

    # src = axisbuffer(pepo)
    apply_four_bond_stepper!(evolved, pepo, stepper)

    return PEPO{G}(evolved)
end

function evolve(pepo::AbstractPEPO, stepper::FullStepper)
    state = pepo.state

    ten = pepo
    ste = stepper

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
    ops = axisbuffer(stepper)[bond]

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
    opspair = stepper[bond]

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

function stitchisometries(template::AbstractUnitCell, src::AbstractUnitCell)
    return map(stitchisometries, template, src)
end
function stitchisometries(template::PermuteTranspose, src::AbstractUnitCell)
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
