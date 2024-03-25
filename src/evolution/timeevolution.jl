struct SequentialUpdate end
struct ParallelUpdate end

function newevolution(pepo::AbstractPEPO, map, method)
    xpepo = newevolution(pepo, map[1])
    ypepo = newevolution(changeaxis(pepo), map[2])
    return xpepo, ypepo
end

function newevolution(pepo::AbstractPEPO, map::LocalStepper)
    evolved = broadcast(latticetensors(pepo), map.u, map.v) do p, u, v
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

function newevolution(pepo::AbstractPEPO, map::GlobalStepper)
    evolved = broadcast(latticetensors(pepo), currentstepper(map)) do p, t
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

# One full cycle of the trotter evolution. That is, t -> t + τ
function timestep!(pepo, truncator, stepper, order; kwargs...)
    # Do forward staircase on horizontal then vertical, switch inbetween, i.e does the 
    # sequence: H <-> V
    trotterstep!(pepo, truncator, stepper; reverse=false, kwargs...)

    if order == 1
        # If first-order trotter, then we just need to switch back to horizontal
        # from vertical.
        # Before: H <-> V
        changeaxis!(pepo[1], pepo[2])
        # After: H <-> V <->
    elseif order == 1
        # If second-order trotter, then we need to do a backwards staircase sweep, 
        # which starts from vertical and switches back to horizontal.
        # Before: H <-> V
        trotterstep!(pepo, truncator, stepper; reverse=true, kwargs...)
        # After: H <-> V V <-> H
    end

    return pepo
end

function trotterstep!(pepo, truncator, stepper; reverse, kwargs...)
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
    onlybond = getbonds(pepo)[1, 1]

    src_pepo = evolvedtensors(pepo)
    dst_pepo = latticetensors(pepo)

    src_site = src_pepo[1, 1]
    dst_site = dst_pepo[1, 1]

    renormalize!(truncator)

    iso = isometries(pepo, truncator, onlybond, truncalg)

    setweights!(pepo, iso.s, onlybond)

    temp = partialtruncate(src_site, iso)
    rv = partialtruncate(iso, temp)

    truncalg.gauge isa MCF && minimalcanonicalform!(rv)

    copy!(dst_site, rv)

    return pepo
end

function ssquickupdate!(pepo::GlobalUpdatablePEPO, truncator; truncalg)
    onlybond = getbonds(pepo)[1, 1]

    src_pepo = evolvedtensors(pepo)
    dst_pepo = latticetensors(pepo)

    src_site = src_pepo[1, 1]
    dst_site = dst_pepo[1, 1]

    D = bondspace(pepo, onlybond)

    w = getweights(pepo)

    src_site, _ = absorbweights!(src_site, deepcopy(src_site), w, onlybond; inverse=false)
    P1, R, L, _ = rankreduce(src_site, src_site)

    u,s,v = tsvd!(R * L; trunc=truncspace(D), alg=TensorKit.SVD())

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

    setweights!(pepo, s, onlybond)

    # copy!(dst_site, rv)

    return pepo
end

function quickupdate!(pepo::GlobalUpdatablePEPO, truncator; truncalg)
    # if truncalg.rankreduce == true
    #     throw(ErrorException("Cannot do rank reduction when updating in parallel"))
    # end

    bonds = uniquebonds(pepo)

    srcpepo = evolvedtensors(pepo)
    dstpepo = latticetensors(pepo)

    renormalize!(truncator)

    infidelity = similar(bonds, Float64)

    isoms = broadcast(enumerate(bonds)) do ebond
        i, bond = ebond
        iso, infid = isometries(pepo, truncator, bond, truncalg)

        infidelity[i] = infid

        setweights!(pepo, iso.s, bond)

        return iso
    end

    us = map(iso -> iso.u, isoms)
    vs = map(iso -> iso.v, circshift(isoms, (1, 0)))

    foreach(dstpepo, srcpepo, us, vs) do dst, src, u, v
        truncateto!(dst, src, u, v)
    end

    if issymmetric(pepo)
        # If our unitcell is symmetric, then copy the updated bond to all
        # other columns with an appropriate shift
        deepcopysymmetric!(dstpepo)
    end

    truncalg.gauge isa MCF && minimalcanonicalform!(dstpepo)

    return pepo, infidelity
end

function updatestep!(pepo::GlobalUpdatablePEPO, truncator, stepper; truncalg, reverse)
    # Evolve, all sites

    bonds = getbonds(pepo)

    if reverse
        bonds = Base.reverse(bonds)
        stepper = reverse(stepper)
    end

    pepo = evolve!(pepo, stepper)

    _, infid = quickupdate!(pepo, truncator; truncalg=truncalg)

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

    quicknormalize!(pepo)

    hermitianize!(pepo)

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

function updatestep!(pepo::LocalUpdatablePEPO, truncator, map; reverse=false, kwargs...)
    symmetric = issymmetric(pepo)
    bonds = getbonds(pepo)

    if reverse
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
                deepcopysymmetric!(latticetensors(pepo))
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

function evolve!(pepo::GlobalUpdatablePEPO, stepper::GlobalStepper)
    applystepper!(pepo, stepper)
    setisevolved!(pepo, true)
    return pepo
end

function applybothtrotter!(pepo::GlobalUpdatablePEPO, stepper; kwargs...)
    setisevolved!(pepo, false)

    pin = latticetensors(pepo)
    pev = evolvedtensors(pepo)

    applybothtrotter!(pev, pin, stepper; kwargs...)

    return pepo
end
function applybothtrotter!(pev, pin, gates::LocalStepper; kwargs...)
    return _applybothtrotter!.(pev, pin, gates.u, gates.v; kwargs...)
end
# function applybothtrotter!(pev, pin, gates::SimpleGates)
#     return _applybothtrotter!.(pev, pin, gates.east, gates.west, gates.loc)
# end
function applybothtrotter!(pev, pin, mpo::GlobalStepper; kwargs...)
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
