# function timestep!(pepo, truncator::SimpleUpdateState, stepper, ::Sequential, ::Sequential; kwargs...)
#     sequential_update_step!(pepo, truncator, stepper; kwargs...)
# end

function axial_trotter_step!(pepo, truncator, stepper; truncalg, kwargs...)

    axis = bondaxis(stepper)

    axistrunc = truncator[axis]

    axisbuffer = currentbuffer(pepo)

    if !ispostenv(pepo)
        renormalize!(truncator, currentbuffer(pepo))
    end

    sequential_update_step!(pepo, axistrunc, stepper; truncalg=truncalg, kwargs...)


    # This does nothing if `stepstyle isa Sequential` as truncation has already occured 
    # for all bonds
    truncate!(pepo)

    return pepo
end

function sequential_update_step!(pepo, truncator, stepper, weights; kwargs...)

    # Evolve and then truncate each bond in sequence

    buffer = currentbuffer(pepo)

    infidelity = similar(buffer)

    bonds = bonds(buffer, 1)

    isreversed(stepper) && (bonds = Iterators.Reverse(bonds))

    foreach(bonds) do bond
        D = bondspace(buffer, bond)

        # Evolved the pair of tensors connected by `bond`, leaving the original pair
        # unmutated.
        pair = evolve(buffer[bond], stepper)

        iso, s, _ = isometries!(pair, pepo, truncator, bond, truncalg)

        truncate!(pepo[bond], pair, iso)

        setweight!(pepo, s, bond; copy=true)
    end

    quicknormalize!(pepo)
    hermitianize!(pepo)

    return pepo, infidelity
end

# isometries for sequential simple update
function isometries!(
    pair::Pair, pepo::AbstractPEPO, ::SimpleUpdateState, bond::Bond, truncalg::SU
)
    D = bondspace(pepo, bond)

    trunc = truncspace(D)

    xi = truncalg.ξ

    iso, s, infid = simpleupdate!(pair, pepo, bond; trunc=trunc, xi=xi)

    return iso, s, infid
end

# wrapper for updating in parallel
function parallelisometries(
    pepo::AbstractPEPO, truncator::SimpleUpdateState, bond::Bond, truncalg::SU
)
    a, b = pepo[bond]

    pair = copy(a) => copy(b)

    return isometries!(pair, pepo, truncator, bond, truncalg)
end

function parallelisometries!(pair::Pair, pepo::AbstractPEPO, truncator::SimpleUpdateState, bond::Bond, truncalg::SU
)
    return isometries!(pair, pepo, truncator, bond, truncalg)
end
# Sequential simple update
function simpleupdate!(pair, weights, bond; trunc, absorb=true, kwargs...)
    axis = bondaxis(bond)

    absorb && absorb_adjacent_weights!(pair, weights, bond; inverse=false)

    iso, s, infid = simple_update_svd!(pair; trunc=trunc, absorb=true, axis=axis, kwargs...)

    absorb && absorb_adjacent_weights!(pair, weights, bond; inverse=true)

    return iso, s, infid
end

function simpleupdate(oldpair, weights, bond; trunc, absorb=true, kwargs...)
    axis = bondaxis(bond)

    pair = mapreduce(copy, =>, oldpair)

    absorb && absorb_adjacent_weights!(pair, weights, bond; inverse=false)

    iso, s, infid = simple_update_svd(pair; trunc=trunc, absorb=true, axis=axis, kwargs...)

    return iso, s, infid
end

function simple_update_svd(pair; axis, kwargs...)
    (P, Q), (u0, v0) = rankreduce(pair; axis=axis)

    (u, v), s, infid = isometries_from_svd(P, Q, u0, v0; kwargs...)

    return pinv(u0) * u => v * pinv(v0), s, infid
end

function simple_update_svd!(pair; axis, kwargs...)
    (P, Q), (u0, v0) = rankreduce!(pair; axis=axis)

    iso, s, infid = isometries_from_svd(P, Q, u0, v0; kwargs...)

    return iso, s, infid
end

function isometries_from_svd(P, Q, u0, v0; xi=Inf, trunc=notrunc(), absorb=true)
    u, s, v, _ = safetsvd!(u0 * v0; trunc=trunc)

    maxnormalize!(s)

    if absorb
        u = u * sqrt(s)
        v = sqrt(s) * v
        # f = fidelity(u, v, P, Q, u0, v0)
    else
        # f = fidelity(u * sqrt(s), sqrt(s) * v, P, Q, u0, v0)
    end
    f = 0

    iso = u => v

    xi < Inf && rescale!(iso, xi)

    return iso, s, 1 - f
end

function absorbweights!(pepo::AbstractPEPO; kwargs...)
    lattice = currentbuffer(pepo)

    for ind in uniqueindices(pepo)
        absorb_axial_weights!(lattice[ind], weights(pepo), ind; kwargs...)
    end
    return pepo
end

function absorb_adjacent_weights!(
    pair::Pair{T}, weights::EdgeLattice, bond::Bond; inverse=false
) where {T<:AbsTen{2,4}}
    uten, vten = pair

    if inverse
        func = w -> try
            inv(sqrt(w))
        catch
            # @warn "inv failed"
            pinv(sqrt(w))
        end
    else
        func = w -> sqrt(w)
    end

    U, V = mapadjacent(func, weights, bond)

    axis = bondaxis(bond)

    if axis == 1
        s1, w1, n1 = U
        e2, s2, n2 = V

        @tensoropt begin
            unew[k b; e ss ww nn] := uten[k b; e s w n] * s1[s; ss] * w1[ww; w] * n1[nn; n]
            vnew[k b; ee ss w nn] := vten[k b; e s w n] * s2[s; ss] * e2[e; ee] * n2[nn; n]
        end

    elseif axis == 2
        e1, w1, n1 = U
        e2, s2, w2 = V

        @tensoropt begin
            unew[k b; ee s ww nn] := uten[k b; e s w n] * e1[e; ee] * w1[ww; w] * n1[nn; n]
            vnew[k b; ee ss ww n] := vten[k b; e s w n] * s2[s; ss] * e2[e; ee] * w2[ww; w]
        end
    end

    copy!(uten, unew)
    copy!(vten, vnew)

    return uten => vten
end

function absorb_axial_weights!(
    site::AbsTen{2,4},
    weights::EdgeLattice{G},
    ind::CartesianIndex{2};
    inverse=false,
    axis=1,
) where {G}
    if inverse
        gw = x -> begin
            try
                return inv(sqrt(weights[x]))
            catch e
                if e isa LAPACKException
                    return pinv(sqrt(weights[x]))
                else
                    rethrow()
                end
            end
        end
    else
        gw = x -> sqrt(weights[x])
    end

    e, s, w, n = gw.(bondswith(G, ind))

    if axis == 1
        @tensoropt begin
            site[k b; e ss w nn] = copy(site)[k b; e s w n] * s[s; ss] * n[nn; n]
        end

    elseif axis == 2
        @tensoropt begin
            site[k b; ee s ww n] = copy(site)[k b; e s w n] * e[e; ee] * w[ww; w]
        end
    end

    return site
end
