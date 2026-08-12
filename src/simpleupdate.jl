# One simple-update sweep for a Liouvillian propagator, mirroring PEPSKit's `_su_iter!`. Only the
# reduced bond split and the propagator application differ; cluster extraction, weight absorption and
# bond rotation are PEPSKit's.

"""
    $(TYPEDSIGNATURES)

Update the bond joining `sites` in place: absorb the neighbouring weights, apply `propagator` to the
pair, truncate, and write the two tensors and the new bond weight back. Returns the truncation error.

Without `trunc` the bond space is preserved, taken from the weight currently on that bond.
"""
function bondstep!(
        state::InfinitePEPO, weights::SUWeight, propagator::AbstractTensorMap,
        sites::Vector{CartesianIndex{2}}; trunc = nothing
    )
    length(sites) == 2 || throw(ArgumentError("a bond joins two sites, got $(length(sites))"))

    bond, rev = PEPSKit._nn_bondrev(sites...)
    truncation = if isnothing(trunc)
        V = space(weights[bond...], 1)
        truncspace(isdual(V) ? flip(V) : V)   # `truncspace` rejects dual spaces
    else
        trunc
    end

    Ms, open_vaxs, = PEPSKit._get_cluster(state, sites)
    PEPSKit._absorb_weight!(Ms, sites, open_vaxs, weights)

    # After rotation the updated bond is always the east bond of the first tensor.
    dir = first(bond)
    A, B = PEPSKit._bond_rotation.(Ms, dir, rev; inv = false)

    a, X = reducedbond_first(A)
    b, Y = reducedbond_last(B)
    a, s, b, ϵ = applypropagator(a, b, propagator, truncation)
    A = undo_reducedbond_first(a, X)
    B = undo_reducedbond_last(b, Y)

    A = PEPSKit._bond_rotation(A, dir, rev; inv = true)
    B = PEPSKit._bond_rotation(B, dir, rev; inv = true)
    rev && (s = transpose(s))

    siteA, siteB = sites
    A = absorb_weight(A, weights, siteA[1], siteA[2], open_vaxs[1]; inv = true)
    B = absorb_weight(B, weights, siteB[1], siteB[2], open_vaxs[2]; inv = true)

    state[siteA] = normalize!(A, Inf)
    state[siteB] = normalize!(B, Inf)
    weights[bond...] = normalize!(s, Inf)

    return ϵ
end

"""
    $(TYPEDSIGNATURES)

One Trotter layer: update every bond of the unit cell, along the first axis and then the second.
Returns the largest truncation error over the sweep.

Each site belongs to four bonds, and each bond carries half of the one-local part of `propagator`
(see `twolocal`), so a sweep applies that part twice per site. Build the propagator from a model whose
one-local terms are halved if the sweep is to realise the intended generator.
"""
function trotterstep!(
        state::InfinitePEPO, weights::SUWeight, propagator::AbstractTensorMap; trunc = nothing
    )
    Nr, Nc, = size(state)
    (Nr >= 2 && Nc >= 2) ||
        throw(ArgumentError("a two-site update needs a unit cell of at least 2x2, got ($Nr, $Nc)"))

    ϵ = 0.0
    for axis in 1:2, r in 1:Nr, c in 1:Nc
        site1 = CartesianIndex(r, c)
        site2 = axis == 1 ? CartesianIndex(r, c + 1) : CartesianIndex(r + 1, c)
        ϵ = max(ϵ, bondstep!(state, weights, propagator, [site1, site2]; trunc))
    end

    return ϵ
end
