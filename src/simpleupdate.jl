# One simple-update sweep for a Liouvillian propagator, mirroring PEPSKit's `_su_iter!`. Only the
# reduced bond split and the propagator application differ; cluster extraction, weight absorption and
# bond rotation are PEPSKit's.

"""
    $(TYPEDSIGNATURES)

Update the bond joining `sites` in place: absorb the neighbouring weights, apply `propagator` to the
pair, truncate, and write the two tensors and the new bond weight back. Returns the truncation error.

Without `trunc` the bond space is preserved, taken from the weight currently on that bond. Passing
`truncalg` (a `FullEnvTruncation` or `ALSTruncation`) together with `env`, the CTMRG boundary of the
PEPO sandwiched with its adjoint, truncates against that environment instead of by a bare SVD.
"""
function bondstep!(
        state::InfinitePEPO, weights::SUWeight, propagator::AbstractTensorMap,
        sites::Vector{CartesianIndex{2}}; trunc = nothing, truncalg = nothing, env = nothing
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

    a, s, b, ϵ = if isnothing(truncalg)
        applypropagator(a, b, propagator, truncation)
    else
        # The environment metric is the Hilbert-Schmidt norm: `env` is the boundary of the PEPO
        # sandwiched with its adjoint, and both physical legs sit on the reduced tensors, so `X'X`
        # in `bondenv_ctm` is a purely virtual contraction.
        benv = PEPSKit.bondenv_ctm(bond[2], bond[3], X, Y, env)
        P = space(A, 1)
        aout, sout, bout, info = truncatebond(
            applypropagator(a, b, propagator), benv, truncalg, P
        )
        (aout, sout, bout, info.fid)
    end

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

Update every bond of the unit cell along `axis` (1 for the column direction, 2 for the row direction),
returning the largest truncation error.

Each site sits on two bonds of a given axis and each bond carries half of the one-local part of
`propagator` (see `twolocal`), so one sweep applies that part once per site. Sweeping both axes
therefore applies it twice, which is why [`trotterise`](@ref) halves it.
"""
function axissweep!(
        state::InfinitePEPO, weights::SUWeight, propagator::AbstractTensorMap, axis::Int;
        trunc = nothing, truncalg = nothing, env = nothing
    )
    axis in (1, 2) || throw(ArgumentError("axis must be 1 or 2, got $axis"))
    Nr, Nc, = size(state)
    (Nr >= 2 && Nc >= 2) ||
        throw(ArgumentError("a two-site update needs a unit cell of at least 2x2, got ($Nr, $Nc)"))

    ϵ = 0.0
    for r in 1:Nr, c in 1:Nc
        site1 = CartesianIndex(r, c)
        site2 = axis == 1 ? CartesianIndex(r, c + 1) : CartesianIndex(r + 1, c)
        ϵ = max(ϵ, bondstep!(state, weights, propagator, [site1, site2]; trunc, truncalg, env))
    end

    return ϵ
end

"""
    $(TYPEDSIGNATURES)

One time step: sweep each `(propagator, axis)` of a Trotter decomposition in turn, returning the
largest truncation error.
"""
function timestep!(
        state::InfinitePEPO, weights::SUWeight, decomposition::AbstractVector;
        trunc = nothing, truncalg = nothing, env = nothing
    )
    ϵ = 0.0
    for (propagator, axis) in decomposition
        ϵ = max(ϵ, axissweep!(state, weights, propagator, axis; trunc, truncalg, env))
    end
    return ϵ
end
