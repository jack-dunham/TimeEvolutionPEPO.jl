# One simple-update sweep for a Liouvillian propagator, mirroring PEPSKit's `_su_iter!`. Only the
# reduced bond split and the propagator application differ; cluster extraction, weight absorption and
# bond rotation are PEPSKit's.

"""
    $(TYPEDSIGNATURES)

Position of a bond's first site after `_bond_rotation` has turned the bond horizontal. Julia's
`rotr90(A)[i, j] == A[Nr + 1 - j, i]`, so a site at `(r, c)` lands at `(c, Nr + 1 - r)`, and `rotl90`
sends it to `(Nc + 1 - c, r)`.
"""
function _rotatedbond(state::InfinitePEPO, sites::Vector{CartesianIndex{2}}, dir::Int, rev::Bool)
    Nr, Nc, = size(state)
    r, c = Tuple(first(sites))
    return if dir == 1
        rev ? (Nr + 1 - r, Nc + 1 - c) : (r, c)
    else
        rev ? (Nc + 1 - c, r) : (c, Nr + 1 - r)
    end
end

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
        # `bondenv_ctm` reads a horizontal bond, so the environment is rotated exactly as
        # `_bond_rotation` rotated the tensors, and the bond's position is carried through that
        # rotation. `_prepare_site_tensor` pads our physical-leg-free remainder with a trivial leg.
        benv = PEPSKit.bondenv_ctm(_rotatedbond(state, sites, dir, rev)..., X, Y,
            PEPSKit._bond_rotation(env, dir, rev))
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
`propagators` (see `twolocal`), so one sweep applies that part once per site. Sweeping both axes
therefore applies it twice, which is why [`trotterise`](@ref) halves it.
"""
function axissweep!(
        state::InfinitePEPO, weights::SUWeight, propagators::AbstractMatrix, axis::Int;
        trunc = nothing, truncalg = nothing, env = nothing
    )
    axis in (1, 2) || throw(ArgumentError("axis must be 1 or 2, got $axis"))
    Nr, Nc, = size(state)
    (Nr >= 2 && Nc >= 2) ||
        throw(ArgumentError("a two-site update needs a unit cell of at least 2x2, got ($Nr, $Nc)"))

    ϵ = 0.0
    for r in 1:Nr, c in 1:Nc
        site1, site2 = if axis == 1
            CartesianIndex(r, c), CartesianIndex(r, c + 1)
        else
            CartesianIndex(r + 1, c), CartesianIndex(r, c)
        end
        p = propagators[mod1(r, size(propagators, 1)), mod1(c, size(propagators, 2))]
        # Vertical bonds run south to north: the other order sets `rev` in `_nn_bondrev`, and the
        # update then writes each bond back with the opposite arrow.
        prop = axis == 1 ? p : _swapsites(p)
        ϵ = max(ϵ, bondstep!(state, weights, prop, [site1, site2]; trunc, truncalg, env))
    end

    return ϵ
end

"""
    $(TYPEDSIGNATURES)

Compatibility form of [`axissweep!`](@ref) for a single bare propagator, applied to every bond of the
unit cell.
"""
function axissweep!(
        state::InfinitePEPO, weights::SUWeight, propagator::AbstractTensorMap, axis::Int; kwargs...
    )
    return axissweep!(state, weights, fill(propagator, 1, 1), axis; kwargs...)
end

"""
    $(TYPEDSIGNATURES)

Exchange the two sites of a two-site propagator, whose physical legs are ordered
`(ket₁, bra₁, ket₂, bra₂)` in both codomain and domain.
"""
_swapsites(propagator::AbstractTensorMap) = permute(propagator, ((3, 4, 1, 2), (7, 8, 5, 6)))

"""
    $(TYPEDSIGNATURES)

One time step: sweep each `(propagators, axis)` of a Trotter decomposition in turn, returning the
largest truncation error.

`truncalg` selects environment-based truncation and needs `envalg`, a [`BondEnvironment`](@ref) or
the boundary bond dimension. One CTMRG boundary is solved for per axis sweep.
"""
function timestep!(
        state::InfinitePEPO, weights::SUWeight, decomposition::AbstractVector;
        trunc = nothing, truncalg = nothing, envalg = nothing
    )
    isnothing(truncalg) == isnothing(envalg) ||
        throw(ArgumentError("environment-based truncation needs both `truncalg` and `envalg`"))

    ϵ = 0.0
    env = nothing
    for (propagators, axis) in decomposition
        isnothing(truncalg) || ((env, _) = bondenvironment(state, envalg; env₀ = env))
        ϵ = max(ϵ, axissweep!(state, weights, propagators, axis; trunc, truncalg, env))
    end
    return ϵ
end
