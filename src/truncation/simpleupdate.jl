# One simple-update sweep for a Liouvillian propagator, mirroring PEPSKit's `_su_iter!`. Only the
# reduced bond split and the propagator application differ; cluster extraction, weight absorption and
# bond rotation are PEPSKit's.

"""
$(TYPEDEF)

How a sweep updates each bond and maintains the state between steps. PEPSKit exports its own
`SimpleUpdate`, so this carries a different name; it configures the open-system sweep in
[`bondstep!`](@ref) rather than PEPSKit's.

# Constructors

    DampedUpdate(; trunc, truncalg, envalg, zeta, gaugeevery, iterative)

- `trunc`: a TensorKit `TruncationStrategy`; `nothing` keeps each bond at the space it already has
- `truncalg`: a `FullEnvTruncation` or `ALSTruncation`, optimising against a bond environment
- `envalg`: a [`BondEnvironment`](@ref) for the CTMRG boundary `truncalg` needs
- `zeta`: the damping parameter of [`dampweight`](@ref); `Inf` leaves the spectrum alone
- `gaugeevery`: re-fix the gauge every this many steps, and once more at the end; `0` never
- `iterative`: an [`IterativeSU`](@ref), how a long-range layer ([`PEPSOLayer`](@ref)) is applied

`truncalg` and `envalg` are given together or not at all.
"""
struct DampedUpdate{T, A, E, I}
    trunc::T
    truncalg::A
    envalg::E
    zeta::Float64
    gaugeevery::Int
    iterative::I

    function DampedUpdate(
            trunc::T, truncalg::A, envalg::E, zeta, gaugeevery, iterative::I
        ) where {T, A, E, I}
        isnothing(truncalg) == isnothing(envalg) ||
            throw(ArgumentError("environment-based truncation needs both `truncalg` and `envalg`"))
        zeta >= 0 || throw(ArgumentError("the damping parameter must be non-negative, got $zeta"))
        gaugeevery >= 0 ||
            throw(ArgumentError("`gaugeevery` counts steps and cannot be negative, got $gaugeevery"))
        return new{T, A, E, I}(trunc, truncalg, envalg, zeta, gaugeevery, iterative)
    end
end

function DampedUpdate(;
        trunc = nothing, truncalg = nothing, envalg = nothing, zeta = Inf, gaugeevery = 0,
        iterative = IterativeSU()
    )
    return DampedUpdate(trunc, truncalg, envalg, zeta, gaugeevery, iterative)
end

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
        sites::Vector{CartesianIndex{2}}, alg::DampedUpdate; env = nothing
    )
    length(sites) == 2 || throw(ArgumentError("a bond joins two sites, got $(length(sites))"))
    trunc, truncalg, zeta = alg.trunc, alg.truncalg, alg.zeta

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
        applypropagator(a, b, propagator, truncation; zeta)
    else
        # `bondenv_ctm` reads a horizontal bond, so the environment is rotated exactly as
        # `_bond_rotation` rotated the tensors, and the bond's position is carried through that
        # rotation. `_prepare_site_tensor` pads our physical-leg-free remainder with a trivial leg.
        benv = PEPSKit.bondenv_ctm(_rotatedbond(state, sites, dir, rev)..., X, Y,
            PEPSKit._bond_rotation(env, dir, rev))
        P = space(A, 1)
        aout, sout, bout, info = truncatebond(
            applypropagator(a, b, propagator), benv, truncalg, P; zeta
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
        state::InfinitePEPO, weights::SUWeight, propagators::AbstractMatrix, axis::Int,
        alg::DampedUpdate; env = nothing
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
        ϵ = max(ϵ, bondstep!(state, weights, prop, [site1, site2], alg; env))
    end

    return ϵ
end

# Keyword forms. `alg` is the primary interface; these build one for a call site that names fields.
# Keywords do not take part in dispatch, so each entry point needs its own method rather than a
# default argument.
const _ALGKW = (:trunc, :truncalg, :envalg, :zeta, :gaugeevery, :iterative)

function bondstep!(
        state::InfinitePEPO, weights::SUWeight, propagator::AbstractTensorMap,
        sites::Vector{CartesianIndex{2}}; env = nothing, kwargs...
    )
    return bondstep!(state, weights, propagator, sites, DampedUpdate(; kwargs...); env)
end

function axissweep!(
        state::InfinitePEPO, weights::SUWeight, propagators::AbstractMatrix, axis::Int;
        env = nothing, kwargs...
    )
    return axissweep!(state, weights, propagators, axis, DampedUpdate(; kwargs...); env)
end

function axissweep!(
        state::InfinitePEPO, weights::SUWeight, propagator::AbstractTensorMap, axis::Int;
        env = nothing, kwargs...
    )
    return axissweep!(state, weights, propagator, axis, DampedUpdate(; kwargs...); env)
end

function timestep!(
        state::InfinitePEPO, weights::SUWeight, decomposition::AbstractVector; kwargs...
    )
    return timestep!(state, weights, decomposition, DampedUpdate(; kwargs...))
end

"""
    $(TYPEDSIGNATURES)

Compatibility form of [`axissweep!`](@ref) for a single bare propagator, applied to every bond of the
unit cell.
"""
function axissweep!(
        state::InfinitePEPO, weights::SUWeight, propagator::AbstractTensorMap, axis::Int,
        alg::DampedUpdate; kwargs...
    )
    return axissweep!(state, weights, fill(propagator, 1, 1), axis, alg; kwargs...)
end

"""
    $(TYPEDSIGNATURES)

Exchange the two sites of a two-site propagator, whose physical legs are ordered
`(ket₁, bra₁, ket₂, bra₂)` in both codomain and domain.
"""
_swapsites(propagator::AbstractTensorMap) = permute(propagator, ((3, 4, 1, 2), (7, 8, 5, 6)))

"""
    $(TYPEDSIGNATURES)

One time step: sweep each `(propagators, axis)` of a Trotter decomposition in turn, and apply each
[`PEPSOLayer`](@ref) by [`iterativestep!`](@ref), returning the largest truncation error.

`truncalg` selects environment-based truncation and needs `envalg`, a [`BondEnvironment`](@ref) or
the boundary bond dimension. One CTMRG boundary is solved for per axis sweep.

`zeta` is the damping parameter: after each bond is truncated its spectrum is multiplied by
`Ξ(ζ)ᵢⱼ = δᵢⱼ exp(-(i-1)/ζ)` (see [`dampweight`](@ref)), suppressing the correlation length of the
evolved state. `Inf` leaves the update unchanged.
"""
function timestep!(
        state::InfinitePEPO, weights::SUWeight, decomposition::AbstractVector, alg::DampedUpdate
    )
    ϵ = 0.0
    env = nothing
    for entry in decomposition
        if entry isa PEPSOLayer
            ϵ = max(ϵ, iterativestep!(state, weights, entry, alg))
            continue
        end
        propagators, axis = entry
        isnothing(alg.truncalg) || ((env, _) = bondenvironment(state, alg.envalg; env₀ = env))
        ϵ = max(ϵ, axissweep!(state, weights, propagators, axis, alg; env))
    end
    return ϵ
end
