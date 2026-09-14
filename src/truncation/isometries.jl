# Reduced bond tensors carrying both physical legs. PEPSKit's `bond_tensor_first`/`bond_tensor_last`
# split off one physical leg, selected by `gate_ax`, because their two-site operators act on the ket and
# the bra in separate passes; a Liouvillian acts on both at once. These four functions are the only part
# of PEPSKit's simple-update step that has to be mirrored — cluster extraction, weight absorption, bond
# rotation and the truncations are reused unchanged.
#
# A `PEPOTensor` has indices `(ket, bra; north, east, south, west)`. Splitting the first tensor of a
# bond leaves `X` with `(north, χ, south, west)` and the second leaves `Y` with `(north, east, south, χ)`.

"""
    $(TYPEDSIGNATURES)

Reduced tensor on the east bond of the first tensor of a bond, with both physical legs retained:
`(χ ⊗ ket ⊗ bra) ← east`, alongside the orthogonal remainder.
"""
function reducedbond_first(A::PEPOTensor; kwargs...)
    X, a = left_orth!(permute(A, ((3, 5, 6), (1, 2, 4)); copy = true); kwargs...)
    return permute(a, ((1, 2, 3), (4,))), permute(X, (1, 4, 2, 3))
end

"""
    $(TYPEDSIGNATURES)

Inverse of [`reducedbond_first`](@ref).
"""
function undo_reducedbond_first(a::GenericMPSTensor{S, 3}, X::AbstractTensorMap) where {S}
    return @tensor A[-1 -2; -3 -4 -5 -6] := X[-3 1 -5 -6] * a[1 -1 -2 -4]
end

"""
    $(TYPEDSIGNATURES)

Reduced tensor on the west bond of the second tensor of a bond, with both physical legs retained:
`(west ⊗ ket ⊗ bra) ← χ`, alongside the orthogonal remainder.
"""
function reducedbond_last(A::PEPOTensor; kwargs...)
    Y, b = left_orth!(permute(A, ((3, 4, 5), (1, 2, 6)); copy = true); kwargs...)
    return permute(b, ((4, 2, 3), (1,))), permute(Y, (1, 2, 3, 4))
end

"""
    $(TYPEDSIGNATURES)

Inverse of [`reducedbond_last`](@ref).
"""
function undo_reducedbond_last(b::GenericMPSTensor{S, 3}, Y::AbstractTensorMap) where {S}
    return @tensor A[-1 -2; -3 -4 -5 -6] := b[-6 -1 -2 1] * Y[-3 -4 -5 1]
end

"""
    $(TYPEDSIGNATURES)

Apply a two-site `propagator` to the reduced bond tensors `a` and `b` and truncate the bond between
them, returning the updated pair and the bond weight. Mirrors PEPSKit's `_apply_gate` for a propagator
that acts on the ket and bra of both sites at once.
"""
function applypropagator(
        a::GenericMPSTensor{S, 3}, b::GenericMPSTensor{S, 3},
        propagator::AbstractTensorMap, trunc::TruncationStrategy; zeta = Inf
    ) where {S}
    u, s, v, ϵ = svd_trunc!(applypropagator(a, b, propagator); trunc)
    s = dampweight(s, zeta)
    u, v = PEPSKit.absorb_s(u, s, v)

    # `v` arrives as `χ ← (bond, ket, bra)`; the second tensor of a bond is `(bond, ket, bra) ← χ`.
    return u, s, permute(v, ((1, 3, 4), (2,))), ϵ
end

"""
    $(TYPEDSIGNATURES)

Apply `propagator` to the pair without truncating, returning the merged tensor split as
`(χ ⊗ ket₁ ⊗ bra₁) ← (bond ⊗ ket₂ ⊗ bra₂)`.
"""
function applypropagator(
        a::GenericMPSTensor{S, 3}, b::GenericMPSTensor{S, 3}, propagator::AbstractTensorMap
    ) where {S}
    @tensor ab[-1 -2 -3; -4 -5 -6] :=
        propagator[-2 -3 -5 -6; k1 c1 k2 c2] * a[-1 k1 c1 x] * b[x k2 c2 -4]
    return permute(ab, ((1, 2, 3), (4, 5, 6)))
end

"""
    $(TYPEDSIGNATURES)

Multiply a bond spectrum by the damping matrix `Ξ(ζ)ᵢⱼ = δᵢⱼ exp(-(i-1)/ζ)` and renormalise it to
unit trace, throttling the growth of bond entropy. `ζ = Inf` returns the spectrum unchanged and
`ζ = 0` keeps only the leading value.

Damping is by rank rather than by stored position. The trace normalisation follows the definition of
the transformation; it has no effect on the state here, because [`bondstep!`](@ref) rescales the two
site tensors and the weight to unit maximum afterwards.
"""
function dampweight(s::DiagonalTensorMap, ζ::Real)
    ζ >= 0 || throw(ArgumentError("the damping parameter must be non-negative, got ζ = $ζ"))
    isinf(ζ) && return s
    vals = copy(s.data)
    rank = invperm(sortperm(vals; by = abs, rev = true))
    for i in eachindex(vals)
        k = rank[i] - 1
        vals[i] *= iszero(ζ) ? (iszero(k) ? one(k) : zero(k)) : exp(-k / ζ)
    end
    total = sum(abs, vals)
    iszero(total) || (vals ./= total)
    return DiagonalTensorMap(vals, s.domain)
end

"""
    $(TYPEDSIGNATURES)

Isometry fusing a site's physical pair into one leg. Used only to hand a reduced bond tensor to
PEPSKit's `bond_truncate`, which takes one physical index; the state itself keeps ket and bra apart.
"""
fusephysical(P::IndexSpace) = isomorphism(fuse(P ⊗ P'), P ⊗ P')

"""
    $(TYPEDSIGNATURES)

Truncate an evolved pair against the bond environment `benv`, using `FullEnvTruncation` or
`ALSTruncation`. The physical pair of each reduced tensor is fused for the call and unfused after.
"""
function truncatebond(
        ab::AbstractTensorMap, benv::AbstractTensorMap, alg, P::IndexSpace; zeta = Inf
    )
    # Split without truncating, so the pair enters the optimisation at its full bond dimension.
    u, s, v, = svd_trunc!(copy(ab); trunc = notrunc())
    s = dampweight(s, zeta)
    u, v = PEPSKit.absorb_s(u, s, v)
    a = u
    b = permute(v, ((1, 3, 4), (2,)))

    F = fusephysical(P)
    Fd = F'
    @tensor af[-1 -2; -3] := a[-1 k c -3] * F[-2; k c]
    @tensor bf[-1 -2; -3] := b[-1 k c -3] * F[-2; k c]

    a′, s′, b′, info = PEPSKit.bond_truncate(af, bf, benv, alg)

    @tensor aout[-1 -2 -3; -4] := a′[-1 f -4] * Fd[-2 -3; f]
    @tensor bout[-1 -2 -3; -4] := b′[-1 f -4] * Fd[-2 -3; f]

    return aout, s′, bout, info
end
