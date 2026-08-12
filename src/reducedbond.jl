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
        propagator::AbstractTensorMap, trunc::TruncationStrategy
    ) where {S}
    @tensor ab[-1 -2 -3; -4 -5 -6] :=
        propagator[-2 -3 -5 -6; k1 c1 k2 c2] * a[-1 k1 c1 x] * b[x k2 c2 -4]

    u, s, v, ϵ = svd_trunc!(permute(ab, ((1, 2, 3), (4, 5, 6))); trunc)
    u, v = PEPSKit.absorb_s(u, s, v)

    # `v` arrives as `χ ← (bond, ket, bra)`; the second tensor of a bond is `(bond, ket, bra) ← χ`.
    return u, s, permute(v, ((1, 3, 4), (2,))), ϵ
end
