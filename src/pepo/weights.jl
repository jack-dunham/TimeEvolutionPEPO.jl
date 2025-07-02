function setweight!(weights::UnitCellBonds, value, bond::Bond; copy=false)
    if copy
        copyindex!(weights, value, bond)
    else
        weights[bond] = value
    end

    return weights
end

function tolerance!(buffer::W, weights::W, tau::Float64=1.0) where {W<:UnitCellBonds}
    tol = broadcast(buffer, weights) do dst, src
        return _tolerance!(dst, src, tau)
    end

    return maximum(tol[bond] for bond in eachbond(tol))
end
function _tolerance!(old, new, tau::Float64)
    if space(old) != space(new)
        return Inf
    end

    rv = norm(old - new) / (tau * norm(new) * dim(codomain(new)))

    # rv = abs(bondentropy(new; normalised=true) - bondentropy(old; normalised=true)) / tau

    # Its a good idea to create a copy of the old weights, rather than updating the buffer
    # lattice to prevent further mutation to the old weights
    copy!(old, new)
    return rv
end

function bondentropy(weights::UnitCellBonds; kwargs...)
    return broadcast(w -> bondentropy(w; kwargs...), weights)
end
"""
    $(TYPEDSIGNATURES)

Compute the bond entropy of bond weight `w`. If `normalised = true`, then also divide the
result by `log(D)` where `D` is the dimension of the codomain/domain of `w`.

See also: [`PEPO`](@ref), [`bondweights`](@ref)
"""
function bondentropy(w::AbstractTensorMap; normalized=false)
    w = tracenormalize!(copy(w))

    wvec = diag(convert(Array, w))

    wvec /= sum(wvec)

    entf = λ -> λ ≈ 0 ? zero(λ) : -λ * log(λ)

    entropy = sum(entf, wvec) / log(length(wvec))^normalized

    return entropy
end

# function bondentropy(weight::DiagonalTensorMap; normalised=false)
#     entropies = broadcast(weights) do w
#         w = maxnormalize!(copy(w))
#
#         wvec = diag(convert(Array, w))
#
#         entropy = sum(λ -> -λ * log(λ), wvec) / log(length(wvec)^normalised)
#
#         return entropy
#     end
#
#     return entropies
# end
