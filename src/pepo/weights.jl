struct Weights{G,CType<:AbsTen{1,1}}
    horizontal::Lattice{G,CType,Matrix{CType}}
    vertical::Lattice{G,CType,Matrix{CType}}
end

function Base.getindex(weights::Weights, bond::Bond)
    axis = getaxis(bond)
    if axis == 1
        return weights.horizontal[left(bond)]
    elseif axis == 2
        return weights.vertical[left(bond)]
    else
        throw(
            ErrorException(
                "Impossible `else` reached! Value of axis = $axis returned from `getaxis` neither 1 or 2",
            ),
        )
    end
end

function TensorKit.truncspace(w::Weights, bond::Bond)
    return TensorKit.TruncationSpace(codomain(w[bond])[1])
end

function changeaxis(weights::Weights)
    return Weights(permutedims(weights.vertical), permutedims(weights.horizontal))
end

function tolerance!(weights::W, new_weights::W, tau::Float64) where {W<:Weights}
    h = _tolerance!.(weights.horizontal, new_weights.horizontal, tau)
    v = _tolerance!.(weights.vertical, new_weights.vertical, tau)
    return max(h..., v...)
end
function _tolerance!(ow, nw, tau::Float64)
    rv = norm(ow - nw) / (tau * norm(nw) * dim(codomain(nw)))
    copy!(ow, nw)
    return rv
end
