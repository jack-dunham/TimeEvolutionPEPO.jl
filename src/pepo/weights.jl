function setweight!(weights::EdgeLattice, value, bond::Bond; copy=true, axis=bondaxis(bond))
    bond = axis == bondaxis(bond) ? bond : transpose(bond)

    if copy
        copyindex!(weights, value, bond)
    else
        weights[bond] = value
    end

    return weights
end

function tolerance!(buffer::W, weights::W, tau::Float64) where {W<:EdgeLattice}
    tol = broadcast(buffer, weights) do dst, src
        return _tolerance!(dst, src, tau)
    end

    return maximum(tol[bond] for bond in bonditer(tol))
end
function _tolerance!(ow, nw, tau::Float64)
    if space(ow) != space(nw)
        return Inf
    end

    rv = norm(ow - nw) / (tau * norm(nw) * dim(codomain(nw)))
    copy!(ow, nw)
    return rv
end

function bondentropy(weights::W) where {W<:EdgeLattice}

    entropies = broadcast(weights) do w
        wvec = diag(convert(Array, w))

        entropy = sum(λ -> - λ * log(λ), wvec)

        return entropy 
    end

    return entropies
end
