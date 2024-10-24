abstract type AbstractPEPO end

ischeckerboard(pepo::AbstractPEPO) = ischeckerboard(tensors(pepo))
physicalspace(pepo::AbstractPEPO) = getindex.(codomain.(tensors(pepo)), 1)
uniqueindices(pepo::AbstractPEPO) = uniqueindices(tensors(pepo))

Base.size(pepo::AbstractPEPO) = size(tensors(pepo))

function TensorKit.truncspace(pepo::AbstractPEPO, bond::Bond)
    D = bondspace(pepo, bond)
    return truncspace(D)
end

quicknormalize!(pepo::AbstractPEPO) = normalize!.(tensors(pepo))

TensorKit.scalartype(pepo::AbstractPEPO) = scalartype(tensors(pepo))

function setweight!(pepo::AbstractPEPO, value, bond::Bond; kwargs...)
    setweight!(weights(pepo), value, bond; kwargs...)
    return pepo
end

bonds(pepo::AbstractPEPO, args...) = bonds(tensors(pepo), args...)

function changeaxis(pepo::AbstractPEPO)
    pepo_permuted = broadcast(tensors(pepo)) do tensor
        pten = permute(tensor, ((1, 2), (4, 3, 6, 5)); copy=true)
        return pten
    end
    return PEPO(permutedims(pepo_permuted), changeaxis(weights(pepo)))
end

function changeaxis!(pepo_permuted::AbstractPEPO, pepo::AbstractPEPO)
    broadcast(tensors(pepo_permuted), permutedims(tensors(pepo))) do pdst, psrc
        permute!(pdst, psrc, ((1, 2), (4, 3, 6, 5)))
    end
    # New line below
    changeaxis!(weights(pepo_permuted), weights(pepo))
    return pepo_permuted
end

function permutetranspose!(dstlattice, srclattice)
    broadcast(dstlattice, permutedims(srclattice)) do pdst, psrc
        if pdst === psrc
            psrc = copy(psrc)
        end
        permute!(pdst, psrc, ((1, 2), (4, 3, 6, 5)))
    end
    return dstlattice
end

function permutetranspose(lattice)
    rv = broadcast(permutedims(lattice)) do p
        return permute(p, ((1, 2), (4, 3, 6, 5)))
    end
    return rv
end

function permuteweights!(dstpepo, srcpepo)
    w1 = getweight(dstpepo)
    w2 = getweight(srcpepo)

    changeaxis!(w1, w2)

    return dstpepo
end

bondspace(pepo::AbstractPEPO, bond; kwargs...) = bondspace(tensors(pepo), bond; kwargs...)
function bondspace(
    lat::AbstractLattice{G,<:AbsTen{2,4}}, bond; axis=bondaxis(bond)
) where {G}
    @assert axis in (1, 2)
    return domain.(lat, axis)[first(bond)]
end
function weights2identity!(pepo::AbstractPEPO)
    tensors = currentbuffer(pepo)

    DH = domain.(tensors, 1)
    DV = domain.(tensors, 2)

    w = weights(pepo)

    WH = id.(DH)
    WV = id.(DV)

    setweight!.(Ref(w), WH, bonds(w, 1); copy=false)
    setweight!.(Ref(w), WV, bonds(w, 2); copy=false)

    return pepo
end

Base.getindex(pepo::AbstractPEPO, bond::Bond) = currentbuffer(pepo)[bond]

hermitianize!(pepo::AbstractPEPO) = hermitianize!.(tensors(pepo))
