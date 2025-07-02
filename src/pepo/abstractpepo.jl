"""
$(TYPEDEF)

Abstract supertype of all `PEPO`-like objects.
"""
abstract type AbstractPEPO{G,PType<:TenAbs{4}} <: AbstractUnitCell{G,PType,Matrix{PType}} end

physicalspace(pepo::AbstractPEPO) = getindex.(codomain.(pepo), 1)

function TensorKit.truncspace(pepo::AbstractPEPO, bond::Bond)
    D = bondspace(pepo, bond)
    return truncspace(D)
end

quicknormalize!(pepo::AbstractPEPO) = normalize!.(pepo)

function setweight!(pepo::AbstractPEPO, value, bond::Bond; kwargs...)
    setweight!(bondweights(pepo), value, bond; kwargs...)
    return pepo
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

function bondspace(lat::AbstractPEPO, bond; axis=bondaxis(bond))
    @assert axis in (1, 2)
    return domain.(lat, axis)[first(bond)]
end
function weights2identity!(pepo::AbstractPEPO)
    DH = domain.(pepo, 1)
    DV = domain.(pepo, 2)

    w = bondweights(pepo)

    WH = id.(DH)
    WV = id.(DV)

    setweight!.(Ref(w), WH, bonds(w, 1); copy=false)
    setweight!.(Ref(w), WV, bonds(w, 2); copy=false)

    return pepo
end

hermitianize!(pepo::AbstractPEPO) = hermitianize!.(pepo)
bondentropy(pepo::AbstractPEPO) = bondentropy(bondweights(pepo))
