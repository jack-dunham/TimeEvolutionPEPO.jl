struct GlobalUpdatablePEPO{G,PType,CType} <: AbstractUpdatablePEPO
    parent::PEPO{G,PType,CType}
    evolved_tensors::Lattice{G,PType,Matrix{PType}}
    isevolved::Ref{Bool}
end

#? intermediate tensors

# GloballyEvolvingPEPO(pepo::PEPO) = GloballyEvolvingPEPO(pepo, deepcopy(getstate(pepo)))

getcurrentstate(p::GlobalUpdatablePEPO) = _getcurrentstate(p, Val(isevolved(p)))
_getcurrentstate(p, ::Val{true}) = evolvedtensors(p)
_getcurrentstate(p, ::Val{false}) = latticetensors(p)

function _getindex(p::GlobalUpdatablePEPO, bond::Bond, ::Val{true})
    return getindex(evolvedtensors(p), bond)
end

function copyevolved(pepo::G) where {G<:GlobalUpdatablePEPO}
    newpepo = G(pepo.parent, copy(pepo.evolved_tensors), pepo.isevolved)
    return newpepo
end

function truncateinto!(dst::G, src::G) where {G<:GlobalUpdatablePEPO}
    copy!.(latticetensors(dst), evolvedtensors(src))
    setisevolved!(dst, false)
    return dst
end
