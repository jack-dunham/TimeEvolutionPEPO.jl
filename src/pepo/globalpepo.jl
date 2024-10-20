mutable struct GlobalUpdatablePEPO{G,PType,CType} <: AbstractUpdatablePEPO
    const parent::PEPO{G,PType,CType}
    const evolved_tensors::Lattice{G,PType,Matrix{PType}}
    isevolved::Bool
end

#? intermediate tensors

# GloballyEvolvingPEPO(pepo::PEPO) = GloballyEvolvingPEPO(pepo, deepcopy(getstate(pepo)))

activetensors(p::GlobalUpdatablePEPO) = _getcurrentstate(p, Val(isevolved(p)))
_getcurrentstate(p, ::Val{true}) = currentbuffer(p)
_getcurrentstate(p, ::Val{false}) = tensors(p)

function _getindex(p::GlobalUpdatablePEPO, bond::Bond, ::Val{true})
    return getindex(currentbuffer(p), bond)
end

function copyevolved(pepo::G) where {G<:GlobalUpdatablePEPO}
    newpepo = G(pepo.parent, copy(pepo.evolved_tensors), pepo.isevolved)
    return newpepo
end

function truncateinto!(dst::G, src::G) where {G<:GlobalUpdatablePEPO}
    copy!.(tensors(dst), currentbuffer(src))
    setisevolved!(dst, false)
    return dst
end

function changeaxis!(dstpepo::GlobalUpdatablePEPO, srcpepo::GlobalUpdatablePEPO)
    srclattice = activetensors(srcpepo)

    if isevolved(srcpepo)
        dstlattice = currentbuffer(dstpepo)
        setisevolved!(dstpepo, true)
    else
        dstlattice = tensors(dstpepo)
        setisevolved!(dstpepo, false)
    end

    permutetranspose!(dstlattice, srclattice)
    # New line below
    changeaxis!(weights(dstpepo), weights(srcpepo))
    return dstpepo
end
