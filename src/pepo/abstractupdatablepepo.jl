abstract type AbstractUpdatablePEPO <: AbstractPEPO end

# Get the underlying PEPO.
pepo(p::AbstractUpdatablePEPO) = p.parent

# Get the tensors of the underlying parent PEPO structure
tensors(p::AbstractUpdatablePEPO) = tensors(pepo(p))

# Get the evolved state of the PEPO. May or may not be up to date.
currentbuffer(p::AbstractUpdatablePEPO) = p.evolved_tensors

# Check if PEPO is in an evolved state (i.e. if `getevolved` outputs the current state.)
isevolved(p::AbstractUpdatablePEPO) = p.isevolved

# Set evolved state.
setisevolved!(p::AbstractUpdatablePEPO, bool::Bool) = p.isevolved = bool

function _getindex(p::AbstractUpdatablePEPO, bond::Bond, ::Val{false})
    return getindex(tensors(p), bond)
end

Base.getindex(p::AbstractUpdatablePEPO, bond::Bond) = _getindex(p, bond, Val(isevolved(p)))

getweight(pepo::AbstractUpdatablePEPO, args...) = getweight(pepo(pepo), args...)
