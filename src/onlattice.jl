@doc raw"""
    AbstractTensorLattice{T<:TensorStyle, L<:LatticeSize,D}

Abstract supertype denoting a size `L` lattice of tensor-like objects of type `D`. The 
parameter `T` refers to the style of the underlying tensors-like objects and is used for 
dispatch only.
"""
abstract type AbstractOnLattice{Size<:AbstractLattice,D} <: AbstractLatticeLike{D} end

# const AbstractOnChain{L<:VectorLattice} = AbstractOnLattice{L}
# const AbstractOnUnitCell{Nx,Ny} = AbstractOnLattice{Infinite{Nx,Ny}}
@doc raw"""

    OnLattice{L<:LatticeSize, P<:AbstractPhysicalTensor} <: AbstractTensorLattice

Concrete type denoting a lattice with size `LatticeSize` of physical tensors of type `P`. 

# Fields
- `matrix::Matrix{P}`: the static-sized matrix of physical tensors
"""
struct OnLattice{L<:AbstractLattice,D,A<:AbstractMatrix{D}} <: AbstractOnLattice{L,D}
    lattice::L
    data::A
    function OnLattice(l::L, data::A) where {L,D,A<:AbstractMatrix{D}}
        if size(l) != size(data)
            # throw(
            #     DimensionMismatch("Size of the data does not match the size of the lattice")
            # )
            new{L,D,A}(l, data)
            # elseif !(_is_matching_sub(l, data))
            #     throw(ArgumentError("If using a SubLattice, data must be of type <:SubArray"))
        else
            new{L,D,A}(l, data)
        end
    end
end

function _is_matching_sub(L::AbstractLattice, A::AbstractArray)
    if (typeof(L) <: SubLattice) && !(typeof(A) <: SubArray)
        return false
    else
        return true
    end
end

## Fields
getmatrix(M::AbstractOnLattice) = M.data
lattice(l::AbstractOnLattice) = l.lattice

# Sometimes useful getting the underlying type of the lattice.
latticetype(::Type{<:AbstractOnLattice{L}}) where {L} = L

## Forward `Lattice` methods to `OnLattice`
latticestyle(l::AbstractOnLattice) = latticestyle(lattice(l))
boundarystyle(l::AbstractOnLattice) = boundarystyle(lattice(l))

## Indexing and iteration
# We index based on the underlying lattice. 
function to_latticeindices(L::AbstractOnLattice, i::Vararg{Int,2})
    return to_latticeindices_inbounds(lattice(L), i...)
end

Base.firstindex(M::AbstractOnLattice) = firstindex(getmatrix(M))
Base.lastindex(M::AbstractOnLattice) = lastindex(getmatrix(M))

function Base.setindex!(M::AbstractOnLattice, v, i::Vararg{Int,2})
    inds = to_latticeindices_inbounds(lattice(M), i...)
    getmatrix(M)[inds...] = v
    return M
end
function Base.setindex!(M::AbstractOnLattice, v::NTuple{2}, bond::Bond)
    getmatrix(M)[bond.in] = v[1]
    getmatrix(M)[bond.out] = v[2]
    return M
end
function Base.iterate(M::AbstractOnLattice, state=1)
    l = length(M)
    if state > l
        return nothing
    else
        return M[state], state + 1
    end
end

@doc raw"""
    setsite!(M::AbstractOnLattice, v, i...)

Alternative to `getindex` that ignores transpose etc.
"""
function setsite!(M::AbstractOnLattice, v, i...)
    getmatrix(M)[i...] = v
    return M
end
const SubOnLattice{L<:SubLattice} = OnLattice{L}

const OnUnitCell{Nx,Ny,D,S} = OnLattice{UnitCell{Nx,Ny,S},D}

# function lview(l::AbstractOnLattice, i1, i2)
#     sublattice = lview(lattice(l), i1, i2)
#     return OnLattice(sublattice, @view to_matrix(l)[i1, i2])
# end

function lview(l::AbstractOnLattice, i1, i2)
    sublattice = lview(lattice(l), i1, i2)
    return OnLattice(sublattice, @view getmatrix(l)[latticeviewindices(sublattice)...])
end
# function lview(l::AbstractOnLattice{<:AbstractLattice{1,1}}, i1, i2)
#     return l
# end
# function lview(l::AbstractOnLattice, i1, i2)
#     sublattice = lview(lattice(l), i1, i2)
#     OnLattice(sublattice, to_matrix(l))
# end

# Deal with single tensor inputs
function OnLattice(lattice::AbstractLattice, data::AbstractTensorMap)
    return OnLattice(lattice, fill(data, size(lattice)...))
end

# Dealing with vector inputs
# function OnLattice(lattice::RowLattice{Nx}, data::Vector) where {Nx}
#     return OnLattice(lattice, hcat(data))
# end
# function OnLattice(lattice::ColLattice{Ny}, data::Vector) where {Ny}
#     return OnLattice(lattice, permutedims(hcat(data)))
# end
function OnLattice(lattice::AbstractLattice{1,1}, data::Vector)
    return OnLattice(lattice, data[1])
end

function Base.similar(l::AbstractOnLattice)
    return OnLattice(lattice(l), similar(l.data))
end
function Base.similar(l::AbstractOnLattice, lattice::AbstractLattice)
    return OnLattice(lattice, similar(l.data))
end
function Base.similar(l::AbstractOnLattice, eltype::DataType)
    return OnLattice(lattice(l), similar(l.data, eltype))
end

function Base.permutedims(m::AbstractOnLattice)
    return OnLattice(permutedims(lattice(m)), permutedims(m.data))
end

Base.size(::Type{<:AbstractOnLattice{L}}) where {L} = size(L)
Base.size(m::AbstractOnLattice) = size(typeof(m))

# this function behaves as eltype does for AbstractTensorMap
numbertype(t::Type{<:AbstractTensorMap}) = eltype(t)
numbertype(L::Type{<:AbstractOnLattice}) = numbertype(eltype(L))
numbertype(::DataType) = ComplexF64
numbertype(l) = numbertype(typeof(l))

# # A lattice always has two dimensions as far as iteration is concerned.
Base.ndims(M::AbstractOnLattice) = 2

abstract type AbstractOnLatticeBroadcast <: Broadcast.AbstractArrayStyle{2} end
struct LatticeBroadcast{LType} <: AbstractOnLatticeBroadcast end

LatticeBroadcast{LType}(::Val{2}) where {LType} = LatticeBroadcast{LType}()

function Base.BroadcastStyle(
    ::Type{<:AbstractOnLattice{LType}}
) where {LType<:AbstractLattice}
    return LatticeBroadcast{LType}()
end
function Base.BroadcastStyle(::Type{LType}) where {LType<:AbstractLattice}
    return LatticeBroadcast{LType}()
end

function Base.similar(
    bc::Broadcast.Broadcasted{<:LatticeBroadcast{LType}}, ::Type{ElType}
) where {LType,ElType}
    obj = find_lattice(bc)
    return OnLattice(
        lattice(obj), similar(Matrix{ElType}, _broadcast_axes(lattice(obj), bc))
    )
end

_broadcast_axes(::TransposedLattice, bc) = axes(bc)[2], axes(bc)[1]
_broadcast_axes(::Lattice, bc) = axes(bc)
_broadcast_axes(lat::SubLattice, bc) = _broadcast_axes(lat.parent, bc)

find_lattice(bc::Base.Broadcast.Broadcasted) = find_lattice(bc.args)
find_lattice(args::Tuple) = find_lattice(find_lattice(args[1]), Base.tail(args))
find_lattice(x) = x
find_lattice(::Tuple{}) = nothing
find_lattice(a::AbstractLatticeLike, rest) = a
# find_lattice(a::Base.Broadcast.Extruded{<:AbstractLatticeLike}, rest) = a.x #?
find_lattice(::Any, rest) = find_lattice(rest)

function Base.hcat(
    l1::AbstractOnLattice,
    l2::AbstractOnLattice,
    l3::AbstractOnLattice,
    args::AbstractOnLattice...,
)
    return hcat(hcat(l1, l2), l3, args...)
end
Base.hcat(a::AbstractOnLattice) = a

function Base.hcat(a::AbstractOnLattice, b::AbstractOnLattice)
    return _hcat(latticestyle(a), latticestyle(b), a, b)
end
# function _hcat(
#     ::Infinite{Nx,Ny}, ::Infinite{Mx,My}, a::AbstractOnLattice, b::AbstractOnLattice
# ) where {Nx,Ny,Mx,My}
#     if Nx != Mx
#         throw(DimensionMismatch("Unequal dimensions along axis"))
#     end
#     N = Nx
#     L = Infinite{N,Ny + My}()
#     return OnLattice(L, hcat(a.data, b.data))
# end

LinearAlgebra.transpose(L::OnLattice) = OnLattice(transpose(lattice(L)), getmatrix(L))

@doc raw"""

    fill(value, lattice:::AbstractLattice)

Put `value` on every site of `lattice`, thus constructing an `OnLattice` object.
"""
Base.fill(value, lattice::Lattice) = OnLattice(lattice, fill(value, size(lattice)))
function Base.fill(value, lattice::TransposedLattice)
    return OnLattice(lattice, fill(value, size(lattice.parent)))
end

# The purpose of this function is to allow a combination of inputs to be supplied 
# to contructors. Scalar inputs are filled on a lattice, matrix inputs are converted
# to be on a lattice, and objects already on lattices are left alone.
_fill_maybe(value, lattice::AbstractLattice) = fill(value, lattice)
_fill_maybe(value::AbstractMatrix, lattice::AbstractLattice) = OnLattice(lattice, value)
_fill_maybe(value::AbstractOnLattice{L}, lattice::L) where {L<:AbstractLattice} = value

# The reason for this recursive function is that a basic implementation where we just
# broadcast across the tuple of varargs isn't type stable (I am not sure why...).
function _fill_all_maybe(lattice::AbstractLattice, value1, value2)
    return _fill_maybe(value1, lattice), _fill_maybe(value2, lattice)
end
function _fill_all_maybe(lattice::AbstractLattice, values...)
    return _fill_maybe(values[begin], lattice),
    _fill_all_maybe(lattice, values[2:end]...)...
end

TensorKit.spacetype(onlattice::AbstractOnLattice) = spacetype(lattice(onlattice))
