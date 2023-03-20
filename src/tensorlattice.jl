#PERF: Removed PhysicalTensor
### AbstractTensorLattice

abstract type BoundaryStyle end
struct Periodic <: BoundaryStyle end
struct Open <: BoundaryStyle end

abstract type LatticeStyle{B<:BoundaryStyle} end
struct Finite{B} <: LatticeStyle{B} end
struct Infinite <: LatticeStyle{Periodic} end

isperiodic(::Periodic) = true
isperiodic(::Open) = false
isperiodic(a) = BoundaryStyle(a)

boundarystyle(L::Type{<:LatticeStyle{B}}) where {B} = B()
boundarystyle(L::LatticeStyle{B}) where {B} = B()

abstract type AbstractLatticeLike{S} <: AbstractMatrix{S} end

function to_latticeindices(L::AbstractLatticeLike, i::Vararg{Int,2})
    return to_latticeindices(boundarystyle(L), L, i...)
end
function to_latticeindices(::Periodic, L::AbstractLatticeLike, i::Vararg{Int,2})
    Nx, Ny = size(L)
    return (mod(i[1], 1:Nx), mod(i[2], 1:Ny))
end
to_latticeindices(::Open, L::AbstractLatticeLike, i::Vararg{Int,2}) = i

function Base.getindex(L::AbstractLatticeLike, i::Vararg{Int,2})
    return getindex(to_matrix(L), to_latticeindices(L, i...)...)
end
Base.checkbounds(L::AbstractLatticeLike, i...) = nothing
_checkbounds(::Periodic, L::AbstractLatticeLike, i...) = nothing
_checkbounds(::Open, L::AbstractLatticeLike, i...) = checkbounds(to_matrix(L), i)

abstract type AbstractLattice{Nx,Ny,L<:LatticeStyle,S<:IndexSpace} <: AbstractLatticeLike{S} end

struct Lattice{Nx,Ny,L,S<:IndexSpace} <: AbstractLattice{Nx,Ny,L,S}
    space::Matrix{S}
    issymetric::Bool
    function Lattice{Nx,Ny,B}(space::Matrix{S}, sym::Bool) where {Nx,Ny,B,S<:IndexSpace}
        return new{Nx,Ny,B,S}(space, sym)
    end
end

Base.size(::Type{<:AbstractLattice{Nx,Ny}}) where {Nx,Ny} = (Nx, Ny)

lattice(L::AbstractLattice) = L
latticestyle(::AbstractLattice{Nx,Ny,L}) where {Nx,Ny,L} = L()

to_matrix(L::AbstractLattice) = L.space

boundarystyle(l::AbstractLatticeLike) = boundarystyle(typeof(l))
boundarystyle(l::Type{<:AbstractLattice{Nx,Ny,L}}) where {Nx,Ny,L} = boundarystyle(L)

const UnitCell{Nx,Ny,S} = Lattice{Nx,Ny,Infinite,S}

struct SubLattice{Nx,Ny,L,S,P<:AbstractLattice} <: AbstractLattice{Nx,Ny,L,S}
    parent::P
    index::Int
end

const LatticeViewX{Nx} = SubLattice{Nx,1}
const LatticeViewY = SubLattice{1}

const UnitCellView{Nx,Ny} = SubLattice{Nx,Ny,Infinite}

function SubLattice(P::AbstractLattice{Nx,Ny,L,S}, ::Colon, i::Int) where {Nx,Ny,L,S}
    return SubLattice{Nx,1,L,S,typeof(P)}(P, i)
end
function SubLattice(P::AbstractLattice{Nx,Ny,L,S}, i::Int, ::Colon) where {Nx,Ny,L,S}
    return SubLattice{1,Ny,L,S,typeof(P)}(P, i)
end
lview(P::AbstractLattice, i1, i2) = SubLattice(P, i1, i2)
to_matrix(L::SubLattice) = getindex(to_matrix(L.parent), latticeviewindices(L)...)

function latticeviewindices(L::SubLattice{Nx,1}) where {Nx}
    return to_indices(L.parent, (:, (L.index):(L.index)))
end
function latticeviewindices(L::SubLattice{1,Ny}) where {Ny}
    return to_indices(L.parent, ((L.index):(L.index), :))
end

# Default to symmetric.
function (::Type{T})(sp::AbstractMatrix) where {T<:AbstractLattice}
    return T(sp, true)
end
# function (::Type{T})(
#     localspace::S, sym::Bool
# ) where {Nx,Ny,T<:AbstractLattice{Nx,Ny},S<:IndexSpace}
#     return T(fill(localspace, Nx, Ny), sym)
# end
# function (::Type{T})(d::Int, sym::Bool) where {Nx,Ny,T<:AbstractLattice{Nx,Ny}}
#     return T(ℂ^d, sym)
# end
# function (::Type{T})(d::Matrix{Int}, sym::Bool) where {Nx,Ny,T<:AbstractLattice{Nx,Ny}}
#     return T(broadcast(x -> ℂ^x, d), sym)
# end

function spacematrix(L::Type{<:AbstractLattice{Nx,Ny}}, S::IndexSpace) where {Nx,Ny}
    return fill(S, Nx, Ny)
end
spacematrix(L::Type{<:AbstractLattice}, D::Int) = spacematrix(L, ℂ^D)
spacematrix(L::Type{<:AbstractLattice}, D::Matrix{Int}) = broadcast(x -> ℂ^x, D)
spacematrix(L::Type{<:AbstractLattice}, D::Matrix{<:IndexSpace}) = D
spacematrix(L::AbstractLattice, D) = spacematrix(typeof(L), D)

# boundarystyle(::Type{<:Finite{Nx,Ny,B}}) where {Nx,Ny,B} = B()
# boundarystyle(::Type{<:Infinite}) = Periodic()

Base.size(::Type{<:AbstractLattice{Nx,Ny}}) where {Nx,Ny} = (Nx, Ny)
Base.size(l::AbstractLattice) = size(typeof(l))

# Base.permutedims(::Type{<:Finite{Nx,Ny,L}}) where {Nx,Ny,L} = Finite{Ny,Nx,L}
# Base.permutedims(::Type{<:Infinite{Nx,Ny}}) where {Nx,Ny} = Infinite{Ny,Nx}

Base.permutedims(L::AbstractLattice) = permutedims(typeof(L))(L.issymetric)

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

    TensorLattice{L<:LatticeSize, P<:AbstractPhysicalTensor} <: AbstractTensorLattice

Concrete type denoting a lattice with size `LatticeSize` of physical tensors of type `P`. 

# Fields
- `matrix::Matrix{P}`: the static-sized matrix of physical tensors
"""
struct OnLattice{L<:AbstractLattice,D,A<:AbstractMatrix{D}} <: AbstractOnLattice{L,D}
    lattice::L
    data::A
    function OnLattice(l::L, data::A) where {L,D,A<:AbstractMatrix{D}}
        if size(l) != size(data)
            throw(
                DimensionMismatch("Size of the data does not match the size of the lattice")
            )
        elseif !(_is_matching_sub(l, data))
            throw(ArgumentError("If using a SubLattice, data must be of type <:SubArray"))
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

const SubOnLattice{L<:SubLattice} = OnLattice{L}

Base.similar(L::AbstractOnLattice) = similar.(L)

to_matrix(M::AbstractOnLattice) = M.data

const OnUnitCell{Nx,Ny,D,S} = OnLattice{UnitCell{Nx,Ny,S},D}

function lview(l::AbstractOnLattice, i1, i2)
    sublattice = lview(lattice(l), i1, i2)
    return OnLattice(sublattice, @view to_matrix(l)[i1, i2])
end

# Get lattice, and transform indices based on lattice. 

lattice(l::AbstractOnLattice) = l.lattice
latticetype(::Type{<:AbstractOnLattice{L}}) where {L} = L

latticestyle(l::AbstractOnLattice) = latticestyle(lattice(l))

# boundarystyle(::Type{<:AbstractOnLattice{L}}) where {L} = boundarystyle(L)
boundarystyle(l::AbstractOnLattice) = boundarystyle(latticestyle(l))

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

function Base.permutedims(m::AbstractOnLattice)
    return OnLattice(tensorstyle(m), permutedims(latticestyle(m)), permutedims(m.data))
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

# # Single dimensional indexing
Base.firstindex(M::AbstractOnLattice) = firstindex(to_matrix(M))
Base.lastindex(M::AbstractOnLattice) = lastindex(to_matrix(M))
# Base.getindex(M::AbstractOnLattice, i::Int) = getindex(to_matrix(M), i)

function Base.setindex!(M::AbstractOnLattice{D}, v::D, i::Int) where {D}
    to_matrix(M)[i] = v
    return M
end
function Base.setindex!(M::AbstractOnLattice, v, x::Int, y::Int)
    to_matrix(M)[x, y] = v
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

abstract type AbstractOnLatticeBroadcast <: Broadcast.AbstractArrayStyle{2} end
struct LatticeBroadcast <: AbstractOnLatticeBroadcast end

LatticeBroadcast(::Val{2}) = LatticeBroadcast()

function Base.BroadcastStyle(L::Type{<:AbstractLatticeLike})
    return LatticeBroadcast()
end

function Base.similar(
    bc::Broadcast.Broadcasted{<:LatticeBroadcast}, ::Type{ElType}
) where {ElType}
    obj = find_lattice(bc)
    return OnLattice(lattice(obj), similar(Matrix{ElType}, axes(bc)))
end

find_lattice(bc::Base.Broadcast.Broadcasted) = find_lattice(bc.args)
find_lattice(args::Tuple) = find_lattice(find_lattice(args[1]), Base.tail(args))
find_lattice(x) = x
find_lattice(::Tuple{}) = nothing
find_lattice(a::AbstractOnLattice, rest) = a
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
