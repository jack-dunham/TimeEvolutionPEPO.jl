#PERF: Removed PhysicalTensor
### AbstractTensorLattice

# The type of tensors hosted on the lattice
abstract type TensorStyle end
abstract type PureTensorStyle <: TensorStyle end
struct Generic <: PureTensorStyle end
struct NotTensor <: TensorStyle end

# Standard styles
struct BondMatrix <: PureTensorStyle end
struct MPS <: PureTensorStyle end
struct MPO <: PureTensorStyle end
struct PEPO <: PureTensorStyle end

# Less standard styles
abstract type Env <: PureTensorStyle end
struct LeftEnv <: Env end
struct RightEnv <: Env end
struct TruncMetric <: PureTensorStyle end

Base.promote_rule(::Type{<:PureTensorStyle}, ::Type{<:PureTensorStyle}) = Generic
Base.promote_rule(::Type{NotTensor}, ::Type{<:TensorStyle}) = NotTensor

abstract type BoundaryStyle end
struct Periodic <: BoundaryStyle end
struct Open <: BoundaryStyle end

isperiodic(::Periodic) = true
isperiodic(::Open) = false
isperiodic(a) = BoundaryStyle(a)

abstract type AbstractLattice{Nx,Ny,B<:BoundaryStyle} end

const RowLattice{Nx} = AbstractLattice{Nx,1}
const ColLattice{Ny} = AbstractLattice{1,Ny}
const VectorLattice = Union{RowLattice,ColLattice}

struct Finite{Nx,Ny,B,S<:IndexSpace} <: AbstractLattice{Nx,Ny,B}
    localspace::Matrix{S}
    issymetric::Bool
    function Finite{Nx,Ny,B}(sp::Matrix{S}, sym::Bool) where {Nx,Ny,B,S<:IndexSpace}
        return new{Nx,Ny,B,S}(sp, sym)
    end
end
struct Infinite{Nx,Ny,S<:IndexSpace} <: AbstractLattice{Nx,Ny,Periodic}
    localspace::Matrix{S}
    issymetric::Bool
    function Infinite{Nx,Ny}(sp::Matrix{S}, sym::Bool) where {Nx,Ny,S<:IndexSpace}
        return new{Nx,Ny,S}(sp, sym)
    end
end
# Default to symmetric.
function (::Type{T})(localspace; issymetric::Bool=true) where {T<:AbstractLattice}
    return T(spacematrix(T, localspace), issymetric)
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

boundarystyle(::Type{<:Finite{Nx,Ny,B}}) where {Nx,Ny,B} = B()
boundarystyle(::Type{<:Infinite}) = Periodic()

Base.size(::Type{<:AbstractLattice{Nx,Ny}}) where {Nx,Ny} = (Nx, Ny)
Base.size(l::AbstractLattice) = size(typeof(l))

Base.permutedims(::Type{<:Finite{Nx,Ny,L}}) where {Nx,Ny,L} = Finite{Ny,Nx,L}
Base.permutedims(::Type{<:Infinite{Nx,Ny}}) where {Nx,Ny} = Infinite{Ny,Nx}

Base.permutedims(L::AbstractLattice) = permutedims(typeof(L))(permutedims(L.localspace))

@doc raw"""
    AbstractTensorLattice{T<:TensorStyle, L<:LatticeSize,D}

Abstract supertype denoting a size `L` lattice of tensor-like objects of type `D`. The 
parameter `T` refers to the style of the underlying tensors-like objects and is used for 
dispatch only.
"""
abstract type AbstractOnLattice{T<:TensorStyle,Size<:AbstractLattice,D} <: AbstractMatrix{D} end

const AbstractOnChain{T,L<:VectorLattice} = AbstractOnLattice{T,L}
const AbstractOnUnitCell{T,Nx,Ny} = AbstractOnLattice{T,Infinite{Nx,Ny}}
const AbstractTensorOnLattice{L,D<:AbstractTensorMap} = AbstractOnLattice{L,D}
const AbstractCompositeOnLattice{L,D<:AbstractOnLattice} = AbstractOnLattice{L,D}

@doc raw"""

    TensorLattice{L<:LatticeSize, P<:AbstractPhysicalTensor} <: AbstractTensorLattice

Concrete type denoting a lattice with size `LatticeSize` of physical tensors of type `P`. 

# Fields
- `matrix::Matrix{P}`: the static-sized matrix of physical tensors
"""
struct OnLattice{T<:TensorStyle,L<:AbstractLattice,D} <: AbstractOnLattice{T,L,D}
    style::T
    lattice::L
    data::Matrix{D}
    function OnLattice(t::T, l::L, data::Matrix{D}) where {T,L,D}
        if size(l) != size(data)
            throw(DimensionMismatch(""))
        else
            new{T,L,D}(t, l, data)
        end
    end
end

function convertstyle(T::Type{<:TensorStyle}, l::AbstractOnLattice)
    return OnLattice(T(), latticestyle(l), l.data)
end

to_matrix(M::AbstractOnLattice) = M.data

const OnUnitCell{T,Nx,Ny,D,S} = OnLattice{T,Infinite{Nx,Ny,S},D}
const OnTensorLattice{T,L,D<:AbstractTensorMap} = OnLattice{T,L,D}
const OnCompositeLattice{T,L,D<:AbstractOnLattice} = OnLattice{T,L,D}

tensorstyle(l::AbstractOnLattice) = l.style
tensorstyle(::Type{<:AbstractOnLattice{T}}) where {T} = T

latticestyle(l::AbstractOnLattice) = l.lattice
latticestyle(::Type{<:AbstractOnLattice{T,L}}) where {T,L} = L

boundarystyle(::Type{<:AbstractOnLattice{T,L}}) where {T,L} = boundarystyle(L)
boundarystyle(l::AbstractOnLattice) = boundarystyle(typeof(latticestyle(l)))

# Default
OnLattice(size::AbstractLattice, data) = OnLattice(Generic(), size, data)

# Deal with single tensor inputs
function OnLattice(style::TensorStyle, lattice::AbstractLattice, data::AbstractTensorMap)
    return OnLattice(style, lattice, fill(data, size(lattice)...))
end

# Dealing with vector inputs
function OnLattice(style::TensorStyle, lattice::RowLattice{Nx}, data::Vector) where {Nx}
    return OnLattice(style, lattice, hcat(data))
end
function OnLattice(style::TensorStyle, lattice::ColLattice{Ny}, data::Vector) where {Ny}
    return OnLattice(style, lattice, permutedims(hcat(data)))
end
function OnLattice(style::TensorStyle, lattice::AbstractLattice{1,1}, data::Vector)
    return OnLattice(style, lattice, data[1])
end

function Base.similar(l::AbstractOnLattice)
    return OnLattice(tensorstyle(l), latticestyle(l), similar(l.data))
end
function Base.similar(l::AbstractOnLattice, style::TensorStyle)
    return OnLattice(style, latticestyle(l), similar(l.data))
end
function Base.similar(l::AbstractOnLattice, style::TensorStyle, lattice::AbstractLattice)
    return OnLattice(style, lattice, similar(l.data))
end

function Base.permutedims(m::AbstractOnLattice)
    return OnLattice(tensorstyle(m), permutedims(latticestyle(m)), permutedims(m.data))
end

Base.size(::Type{<:AbstractOnLattice{T,L}}) where {T,L} = size(L)
Base.size(m::AbstractOnLattice) = size(typeof(m))

## Recursively get the type of the tensors stored on the lattice, if it exits.
# D is a AbstractTensorMap
tensortype(::Type{T}) where {T<:AbstractTensorMap} = T
tensortype(::Type{<:AbstractTensorOnLattice{T,L,D}}) where {T,L,D} = D
# D is an AbstractOnLattice, so recurse through
tensortype(::Type{<:AbstractCompositeOnLattice{T,L,D}}) where {T,L,D} = tensortype(D)
# # Fallback option
# function tensortype(::Type{<:AbstractOnLattice})
#     throw(ArgumentError("The lattice provided does not contain tensors at any depth"))
# end
# Value method
tensortype(::L) where {L<:AbstractOnLattice} = tensortype(L)

TensorKit.spacetype(L::Type{<:AbstractOnLattice}) = spacetype(tensortype(L))
TensorKit.spacetype(::L) where {L<:AbstractOnLattice} = spacetype(tensortype(L))

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
Base.getindex(M::AbstractOnLattice, i::Int) = getindex(to_matrix(M), i)

function Base.getindex(M::AbstractOnChain{T}, i::Int) where {T<:TensorStyle}
    return _getindex(boundarystyle(M), M, i)
end
function _getindex(::Periodic, M::AbstractOnChain, i::Int)
    return getindex(to_matrix(M), mod(i, 1:length(M)))
end
function _getindex(::Open, M::AbstractOnChain, i::Int)
    return getindex(to_matrix(M), i)
end

to_periodicindex(c::Colon, ::UnitRange) = c
to_periodicindex(i::Int, r::UnitRange) = mod(i, r)

function Base.getindex(M::AbstractOnLattice, i1, i2)
    return _getindex(boundarystyle(M), M, i1, i2)
end
function _getindex(
    ::Periodic, M::AbstractOnLattice{T,<:AbstractLattice{Nx,Ny}}, i1, i2
) where {T,Nx,Ny}
    return getindex(to_matrix(M), to_periodicindex(i1, 1:Nx), to_periodicindex(i2, 1:Ny))
end
function _getindex(::Open, M::AbstractOnLattice, i1, i2)
    return getindex(to_matrix(M), i1, i2)
end

function Base.getindex(M::AbstractOnLattice, i::CartesianIndex{2})
    return getindex(M, Tuple(i)...)
end

function Base.setindex!(M::AbstractOnLattice{T,D}, v::D, i::Int) where {T,D}
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

abstract type AbstractOnLatticeBroadcast{L<:AbstractLattice} <:
              Broadcast.AbstractArrayStyle{2} end
struct LatticeBroadcast{T,L,ElType} <: AbstractOnLatticeBroadcast{L} end
struct LatticeFallback{L} <: AbstractOnLatticeBroadcast{L} end

LatticeBroadcast{T,L,ElType}(::Val{2}) where {T,L,ElType} = LatticeBroadcast{T,L,ElType}()
LatticeFallback{L}(::Val{2}) where {L} = LatticeFallback{L}()

function Base.BroadcastStyle(::Type{<:AbstractOnLattice{T,L,D}}) where {T,L,D}
    return LatticeBroadcast{T,L,D}()
end

# If element type matches the element type of broadcast style then assume type of tensor
# has remained unchanged, thus keep the same tensor style. 
function Base.similar(
    bc::Broadcast.Broadcasted{LatticeBroadcast{T,L,ElType}}, ::Type{ElType}
) where {T,L,ElType<:AbstractTensorMap}
    lattice = find_lattice(bc)
    size = latticestyle(lattice)
    return OnLattice(T(), size, similar(Matrix{ElType}, axes(bc)))
end

# Else, fallback to Generic()
function Base.similar(
    bc::Broadcast.Broadcasted{<:AbstractOnLatticeBroadcast{L}}, ::Type{ElType}
) where {L,ElType<:AbstractTensorMap}
    lattice = find_lattice(bc)
    size = latticestyle(lattice)
    return OnLattice(Generic(), size, similar(Matrix{ElType}, axes(bc)))
end

# If elements are not tensors, always use tensor style NotTensor()
function Base.similar(
    bc::Broadcast.Broadcasted{<:AbstractOnLatticeBroadcast{L}}, ::Type{ElType}
) where {L,ElType}
    lattice = find_lattice(bc)
    size = latticestyle(lattice)
    return OnLattice(NotTensor(), size, similar(Matrix{ElType}, axes(bc)))
end

find_lattice(bc::Base.Broadcast.Broadcasted) = find_lattice(bc.args)
find_lattice(args::Tuple) = find_lattice(find_lattice(args[1]), Base.tail(args))
find_lattice(x) = x
find_lattice(::Tuple{}) = nothing
find_lattice(a::AbstractOnLattice, rest) = a
find_lattice(::Any, rest) = find_lattice(rest)

# If broadcasting across lattices of tensors, then we do not know the resulting type so
# best we can do is a Generic lattice.
function Base.BroadcastStyle(
    ::LatticeBroadcast{<:PureTensorStyle,L,E1}, ::LatticeBroadcast{<:PureTensorStyle,L,E2}
) where {E1<:AbstractTensorMap,E2<:AbstractTensorMap,L}
    return LatticeFallback{L}()
end
# Lattices of tensors take precedence over lattices of non-tensor objects. 
function Base.BroadcastStyle(
    ::LatticeBroadcast{T1,L,E1}, ::LatticeBroadcast{T2,L,E2}
) where {T1<:PureTensorStyle,T2<:NotTensor,L,E1<:AbstractTensorMap,E2}
    return LatticeBroadcast{T1,L,E1}()
end

function Base.BroadcastStyle(::LatticeBroadcast, ::LatticeBroadcast)
    return Broadcast.Unknown()
end

function Base.BroadcastStyle(
    t::LatticeFallback{L}, ::LatticeBroadcast{<:TensorStyle,L}
) where {L}
    return t
end
function Base.BroadcastStyle(a::LatticeFallback{L}, ::LatticeFallback{L}) where {L}
    return a
end

function Base.hcat(
    l1::AbstractOnLattice,
    l2::AbstractOnLattice,
    l3::AbstractOnLattice,
    args::AbstractOnLattice...,
)
    return hcat(hcat(l1, l2), l3, args...)
end
Base.hcat(a::AbstractOnLattice) = a

function Base.hcat(a::AbstractOnLattice{T1}, b::AbstractOnLattice{T2}) where {T1,T2}
    T = promote_type(T1, T2)
    return _hcat(latticestyle(a), latticestyle(b), T(), a, b)
end
function _hcat(
    ::Infinite{Nx,Ny},
    ::Infinite{Mx,My},
    T::TensorStyle,
    a::AbstractOnLattice,
    b::AbstractOnLattice,
) where {Nx,Ny,Mx,My}
    if Nx != Mx
        throw(DimensionMismatch("Unequal dimensions along axis"))
    end
    N = Nx
    L = Infinite{N,Ny + My}()
    return OnLattice(T, L, hcat(a.data, b.data))
end
