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


abstract type LatticeBoundaryStyle end
struct Periodic <: LatticeBoundaryStyle end
struct Open <: LatticeBoundaryStyle end

isperiodic(::Periodic) = true
isperiodic(::Open) = false
isperiodic(a) = LatticeBoundaryStyle(a)

abstract type LatticeStyle{Nx,Ny,B<:LatticeBoundaryStyle} end

const RowLatticeStyle{Nx} = LatticeStyle{Nx,1}
const ColLatticeStyle{Ny} = LatticeStyle{1,Ny}
const VectorLatticeStyle = Union{RowLatticeStyle,ColLatticeStyle}

struct Finite{Nx,Ny,B} <: LatticeStyle{Nx,Ny,B} end
struct Infinite{Nx,Ny} <: LatticeStyle{Nx,Ny,Periodic} end

boundarystyle(::Type{<:Finite{Nx,Ny,B}}) where {Nx,Ny,B} = B()
boundarystyle(::Type{<:Infinite}) = Periodic()

Base.size(::Type{<:LatticeStyle{Nx,Ny}}) where {Nx,Ny} = (Nx, Ny)
Base.size(l::LatticeStyle) = size(typeof(l))

Base.permutedims(::Type{<:Finite{Nx,Ny,L}}) where {Nx,Ny,L} = Finite{Ny,Nx,L}
Base.permutedims(::Type{<:Infinite{Nx,Ny}}) where {Nx,Ny} = Infinite{Ny,Nx}

Base.permutedims(L::LatticeStyle) = permutedims(typeof(L))()

@doc raw"""
    AbstractTensorLattice{T<:TensorStyle, L<:LatticeSize,D}

Abstract supertype denoting a size `L` lattice of tensor-like objects of type `D`. The 
parameter `T` refers to the style of the underlying tensors-like objects and is used for 
dispatch only.
"""
abstract type AbstractLattice{T<:TensorStyle,Size<:LatticeStyle,D} <: AbstractMatrix{D} end

const AbstractVectorLattice{T,L<:VectorLatticeStyle} = AbstractLattice{T,L}
const AbstractUnitCell{T,Nx,Ny} = AbstractLattice{T,Infinite{Nx,Ny}}
const AbstractTensorLattice{L,D<:AbstractTensorMap} = AbstractLattice{L,D}
const AbstractCompositeLattice{L,D<:AbstractLattice} = AbstractLattice{L,D}

@doc raw"""

    TensorLattice{L<:LatticeSize, P<:AbstractPhysicalTensor} <: AbstractTensorLattice

Concrete type denoting a lattice with size `LatticeSize` of physical tensors of type `P`. 

# Fields
- `matrix::Matrix{P}`: the static-sized matrix of physical tensors
"""
struct Lattice{T<:TensorStyle,L<:LatticeStyle,D} <: AbstractLattice{T,L,D}
    style::T
    lattice::L
    data::Matrix{D}
    function Lattice(t::T, l::L, data::Matrix{D}) where {T,L,D}
        if size(l) != size(data)
            throw(DimensionMismatch(""))
        else
            new{T,L,D}(t, l, data)
        end
    end
end

function convertstyle(T::Type{<:TensorStyle}, l::AbstractLattice)
    return Lattice(T(), latticestyle(l), l.data)
end

to_matrix(M::AbstractLattice) = M.data

const UnitCell{T,Nx,Ny,D} = Lattice{T,Infinite{Nx,Ny},D}
const TensorLattice{T,L,D<:AbstractTensorMap} = Lattice{T,L,D}
const CompositeLattice{T,L,D<:AbstractLattice} = Lattice{T,L,D}

tensorstyle(l::AbstractLattice) = l.style
tensorstyle(::Type{<:AbstractLattice{T}}) where {T} = T

latticestyle(l::AbstractLattice) = l.lattice
latticestyle(::Type{<:AbstractLattice{T,L}}) where {T,L} = L

boundarystyle(::Type{<:AbstractLattice{T,L}}) where {T,L} = boundarystyle(L)
boundarystyle(l::AbstractLattice) = boundarystyle(typeof(latticestyle(l)))

# Default
Lattice(size::LatticeStyle, data) = Lattice(Generic(), size, data)

# Deal with single tensor inputs
function Lattice(style::TensorStyle, lattice::LatticeStyle, data::AbstractTensorMap)
    return Lattice(style, lattice, fill(data, size(lattice)...))
end

# Dealing with vector inputs
function Lattice(style::TensorStyle, lattice::RowLatticeStyle{Nx}, data::Vector) where {Nx}
    return Lattice(style, lattice, hcat(data))
end
function Lattice(style::TensorStyle, lattice::ColLatticeStyle{Ny}, data::Vector) where {Ny}
    return Lattice(style, lattice, permutedims(hcat(data)))
end
function Lattice(style::TensorStyle, lattice::LatticeStyle{1,1}, data::Vector)
    return Lattice(style, lattice, data[1])
end

Base.similar(l::AbstractLattice) = Lattice(tensorstyle(l), latticestyle(l), similar(l.data))
function Base.similar(l::AbstractLattice, style::TensorStyle)
    return Lattice(style, latticestyle(l), similar(l.data))
end
function Base.similar(l::AbstractLattice, style::TensorStyle, lattice::LatticeStyle)
    return Lattice(style, lattice, similar(l.data))
end

function Base.permutedims(m::AbstractLattice)
    return Lattice(tensorstyle(m), permutedims(latticestyle(m)), permutedims(m.data))
end

Base.size(::Type{<:AbstractLattice{T,L}}) where {T,L} = size(L)
Base.size(m::AbstractLattice) = size(typeof(m))

## Recursively get the type of the tensors stored on the lattice, if it exits.
# D is a AbstractTensorMap
tensortype(::Type{T}) where {T<:AbstractTensorMap} = T
tensortype(::Type{<:AbstractTensorLattice{T,L,D}}) where {T,L,D} = D
# D is an AbstractLattice, so recurse through
tensortype(::Type{<:AbstractCompositeLattice{T,L,D}}) where {T,L,D} = tensortype(D)
# # Fallback option
# function tensortype(::Type{<:AbstractLattice})
#     throw(ArgumentError("The lattice provided does not contain tensors at any depth"))
# end
# Value method
tensortype(::L) where {L<:AbstractLattice} = tensortype(L)

TensorKit.spacetype(L::Type{<:AbstractLattice}) = spacetype(tensortype(L))
TensorKit.spacetype(::L) where {L<:AbstractLattice} = spacetype(tensortype(L))

# this function behaves as eltype does for AbstractTensorMap
numbertype(t::Type{<:AbstractTensorMap}) = eltype(t)
numbertype(L::Type{<:AbstractLattice}) = numbertype(eltype(L))
numbertype(::DataType) = ComplexF64
numbertype(l) = numbertype(typeof(l))

# # A lattice always has two dimensions as far as iteration is concerned.
Base.ndims(M::AbstractLattice) = 2

# # Single dimensional indexing
Base.firstindex(M::AbstractLattice) = firstindex(to_matrix(M))
Base.lastindex(M::AbstractLattice) = lastindex(to_matrix(M))
Base.getindex(M::AbstractLattice, i::Int) = getindex(to_matrix(M), i)

function Base.getindex(M::AbstractVectorLattice{T}, i::Int) where {T<:TensorStyle}
    return _getindex(boundarystyle(M), M, i)
end
function _getindex(::Periodic, M::AbstractVectorLattice, i::Int)
    return getindex(to_matrix(M), mod(i, 1:length(M)))
end
function _getindex(::Open, M::AbstractVectorLattice, i::Int)
    return getindex(to_matrix(M), i)
end

to_periodicindex(c::Colon, ::UnitRange) = c
to_periodicindex(i::Int, r::UnitRange) = mod(i, r)

function Base.getindex(M::AbstractLattice, i1, i2)
    return _getindex(boundarystyle(M), M, i1, i2)
end
function _getindex(
    ::Periodic, M::AbstractLattice{T,<:LatticeStyle{Nx,Ny}}, i1, i2
) where {T,Nx,Ny}
    return getindex(to_matrix(M), to_periodicindex(i1, 1:Nx), to_periodicindex(i2, 1:Ny))
end
function _getindex(::Open, M::AbstractLattice, i1, i2)
    return getindex(to_matrix(M), i1, i2)
end

function Base.getindex(M::AbstractLattice, i::CartesianIndex{2})
    return getindex(M, Tuple(i)...)
end

function Base.setindex!(M::AbstractLattice{T,D}, v::D, i::Int) where {T,D}
    to_matrix(M)[i] = v
    return M
end
function Base.setindex!(M::AbstractLattice, v, x::Int, y::Int)
    to_matrix(M)[x, y] = v
    return M
end
function Base.iterate(M::AbstractLattice, state=1)
    l = length(M)
    if state > l
        return nothing
    else
        return M[state], state + 1
    end
end

abstract type AbstractLatticeBroadcast{L<:LatticeStyle} <: Broadcast.AbstractArrayStyle{2} end
struct LatticeBroadcast{T,L,ElType} <: AbstractLatticeBroadcast{L} end
struct LatticeFallback{L} <: AbstractLatticeBroadcast{L} end

LatticeBroadcast{T,L,ElType}(::Val{2}) where {T,L,ElType} = LatticeBroadcast{T,L,ElType}()
LatticeFallback{L}(::Val{2}) where {L} = LatticeFallback{L}()

function Base.BroadcastStyle(::Type{<:AbstractLattice{T,L,D}}) where {T,L,D}
    return LatticeBroadcast{T,L,D}()
end

# If element type matches the element type of broadcast style then assume type of tensor
# has remained unchanged, thus keep the same tensor style. 
function Base.similar(
    bc::Broadcast.Broadcasted{LatticeBroadcast{T,L,ElType}}, ::Type{ElType}
) where {T,L,ElType<:AbstractTensorMap}
    return Lattice(T(), L(), similar(Matrix{ElType}, axes(bc)))
end

# Else, fallback to Generic()
function Base.similar(
    bc::Broadcast.Broadcasted{<:AbstractLatticeBroadcast{L}}, ::Type{ElType}
) where {L,ElType<:AbstractTensorMap}
    return Lattice(Generic(), L(), similar(Matrix{ElType}, axes(bc)))
end

# If elements are not tensors, always use tensor style NotTensor()
function Base.similar(
    bc::Broadcast.Broadcasted{<:AbstractLatticeBroadcast{L}}, ::Type{ElType}
) where {L,ElType}
    return Lattice(NotTensor(), L(), similar(Matrix{ElType}, axes(bc)))
end

# If broadcasting across lattices of tensors, then we do not know the resulting type so
# best we can do is a Generic lattice.
function Base.BroadcastStyle(
    ::LatticeBroadcast{<:PureTensorStyle,L,E1}, ::LatticeBroadcast{<:PureTensorStyle,L,E2}
) where {
    E1<:AbstractTensorMap,E2<:AbstractTensorMap,L
}
    return LatticeFallback{L}()
end
# Lattices of tensors take precedence over lattices of non-tensor objects. 
function Base.BroadcastStyle(
    ::LatticeBroadcast{T1,L,E1}, ::LatticeBroadcast{T2,L,E2}
) where {T1<:PureTensorStyle,T2<:NotTensor,L,E1<:AbstractTensorMap,E2}
    return LatticeBroadcast{T1, L, E1}()
end

function Base.BroadcastStyle(::LatticeBroadcast,::LatticeBroadcast)
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
    l1::AbstractLattice, l2::AbstractLattice, l3::AbstractLattice, args::AbstractLattice...
)
    return hcat(hcat(l1, l2), l3, args...)
end
Base.hcat(a::AbstractLattice) = a

function Base.hcat(a::AbstractLattice{T1}, b::AbstractLattice{T2}) where {T1,T2}
    T = promote_type(T1, T2)
    return _hcat(latticestyle(a), latticestyle(b), T(), a, b)
end
function _hcat(
    ::Infinite{Nx,Ny},
    ::Infinite{Mx,My},
    T::TensorStyle,
    a::AbstractLattice,
    b::AbstractLattice,
) where {Nx,Ny,Mx,My}
    if Nx != Mx
        throw(DimensionMismatch("Unequal dimensions along axis"))
    end
    N = Nx
    L = Infinite{N,Ny + My}()
    return Lattice(T, L, hcat(a.data, b.data))
end
