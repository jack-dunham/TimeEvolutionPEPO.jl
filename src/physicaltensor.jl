### AbstractPhysicalTensor

@doc raw"""
    AbstractPhysicalTensor{S<:IndexSpace, N₁, N₂, ...} <: AbstractTensorMap{S,N₁, N₂}

Abstract supertype of all physical tensor maps over vectors spaces of type `S<:IndexSpace`.
"""
abstract type AbstractPhysicalTensor{S<:IndexSpace,N₁,N₂,T<:AbstractTensorMap{S,N₁,N₂}} <:
              AbstractTensorMap{S,N₁,N₂} end

@doc raw"""
    PhysicalTensor{S<:IndexSpace, N₁, N₂, ...} <: AbstractPhysicalTensor{S, N₁, N₂}

Wrapper of `<:AbstractTensorMap` types as a concrete subtype of 
[`AbstractPhysicalTensor`](@ref). 

# Fields
- `tensor::AbstractTensorMap{S, N₁, N₂}`: the wrapped tensor
"""
struct PhysicalTensor{S,N₁,N₂,T<:AbstractTensorMap{S,N₁,N₂}} <:
       AbstractPhysicalTensor{S,N₁,N₂,T}
    tensor::T
end

function (P)(t::AbstractTensorMap{S,N₁,N₂}) where {S,N₁,N₂}
    return P{S,N₁,N₂,typeof(t)}(t)
end

Base.convert(::Type{T}, a::AbstractTensorMap) where {T<:AbstractPhysicalTensor} = T(a)
Base.convert(::Type{AbstractTensorMap}, m::AbstractPhysicalTensor) = m.tensor
Base.convert(T::Type{<:TensorMap}, m::AbstractPhysicalTensor) = convert(T, m.tensor)

totensor(m::AbstractPhysicalTensor) = convert(AbstractTensorMap, m)

# approx comparison of two pysical tensors
function Base.isapprox(x::P, y::P; kwargs...) where {P<:AbstractPhysicalTensor}
    return isapprox(totensor(x), totensor(y); kwargs...)
end

@doc raw"""
    tensortype(T::Type{<:AbstractPhysicalTensor})
    tensortype(p::AbstractPhysicalTensor)

Returns the type of the tensor associated with a [`AbstractPhysicalTensor`](@ref) object or 
data type.
"""
tensortype(::Type{<:AbstractPhysicalTensor{<:IndexSpace,N₁,N₂,T}}) where {N₁,N₂,T} = T
tensortype(p::AbstractPhysicalTensor) = tensortype(typeof(p))

@doc raw"""
    spacetype(T::Type{<:AbstractPhysicalTensor})
    spacetype(p::AbstractPhysicalTensor)

Returns the type of the vector space `S<:IndexSpace` associated with a
[`AbstractPhysicalTensor{S}`](@ref) object or type. 
"""
TensorKit.spacetype(::Type{<:AbstractPhysicalTensor{S}}) where {S<:IndexSpace} = S
TensorKit.spacetype(p::AbstractPhysicalTensor) = spacetype(typeof(p))

function Base.eltype(::Type{<:AbstractPhysicalTensor{<:IndexSpace,N₁,N₂,T}}) where {N₁,N₂,T}
    return eltype(T)
end
Base.eltype(p::AbstractPhysicalTensor) = eltype(typeof(p))

Base.similar(p::AbstractPhysicalTensor) = typeof(p)(similar(totensor(p)))

function Base.similar(p::AbstractPhysicalTensor, T::Type{<:Number})
    return (Base.typename(typeof(p)).wrapper)(similar(totensor(p), T))
end

function Base.similar(
    p::AbstractPhysicalTensor,
    ::Type{T},
    P::TensorKit.TensorMapSpace{S}=(domain(p) → codomain(p)),
) where {T,S}
    t = similar(totensor(p), T, P)
    return PhysicalTensor(t)
end

TensorKit.blocksectors(p::AbstractPhysicalTensor) = blocksectors(p.tensor)
TensorKit.block(p::AbstractPhysicalTensor, t) = block(p.tensor, t)

TensorKit.codomain(p::AbstractPhysicalTensor) = codomain(totensor(p))
TensorKit.domain(p::AbstractPhysicalTensor) = domain(totensor(p))

TensorKit.numout(p::AbstractPhysicalTensor) = numout(totensor(p))
TensorKit.numin(p::AbstractPhysicalTensor) = numin(totensor(p))
TensorKit.numind(p::AbstractPhysicalTensor) = numind(totensor(p))

function TensorKit.permute(pt::AbstractPhysicalTensor, p1::NTuple, p2::NTuple)
    ptp = permute(totensor(pt), p1, p2)
    return typeof(pt)(ptp)
end

TensorKit.exp!(p::AbstractPhysicalTensor) = exp!(p.tensor)

LinearAlgebra.norm(pt::AbstractPhysicalTensor, p::Real=2) = norm(pt.tensor, p)
function LinearAlgebra.normalize!(pt::AbstractPhysicalTensor, p::Real=2)
    return rmul!(pt, inv(norm(pt, p)))
end
function LinearAlgebra.normalize(pt::AbstractPhysicalTensor, p::Real=2)
    return mul!(similar(pt), pt, inv(norm(pt, p)))
end

function Base.:*(α::Number, pt::AbstractPhysicalTensor)
    return mul!(similar(pt, promote_type(eltype(pt), typeof(α))), α, pt)
end

#TODO: Move to linalg.jl
LinearAlgebra.rmul!(m::AbstractPhysicalTensor, α::Number) = mul!(m, m, α)
LinearAlgebra.lmul!(α::Number, m::AbstractPhysicalTensor) = mul!(m, α, m)

function LinearAlgebra.mul!(
    m1::AbstractPhysicalTensor, m2::AbstractPhysicalTensor, α::Number
)
    mul!(m1.tensor, m2.tensor, α)
    return m1
end
function LinearAlgebra.mul!(
    m1::AbstractPhysicalTensor, α::Number, m2::AbstractPhysicalTensor
)
    mul!(m1.tensor, α, m2.tensor)
    return m1
end

function LinearAlgebra.axpy!(
    α::Number, m1::AbstractPhysicalTensor, m2::AbstractPhysicalTensor
)
    axpy!(α, m1.tensor, m2.tensor)
    return m2
end
function LinearAlgebra.axpby!(
    α::Number, m1::AbstractPhysicalTensor, β::Number, m2::AbstractPhysicalTensor
)
    axpby!(α, m1.tensor, β, m2.tensor)
    return m2
end

function LinearAlgebra.dot(m1::AbstractPhysicalTensor, m2::AbstractPhysicalTensor)
    return dot(m1.tensor, m2.tensor)
end

###TODO: move these somewhere else 
###Multiplcation of tensors
function _lmul(
    C::AbstractTensorMap{S,1,1}, A::AbstractTensorMap{S,2,1}
) where {S<:IndexSpace}
    return @tensor B[1 2; 3] := C[2; a] * A[1 a; 3]
end
function _lmul(
    C::AbstractTensorMap{S,1,1}, A::AbstractTensorMap{S,2,2}
) where {S<:IndexSpace}
    return @tensor B[1 2; 3 4] := C[2; a] * A[1 a; 3 4]
end

function _rmul(
    A::AbstractTensorMap{S,2,1}, C::AbstractTensorMap{S,1,1}
) where {S<:IndexSpace}
    return @tensor B[1 2; 3] := A[1 2; a] * C[a; 3]
end
function _rmul(
    A::AbstractTensorMap{S,2,2}, C::AbstractTensorMap{S,1,1}
) where {S<:IndexSpace}
    return @tensor B[1 2; 3 4] := A[1 2; 3 a] * C[a; 4]
end

### AbstractTensorLattice

abstract type LatticeBoundaryStyle end
struct Periodic <: LatticeBoundaryStyle end
struct Open <: LatticeBoundaryStyle end

isperiodic(::Periodic) = true
isperiodic(::Open) = false
isperiodic(a) = LatticeBoundaryStyle(a)

abstract type LatticeSize{Nx,Ny} end
const RowLatticeSize{Nx} = LatticeSize{Nx,1}
const ColLatticeSize{Ny} = LatticeSize{1,Ny}
struct Finite{Nx,Ny,Boundary<:LatticeBoundaryStyle} <: LatticeSize{Nx,Ny} end
struct Infinite{Nx,Ny} <: LatticeSize{Nx,Ny} end

LatticeBoundaryStyle(::Type{<:Finite{Nx,Ny,B}}) where {Nx,Ny,B} = B()
LatticeBoundaryStyle(::Type{<:Infinite}) = Periodic()

Base.size(::Type{<:LatticeSize{Nx,Ny}}) where {Nx,Ny} = (Nx, Ny)
Base.size(::L) where {L<:LatticeSize} = size(typeof(L))

Base.permutedims(::Type{<:Finite{Nx,Ny,L}}) where {Nx,Ny,L} = Finite{Ny,Nx,L}
Base.permutedims(::Type{<:Infinite{Nx,Ny}}) where {Nx,Ny} = Infinite{Ny,Nx}

Base.permutedims(L::LatticeSize) = permutedims(typeof(L))()

@doc raw"""
    AbstractTensorMatrix{Nx,Ny,P<:AbstractPhysicalTensor}    

Abstract supertype denoting a `Nx` by `Ny` lattice of physical tensors of type P.
"""
abstract type AbstractLattice{Size<:LatticeSize,P} <: AbstractMatrix{P} end
const AbstractTensorLattice{Size,P<:AbstractPhysicalTensor} = AbstractLattice{Size,P}
const AbstractCompositeLattice{Size,P<:AbstractLattice} = AbstractLattice{Size,P}

@doc raw"""

    TensorLattice{L<:LatticeSize, P<:AbstractPhysicalTensor} <: AbstractTensorLattice

Concrete type denoting a lattice with size `LatticeSize` of physical tensors of type `P`. 

# Fields
- `matrix::Matrix{P}`: the static-sized matrix of physical tensors
"""
struct Lattice{L<:LatticeSize,P} <: AbstractLattice{L,P}
    data::Matrix{P}
    function Lattice{L}(data::Matrix{P}) where {L,P}
        if size(L) != size(data)
            throw(DimensionMismatch(""))
        else
            new{L,P}(data)
        end
    end
end

Base.similar(l::Lattice{L}) where {L<:LatticeSize} = Lattice{L}(similar(l.data))

const TensorLattice{L,P<:AbstractPhysicalTensor} = Lattice{L,P}
const CompositeLattice{L,P<:AbstractLattice} = Lattice{L,P}

function Base.permutedims(l::AbstractLattice{L}) where {L<:LatticeSize}
    return Lattice{permutedims(L)}(permutedims(l.data))
end

Base.size(::Type{<:AbstractLattice{L}}) where {L} = size(L)
Base.size(m::AbstractLattice) = size(typeof(m))

Base.getindex(m::AbstractLattice, inds...) = getindex(m.data, inds...)

LatticeSize(::Type{<:AbstractLattice{L}}) where {L} = L()
LatticeSize(m::AbstractLattice) = LatticeSize(typeof(m))
function LatticeBoundaryStyle(::AbstractLattice{L}) where {L}
    return LatticeBoundaryStyle(LatticeSize(L))
end

to_matrix(M::AbstractLattice) = M.data
to_matrixt(M::AbstractLattice) = totensor.(to_matrix(M))

ptensortype(::AbstractTensorLattice{L,P}) where {L,P} = P
ptensortype(::AbstractCompositeLattice{L,P}) where {L,P} = ptensortype(P)
TensorKit.spacetype(::AbstractLattice{L,P}) where {L,P} = spacetype(P)

Base.eltype(::AbstractLattice{L,P}) where {L,P} = eltype(P)

Base.ndims(M::AbstractLattice) = ndims(to_matrix(M))

#TODO: move to utils
to_periodicindex(c::Colon, ::UnitRange) = c
to_periodicindex(i::Int, r::UnitRange) = mod(i, r)
# 
function Base.getindex(M::AbstractLattice{Nx,Ny}, i1, i2) where {Nx,Ny}
    return _getindex(LatticeBoundaryStyle(M), M, i1, i2)
end

function _getindex(::Periodic, M::AbstractLattice{Nx,Ny}, i1, i2) where {Nx,Ny}
    return getindex(to_matrix(M), to_periodicindex(i1, 1:Nx), to_periodicindex(i2, 1:Ny))
end
function _getindex(::Open, M::AbstractLattice{Nx,Ny}, i1, i2) where {Nx,Ny}
    return getindex(to_matrix(M), i1, i2)
end

Base.getindex(M::AbstractLattice, i::Int) = getindex(to_matrix(M), i)

function Base.getindex(M::AbstractLattice, i::CartesianIndex{2})
    return getindex(M, Tuple(i)...)
end

function Base.setindex!(M::AbstractLattice{L,P}, v::P, x::Int, y::Int) where {L,P}
    return M.data[x, y] = v
end

function tensorise(p::AbstractLattice)
    pm = to_matrixt(p)
    pt = _tensorize(pm)
    return pt
end
