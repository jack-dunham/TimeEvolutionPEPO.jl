### AbstractPhysicalTensor

abstract type AbstractPhysicalTensor{S<:IndexSpace,T<:TensorMap{S}} end

struct PhysicalTensor{S<:IndexSpace,T<:TensorMap{S}} <: AbstractPhysicalTensor{S,T}
    tensor::T
end

Base.convert(::Type{T}, a::AbstractTensorMap) where {T<:AbstractPhysicalTensor} = T(a)
Base.convert(::Type{TensorMap}, m::AbstractPhysicalTensor) = m.tensor
totensor(m::AbstractPhysicalTensor) = convert(TensorMap, m)

function Base.isapprox(x::P, y::P; kwargs...) where {P<:AbstractPhysicalTensor}
    return isapprox(totensor(x), totensor(y); kwargs...)
end

tensortype(::Type{<:AbstractPhysicalTensor{<:IndexSpace,T}}) where {T<:TensorMap} = T
tensortype(m::AbstractPhysicalTensor) = tensortype(typeof(m))

TensorKit.spacetype(::Type{<:AbstractPhysicalTensor{S}}) where {S<:IndexSpace} = S
TensorKit.spacetype(m::AbstractPhysicalTensor) = spacetype(typeof(m))

function Base.eltype(::Type{<:AbstractPhysicalTensor{<:IndexSpace,T}}) where {T<:TensorMap}
    return eltype(T)
end
Base.eltype(m::AbstractPhysicalTensor) = eltype(typeof(m))

Base.similar(m::AbstractPhysicalTensor) = typeof(m)(similar(totensor(m)))
function Base.similar(m::AbstractPhysicalTensor, T::Type{<:Number})
    return typeof(m)(similar(m.tensor, T))
end

function Base.similar(
    m::AbstractPhysicalTensor,
    ::Type{T},
    P::TensorKit.TensorMapSpace{S}=(domain(m) → codomain(m)),
) where {T,S}
    t = similar(totensor(m), T, P)
    return PhysicalTensor(t)
end

TensorKit.codomain(m::AbstractPhysicalTensor) = codomain(totensor(m))
TensorKit.domain(m::AbstractPhysicalTensor) = domain(totensor(m))

TensorKit.numout(m::AbstractPhysicalTensor) = numout(totensor(m))
TensorKit.numin(m::AbstractPhysicalTensor) = numin(totensor(m))
TensorKit.numind(m::AbstractPhysicalTensor) = numind(totensor(m))

function TensorKit.permute(m::AbstractPhysicalTensor, p1::NTuple, p2::NTuple)
    mt = permute(totensor(m), p1, p2)
    return typeof(m)(mt)
end

LinearAlgebra.norm(m::AbstractPhysicalTensor, p::Real=2) = norm(m.tensor, p)
LinearAlgebra.normalize!(m::AbstractPhysicalTensor, p::Real=2) = rmul!(m, inv(norm(m, p)))
function LinearAlgebra.normalize(m::AbstractPhysicalTensor, p::Real=2)
    return mul!(similar(m), m, inv(norm(m, p)))
end

function Base.:*(α::Number, m::AbstractPhysicalTensor)
    return mul!(similar(m, promote_type(eltype(m), typeof(α))), α, m)
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

### AbstractTensorMatrix 

abstract type AbstractTensorMatrix{Nx,Ny,P<:AbstractPhysicalTensor} end

struct TensorMatrix{
    Nx,Ny,S<:IndexSpace,N₁,N₂,P<:AbstractPhysicalTensor{S,<:TensorMap{S,N₁,N₂}}
} <: AbstractTensorMatrix{Nx,Ny,P}
    matrix::SizedMatrix{Nx,Ny,P}
    cell::UnitCell{Nx,Ny,S}
end

function Base.convert(::Type{Matrix}, M::AbstractTensorMatrix{Nx,Ny,P}) where {Nx,Ny,P}
    return convert(Matrix, M.matrix)::Matrix{P}
end
tomatrix(M::AbstractTensorMatrix) = convert(Matrix, M)
tomatrixt(M::AbstractTensorMatrix) = totensor.(tomatrix(M))

Base.size(M::AbstractTensorMatrix) = size(tomatrix(M))

function TensorMatrix(m::AbstractMatrix, cell::UnitCell{Nx,Ny}) where {Nx,Ny}
    return TensorMatrix(SizedMatrix{Nx,Ny}(m), cell)
end

uclabels(M::AbstractTensorMatrix) = (M.cell).label

ptensortype(::AbstractTensorMatrix{Nx,Ny,P}) where {Nx,Ny,P} = P
TensorKit.spacetype(::AbstractTensorMatrix{Nx,Ny,P}) where {Nx,Ny,P} = spacetype(P)

function Base.eltype(
    ::AbstractTensorMatrix{Nx,Ny,P}
) where {Nx,Ny,P<:AbstractPhysicalTensor}
    return eltype(P)
end

Base.ndims(M::AbstractTensorMatrix) = ndims(M.matrix)

Base.getindex(M::AbstractTensorMatrix, i::Int) = getindex(M.matrix, i)
function Base.getindex(M::AbstractTensorMatrix{Nx,Ny}, i1::Int, i2::Int) where {Nx,Ny}
    return getindex(tomatrix(M), mod(i1, 1:Nx), mod(i2, 1:Ny))
end
function Base.getindex(M::AbstractTensorMatrix{Nx,Ny}, i::CartesianIndex{2}) where {Nx,Ny}
    return getindex(tomatrix(M), Tuple(i)...)
end

function Base.getindex(M::AbstractTensorMatrix, s::String)
    inds = findall(x -> x == s, uclabels(M))
    if length(inds) == 0
        throw(IndexError("No tensor associated with label: $s"))
    end
    rv = Dict{CartesianIndex{2},tensortype(M)}()
    for i in inds
        rv[i] = getindex(M, i)
    end
    return rv
end

function Base.getindex(M::AbstractTensorMatrix, i1::Int, ::Colon)
    Nx, _ = size(M)
    return tomatrix(M)[mod(i1, 1:Nx), :]
end
function Base.getindex(M::AbstractTensorMatrix, ::Colon, i2::Int)
    Nx, _ = size(M)
    return tomatrix(M)[:, mod(i2, 1:Nx)]
end

function Base.setindex!(
    M::AbstractTensorMatrix{Nx,Ny,P}, v::P, x::Int, y::Int
) where {Nx,Ny,P<:AbstractPhysicalTensor}
    return M.matrix[x, y] = v
end

unitcell(M::AbstractTensorMatrix) = M.cell

function tensorise(p::AbstractTensorMatrix)
    pm = tomatrixt(p)
    pt = _tensorize(pm)
    return pt
end
