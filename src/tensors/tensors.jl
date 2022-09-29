const Cotensor{S<:IndexSpace,N,I<:Sector,A,F₁,F₂} = TensorMap{S,0,N,I,A,F₁,F₂}

function Cotensor(dataorf, T::Type{<:Number}, P::TensorSpace{S}) where {S<:IndexSpace}
    return TensorMap(dataorf, T, one(P), P)
end

Cotensor(dataorf, P::TensorSpace{S}) where {S<:IndexSpace} = TensorMap(dataorf, one(P), P)

function Cotensor(T::Type{<:Number}, P::TensorSpace{S}) where {S<:IndexSpace}
    return TensorMap(T, one(P), P)
end

Cotensor(P::TensorSpace{S}) where {S<:IndexSpace} = TensorMap(one(P), P)

include("contraction/contraction.jl")
