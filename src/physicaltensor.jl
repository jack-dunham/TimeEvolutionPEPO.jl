abstract type AbstractPhysicalTensor{
    S<:IndexSpace,
    N₁,
    N₂,
    I<:Sector,
    A<:Union{<:DenseMatrix,TensorKit.SectorDict{I,<:DenseMatrix}},
    F₁,
    F₂,
} <: AbstractTensorMap{S,N₁,N₂} end
