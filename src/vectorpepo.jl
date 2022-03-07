"""
    struct VectorPEPOTensor{S<:EuclideanSpace} <: AbstractTensorMap{S,3,2}

Represents a vectorised PEPO state. The `tensor` field gives the `TensorMap` representing
the state, and the `unitary` field gives the unitary who's inverse unvectorises the PEPO.
"""
struct VectorPEPOTensor{
    S<:EuclideanSpace,
    I<:Sector,
    A<:Union{<:DenseMatrix,TensorKit.SectorDict{I,<:DenseMatrix}},
    F₁,
    F₂,
} <: AbstractPhysicalTensor{S,3,2, I, A, F₁, F₂}
    tensor::TensorMap{S,3,2,I,A,F₁,F₂}
    unitary #::TensorMap{S,1,2,I,A}
    # function VectorPEPOTensor(
    #     tensor::TensorMap{S,3,2,I,A,F₁,F₂}, unitary::TensorMap{S,1,2,I,A}
    # ) where {
    #     S<:EuclideanSpace,
    #     I<:Sector,
    #     A<:Union{<:DenseMatrix,TensorKit.SectorDict{I,<:DenseMatrix}},
    #     F₁,
    #     F₂,
    # }
    #     return new{S,I,A,F₁,F₂}(tensor, unitary)
    # end
end

const VPEPOTensor = VectorPEPOTensor
