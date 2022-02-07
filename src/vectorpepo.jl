"""
    struct VectorPEPOTensor{S<:EuclideanSpace} <: AbstractTensorMap{S,3,2}

Represents a vectorised PEPO state. The `tensor` field gives the `TensorMap` representing
the state, and the `unitary` field gives the unitary whos inverse unvectorises the PEPO.
"""
struct VectorPEPOTensor{S<:EuclideanSpace} <: AbstractTensorMap{S,3,2}
    tensor::TensorMap{S,3,2}
    unitary::TensorMap{S,1,2}
    function VectorPEPOTensor(
        tensor::TensorMap{S,3,2}, unitary::TensorMap{S,1,2}
    ) where {S<:EuclideanSpace}
        return new{S}(tensor, unitary)
    end
end

const VPEPOTensor = VectorPEPOTensor
