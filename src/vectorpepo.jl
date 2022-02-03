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
