struct BoundaryInfMPS{S<:IndexSpace,N₁,N₂} <: AbstractInfMPS
    AL::TensorMap{S,N₁,N₂}
    C::TensorMap{S,1,1}
    AR::TensorMap{S,N₁,N₂}
end

const TraceBoundaryInfMPS{S<:IndexSpace} = BoundaryInfMPS{S,2,1}
const HSBoundaryInfMPS{S<:IndexSpace} = BoundaryInfMPS{S,2,2}
