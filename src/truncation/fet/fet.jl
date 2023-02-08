struct FET{Alg<:AbstractBoundary} <: AbstractTruncationAlgorithm
    boundary::BoundaryParameters
    gauge::Int # 0 -> no gauge fixing, 1 ->  gauge fixing, no testing 2 -> with testing
    ξ::Float64 # correlation length
    recycle::Bool
end

# DEPREC:
function initialise_state(
    ::Type{FET{Alg}}, param::BoundaryParameters; ξ=Inf, gauge=2, recycle=true
) where {Alg}
    return FET(boundary, gauge, ξ, recycle)
end

# Default options
function FET{Alg}(
    param::BoundaryParameters; gauge::Int=2, ξ::Float64=Inf, recycle::Bool=true
) where {Alg}
    return FET{Alg}(param, gauge, ξ, recycle)
end

# Default options, with default boundary options. Specify the algorithm and bond space only.
FET{Alg}(bondspace::IndexSpace) where {Alg} = FET{Alg}(BoundaryParameters(bondspace))

include("bondenv.jl")
include("isometries.jl")
include("wtg.jl")
include("alg.jl")
