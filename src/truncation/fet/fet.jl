struct FET{Alg<:AbstractBoundary} <: AbstractTruncationAlgorithm
    boundary::BoundaryParameters
    gauge::Int # 0 -> no gauge fixing, 1 ->  gauge fixing, no testing 2 -> with testing
    ξ::Float64 # correlation length
    recycle::Bool
end

function initialise_state(
    ::Type{FET{Alg}}, param::BoundaryParameters; ξ=Inf, gauge=2, recycle=true
) where {Alg}
    return FET(boundary, gauge, ξ, recycle)
end

include("bondenv.jl")
include("isometries.jl")
include("wtg.jl")
include("alg.jl")
