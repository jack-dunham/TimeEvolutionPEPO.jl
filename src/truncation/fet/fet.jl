struct FET{A<:AbstractBoundaryAlgorithm} <: AbstractTruncationAlgorithm
    alg::A
    gauge::Int # 0 -> no gauge fixing, 1 ->  gauge fixing, no testing 
end

FET() = FET(VUMPS(),2)

include("bondenv.jl")
include("isometries.jl")
include("wtg.jl")
include("alg.jl")
