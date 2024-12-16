using TimeEvolutionPEPO
using TensorKit
using KrylovKit
using LinearAlgebra
using Test
using Combinatorics

@testset verbose = true "All tests" begin
    if isempty(ARGS)
        include("model.jl")
        include("truncation.jl")
    else
        for file in ARGS
            include("$(file).jl")
        end
    end
end
