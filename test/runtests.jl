using PEPOKit
using TensorKit
using KrylovKit
using LinearAlgebra
using Test
using Combinatorics

@testset verbose = true "All tests" begin
    for file in ARGS
        include("$(file).jl")
    end
end
