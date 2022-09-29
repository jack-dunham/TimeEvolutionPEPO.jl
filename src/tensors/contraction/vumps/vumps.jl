struct VUMPS <:AbstractBoundaryAlgorithm
    verbosity::Int
    maxiter::Int
    tol_vumps::Float64
    tol_gauge::Float64
end

VUMPS() = VUMPS(1, 100, 1e-10, 1e-10)

include("alg.jl")
