# OPTIMISATION SCHEMES
abstract type AbstractOptimisationScheme end

@kwdef struct IsometryOptimisation <: AbstractOptimisationScheme
    maxiter::Int = 100
    tol::Float64 = 1e-10
end

@kwdef struct TensorOptimisation <: AbstractOptimisationScheme
    maxiter::Int = 100
    tol::Float64 = 1e-10
    EAT::Integer = 0
end

struct TruncationIsometries{T<:AbsTen{1,1}}
    u::T
    s::T
    v::T
end

# UPDATE SCHeMES

abstract type AbstractUpdateScheme end

# LULT: Update according to a metric generated from the state pre-evolution
@kwdef struct LULT <: AbstractUpdateScheme
    optscheme::AbstractOptimisationScheme = TensorOptimisation()
end

# GULT: Update according to a metric generated from the state post-evolution.
@kwdef struct GULT <: AbstractUpdateScheme
    optscheme::IsometryOptimisation = IsometryOptimisation()
end
