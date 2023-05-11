# SU, FET, NTU
abstract type AbstractTruncationScheme end

# OPTIMISATION SCHEMES

abstract type AbstractOptimisationScheme end

@with_kw struct IsometryOptimisation <: AbstractOptimisationScheme
    maxiter::Int = 100
    tol::Float64 = 1e-10
end

@with_kw struct TensorOptimisation <: AbstractOptimisationScheme
    maxiter::Int = 100
    tol::Float64 = 1e-10
    EAT::Integer = 0
end

struct TruncationIsometries{T<:AbsTen{1,1}}
    u::T
    s::T
    v::T
end

# UPDATE SCHEMES

abstract type AbstractUpdateScheme end

# LULT: Update according to a metric generated from the state pre-evolution
@with_kw struct LULT <: AbstractUpdateScheme
    optscheme::AbstractOptimisationScheme = TensorOptimisation()
end

# LULT: Update according to a metric generated from the state post-evolution.
@with_kw struct GULT <: AbstractUpdateScheme
    optscheme::IsometryOptimisation = IsometryOptimisation()
end

# TRUNCATION SCHEMES

@with_kw struct FET{BAlg<:AbstractBoundaryAlgorithm,UScheme<:AbstractUpdateScheme} <:
                AbstractTruncationScheme
    gauge::Bool = true
    ξ::Float64 = Inf
    boundaryalg::BAlg
    updatescheme::UScheme = LULT()
end

@with_kw struct NTU{UAlg<:AbstractUpdateScheme} <: AbstractTruncationScheme
    gauge::Bool = true
    ξ::Float64 = Inf
    updatescheme::UAlg = LULT()
end

@with_kw struct SU <: AbstractTruncationScheme
    gauge::Bool = true
    ξ::Float64 = Inf
end
