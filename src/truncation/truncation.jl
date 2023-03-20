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

# UPDATE SCHEMES

abstract type AbstractUpdateScheme end

@with_kw struct LULT <: AbstractUpdateScheme
    optscheme::AbstractOptimisationScheme = TensorOptimisation()
end

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

include("fet/fet.jl")
include("clu/clu.jl")
