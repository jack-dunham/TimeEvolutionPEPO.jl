# Either a AbstractBoundary or an AbstractNeighbourhood
abstract type AbstractEnvironment end

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

# GULT: Update according to a metric generated from the state post-evolution.
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

@with_kw struct NTU{UScheme<:AbstractUpdateScheme} <: AbstractTruncationScheme
    gauge::Bool = true
    ξ::Float64 = Inf
    updatescheme::UScheme = LULT()
end

@with_kw struct SU <: AbstractTruncationScheme
    gauge::Bool = true
    ξ::Float64 = Inf
end

struct SimpleUpdateState end

initenvironment(pepo, fet::FET) = initenvironment(HilbertSchmidt(pepo), fet.alg)
function initenvironment(pepo, ::NTU; kwargs...)
    return ntu_initenvironment(HilbertSchmidt(pepo); kwargs...)
end
initenvironment(pepo, ::SU) = SimpleUpdateState()

@doc raw"""
    metric(pepo::LocallyEvolvingPEPO, environment::AbstractEnvironment)
    metric(pepo::GloballyEvolvingPEPO, environment::AbstractEnvironment)

Construct tensors corresponding to the truncation metric from `environmnet`. The first method returns the metric corresponding to 
truncation around `currentbond(pepo)`. The second method returns metrics for all bonds on the lattice.

"""
function metric(pepo::AbstractEnvolvingPEPO, environment::AbstractEnvironment) end

function truncate!(pepo::AbstractEvolvingPEPO, iso::TruncationIsometries)
    truncate!(pepo, iso)
    setisevolved!(pepo, false)
    return pepo
end

# Truncate a locally evolving pepo
function truncate!(pepo::LocallyEvolvingPEPO, iso::TruncationIsometries)
    bond = currentbond(pepo)

    @debug "Truncating bond: $bond"

    new_left, new_right = pepo[bond]
    old_left, old_right = getstate(pepo)[bond]

    # Truncate the left site with an isometry on the right bond
    _truncate!(old_left, new_left, iso.u * sqrt(iso.s))
    # Truncate the right site with an isometry on the left bond
    _truncate!(old_right, sqrt(iso.s) * iso.v, new_right)

    setweights!(pepo, iso.s, bond)

    return pepo
end

# Truncate a globally evolving pepo
function truncate!(pepo::GloballyEvolvingPEPO, iso::TruncationIsometries)
    # Absorb weights into the isometries `u`.
    u = iso.u .* sqrt.(iso.s)

    # Do the same, but let x -> x - 1 for the left isometries `v`.
    v = circshift(sqrt.(iso.s), (-1, 0)) .* circshift(iso.v, (-1, 0))

    # Truncate all bonds simultaneously.
    _truncate!.(getstate(pepo), getevolved(pepo), u, v)

    # setweight!

    return pepo
end

# Truncation macros.

function _truncate!(p::AbsTen{2,4}, q::AbsTen{2,4}, u::AbsTen{1,1})
    @tensoropt p[k b; e s w n] = q[k b; ee s w n] * u[ee; e]
    return p
end
function _truncate!(p::AbsTen{2,4}, v::AbsTen{1,1}, q::AbsTen{2,4})
    @tensoropt p[k b; e s w n] = v[w; ww] * q[k b; e s ww n]
    return p
end
function _truncate!(p::AbsTen{2,4}, q::AbsTen{2,4}, u::AbsTen{1,1}, v::AbsTen{1,1})
    @tensoropt p[k b; e s w n] = v[w; ww] * q[k b; ee s ww n] * u[ee; e]
    return p
end
