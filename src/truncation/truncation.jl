# Either a AbstractBoundary or an AbstractNeighbourhood
struct SimpleUpdateState end

const AbstractEnvironmentState = Union{
    NeighbourhoodState,SimpleUpdateState,AbstractBoundaryState
}

struct Environments{T<:AbstractEnvironmentState}
    x::T
    y::T
end

struct VidalIsometries{T1<:AbsTen{2,1},T2<:AbsTen{1,2}}
    east::T1
    south::T1
    west::T2
    north::T2
end

function setvidal!(vidals::Lattice{G,<:VidalIsometries},u,v, bond) where {G}
    ax = bondaxis(bond)

    src_iso, snk_iso = vidals[bond]
    
    if ax == 1
        copy!(src_iso.east, u)
        copy!(snk_iso.west, v)
    elseif ax == 2
        copy!(src_iso.south, u)
        copy!(snk_iso.north, v)
    end

    return vidals
end


function calculate!(env::Environments)
    ITC.reset!(env.x)
    ITC.reset!(env.y)
    ITC.calculate!(env.x)
    ITC.calculate!(env.y)
    return env
end

# SU, FET, NTU
abstract type AbstractTruncationScheme end

# TRUNCATION SCHEMES
@kwdef struct FET{BAlg<:AbstractBoundaryAlgorithm,UScheme<:AbstractUpdateScheme} <:
              AbstractTruncationScheme
    gauge::Bool = true
    ξ::Float64 = Inf
    alg::BAlg
    updatescheme::UScheme = LULT()
end

@kwdef struct NTU{UScheme<:AbstractUpdateScheme} <: AbstractTruncationScheme
    gauge::Bool = false
    ξ::Float64 = Inf
    updatescheme::UScheme = LULT()
end

@kwdef struct SU <: AbstractTruncationScheme
    gauge::Bool = true
    ξ::Float64 = Inf
end

updatescheme(sch::AbstractTruncationScheme) = sch.updatescheme
updatescheme(::SU) = LULT()

# function Base.show(io::IO, fet::FET)
#     for field in fieldnames(typeof(fet))
#         dump(getfield(fet, field))
#     end
# end

Base.show(::IO, tr::AbstractTruncationScheme) = dump(tr)

function environments(pepo::TimeDependantPEPO, truncalg::AbstractTruncationScheme)
    xenv = initenvironment(pepo.x, truncalg)
    yenv = initenvironment(pepo.y, truncalg)
    return Environments(xenv, yenv)
end

function environments!(env::Environments)
    environment!(env.x)
    environment!(env.y)
    return env
end

function environment!(env::AbstractBoundaryState)
    ITC.reset!(env)
    ITC.calculate!(env)
    return env
end
environment!(env::NeighbourhoodState) = ITC.calculate!(env)
environment!(env::SimpleUpdateState) = env

function initenvironment(pepo::AbstractPEPO, fet::FET)
    return ITC.initialize(hilbertschmidt(pepo), fet.alg)
end
function initenvironment(pepo::AbstractPEPO, ::NTU; kwargs...)
    return ntu_initenvironment(hilbertschmidt(pepo); kwargs...)
end
initenvironment(::AbstractPEPO, ::SU) = SimpleUpdateState()

function renormalize(tensors, env::NeighbourhoodState)
    new_env = ntu_initenvironment(tensors)
    ITC.calculate!(new_env)
    return new_env
end
renormalize(tensors, env::SimpleUpdateState) = env

"""
    metric(pepo::LocallyEvolvingPEPO, environment::AbstractEnvironment)
    metric(pepo::GloballyEvolvingPEPO, environment::AbstractEnvironment)

Construct the truncation metric calculated from `environmnet`. The first method returns the metric corresponding to 
truncation around `currentbond(pepo)`. The second method returns metrics for all bonds on the lattice.

"""
function metric(pepo::LocalUpdatablePEPO, environment)
    return metric(pepo, environment, currentbond(pepo))
end

function metric(pepo::Union{AbstractPEPO,AbstractLattice}, environment, bond::Bond)
    left, right = pepo[bond]
    return _metric(left, right, environment, bond)
end

"""
    truncate!(pepo::AbstractUpdatablePEPO, iso::TruncationIsometries)

Truncate the state `pepo` according to truncation matrices `iso`.
"""
function truncate!(pepo::AbstractUpdatablePEPO, iso::TruncationIsometries)
    @debug "Before truncation:" isevolved(pepo)

    _truncate!(pepo, iso)
    setisevolved!(pepo, false)

    @debug "After truncation:" isevolved(pepo)

    return pepo
end

# Truncate a locally evolving pepo
function _truncate!(pepo::LocalUpdatablePEPO, iso::TruncationIsometries)
    bond = currentbond(pepo)

    @debug "Truncating bond: $bond"

    new_left, new_right = pepo[bond]
    old_left, old_right = getstate(pepo)[bond]

    # Truncate the left site with an isometry on the right bond
    _truncate!(old_left, new_left, iso.u * sqrt(iso.s))
    # Truncate the right site with an isometry on the left bond
    _truncate!(old_right, sqrt(iso.s) * iso.v, new_right)

    setweights!(pepo, normalize(iso.s), bond)

    println(normalize(iso.s))

    return pepo
end

partialtruncate(site::AbsTen, iso) = _partialtruncate(site, iso.u * sqrt(iso.s))
partialtruncate(iso, site::AbsTen) = _partialtruncate(sqrt(iso.s) * iso.v, site)

function _partialtruncate(arg1, arg2)
    site_pt = similar_but_truncated(arg1, arg2)
    @debug "" space(site_pt)
    _truncate!(site_pt, arg1, arg2)
    return site_pt
end

function _similar_but_truncated(pepo_tensor, D, ind)
    dom = [domain(pepo_tensor)...]
    dom[ind] = D
    rv = similar(pepo_tensor, codomain(pepo_tensor), *(dom...))
    return rv
end
function similar_but_truncated(pepo_tensor, iso::AbsTen{1,1})
    return _similar_but_truncated(pepo_tensor, domain(iso), 1)
end
function similar_but_truncated(iso::AbsTen{1,1}, pepo_tensor)
    return _similar_but_truncated(pepo_tensor, codomain(iso)', 3)
end

# Truncate a globally evolving pepo
function _truncate!(
    pepo::GlobalUpdatablePEPO, iso::Lattice{G,<:TruncationIsometries}
) where {G}
    # Absorb weights into the isometries `u`.
    u = iso.u .* sqrt.(iso.s)

    # Do the same, but let x -> x - 1 for the left isometries `v`.
    v = circshift(sqrt.(iso.s), (-1, 0)) .* circshift(iso.v, (-1, 0))

    # Truncate all bonds simultaneously.
    _truncate!.(getstate(pepo), getevolved(pepo), u, v)

    # setweight!

    return pepo
end
function _truncate!(pepo::GlobalUpdatablePEPO, iso::TruncationIsometries)
    @assert size(getstate(pepo)) == (1, 1)

    # Absorb weights into the isometries `u`.
    u = iso.u * sqrt(iso.s)

    v = sqrt(iso.s) * iso.v

    # Truncate all bonds simultaneously.

    _truncate!.(getstate(pepo), getevolved(pepo), Ref(u), Ref(v))

    return pepo
end

function vidaltruncation!(pepo, tno, vidals)
    _vidaltruncation!.(pepo.state, pepo.evolving_state, tno, vidals)
    copy!(pepo.evolving_state, pepo.state)
    return pepo
end
function _vidaltruncation!(pepo_dst, pepo_src, tno, vidal)
    _truncate!(pepo_dst, pepo_src, tno, vidal.east, vidal.west, vidal.south,vidal.north)
    return pepo_dst
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
function _truncate!(
    p::AbsTen{2,4},     # dst
    q::AbsTen{2,4},     # src
    l::AbsTen{4,4},     # time-stepper
    u1::AbsTen{2,1},    # east
    v1::AbsTen{1,2},    # west
    u2::AbsTen{2,1},    # south
    v2::AbsTen{1,2},    # north
)
    @tensoropt p[k b; e s w n] =
        v1[w; ww1 ww2] *
        v2[n; nn1 nn2] *
        q[kk bb; ee ss ww nn] *
        l[k b kk bb; ee2 ss2 ww2 nn2] *
        u1[ee1 ee2; e] *
        u2[ss1 ss2; s]
    return p
end
