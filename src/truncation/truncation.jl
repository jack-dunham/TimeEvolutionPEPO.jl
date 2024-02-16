struct TruncationIsometries{T<:AbsTen{1,1}}
    u::T
    s::T
    v::T
end

function TruncationIsometries(pepo, u, s, v)
    sout = similar(s, scalartype(pepo))
    copy!(sout, s)
    return TruncationIsometries(u, sout, v)
end

abstract type AbstractTruncatorState end

# Most basic algorithm for truncating bonds
struct SimpleUpdateState <: AbstractTruncatorState end

# Nearest neighbourhood update
struct NeighbourhoodUpdateState{N<:NTUState} <: AbstractTruncatorState
    x::N
    y::N
end

# Full enviornment truncation
struct FullEnvironmentState{E<:InfiniteContraction} <: AbstractTruncatorState
    x::E
    y::E
end


# SU, FET, NTU
abstract type AbstractTruncationScheme end

function Base.show(io::IO, m::MIME"text/plain", sim::AbstractTruncationScheme)
    shownested(io, m, sim)
    return nothing
end

# TRUNCATION SCHEMES
@kwdef struct FET{BAlg<:AbstractBoundaryAlgorithm} <: AbstractTruncationScheme
    gauge::Bool = true
    ξ::Float64 = Inf
    rankreduce = true
    alg::BAlg
end

showheader(::Type{<:FET}) = "Full environment truncation (FET)"

@kwdef struct NTU <: AbstractTruncationScheme
    gauge::Bool = false
    ξ::Float64 = Inf
    rankreduce = true
end

showheader(::Type{<:NTU}) = "Neighbourhood tensor update (NTU)"

@kwdef struct SU <: AbstractTruncationScheme
    ξ::Float64 = Inf
end

showheader(::Type{<:SU}) = "Simple update"

@kwdef struct VGU <: AbstractTruncationScheme
    ξ::Float64 = Inf
end

## Initalialize truncators for a full TD-PEPO
function newtruncators(
    pepo::NTuple{2,P}, ::NTuple{2,L}, truncalg::Union{FET,NTU}
) where {P<:AbstractPEPO,L}
    xenv = newtruncators(pepo[1], truncalg)
    yenv = newtruncators(pepo[2], truncalg)
    return xenv, yenv
end
function newtruncators(::NTuple{2,P}, ::NTuple{2,L}, truncalg::SU) where {P<:AbstractPEPO,L}
    return SimpleUpdateState(), SimpleUpdateState()
end

## 

function ITC.newcontraction(conalg, net::HilbertSchmidt; kwargs...)
    return newcontraction(conalg, net.data; kwargs...)
end

## Initalialize truncators for lattice of tensors
newtruncators(bulk, fet::FET) = newtruncators(bulk, fet.alg)
newtruncators(bulk, conalg::InfiniteContraction) = newtruncators(bulk, conalg.algorithm)

function newtruncators(bulk, conalg::AbstractBoundaryAlgorithm)
    return newcontraction(conalg, hilbertschmidt(bulk); store_initial=true)
end

function newtruncators(bulk, ::NTU)
    return initneighbourhood(hilbertschmidt(bulk))
end

## Initalialize truncators using existing environment state
newtruncators(tensors, ::NTUState) = newtruncators(tensors, NTU())
function newtruncators(tensors, st::AbstractBoundaryRuntime)
    return newtruncators(tensors, st.alg)
end

newtruncators(tensors, st::SimpleUpdateState) = st
newtruncators(st::SimpleUpdateState) = st

function renormalize!(env::InfiniteContraction, network)
    ITC.recycle!(env, network)
    ITC.run!(env)
    return env
end
function renormalize!(env::InfiniteContraction)
    ITC.reset!(env)
    ITC.run!(env)
    return env
end

renormalize!(st::NTUState) = calculate_neighbourhood!(st)
renormalize!(st::SimpleUpdateState) = st

function renormalize(tensors, truncalg)
    truncs = newtruncators(tensors, truncalg)
    renormalize!(truncs)
    return truncs
end

"""
    metric(pepo::LocallyEvolvingPEPO, environment::AbstractEnvironment)
    metric(pepo::GloballyEvolvingPEPO, environment::AbstractEnvironment)

Construct the truncation metric calculated from `environmnet`. The first method returns the metric corresponding to 
truncation around `currentbond(pepo)`. The second method returns metrics for all bonds on the lattice.

"""
function metric(pepo::LocalUpdatablePEPO, environment)
    return metric(pepo, environment, currentbond(pepo))
end

function metric(state::Union{AbstractPEPO,AbstractLattice}, environment, bond::Bond)
    left, right = state[bond]
    return _metric(left, right, environment, bond)
end

function _metric(left, right, env::InfiniteContraction, bond::Bond)
    return _metric(left, right, env.runtime, bond)
end
function _metric(left, right, env::NTUState, bond::Bond)
    return _metric(left, right, env.tensors, bond)
end

"""
    truncate!(pepo::AbstractUpdatablePEPO, iso::TruncationIsometries)

Truncate the state `pepo` according to truncation matrices `iso`.
"""
function truncate!(pepo::AbstractUpdatablePEPO, iso)
    @debug "Before truncation:" isevolved(pepo)

    _truncate!(pepo, iso)
    setisevolved!(pepo, false)

    @debug "After truncation:" isevolved(pepo)

    return pepo
end

# Truncate a locally evolving pepo
function _truncate!(pepo::LocalUpdatablePEPO, iso::TruncationIsometries)
    bond = currentbond(pepo)

    @debug "Truncating bond $bond"

    new_left, new_right = evolvedpair(pepo)
    old_left, old_right = parentpair(pepo, bond)


    # Truncate the left site with an isometry on the right bond
    truncateto!(old_left, new_left, iso.u * sqrt(iso.s))
    # Truncate the right site with an isometry on the left bond
    truncateto!(old_right, sqrt(iso.s) * iso.v, new_right)

    setweights!(pepo, normalize(iso.s), bond)

    @debug "" weight = normalize(iso.s)

    return pepo
end

partialtruncate(site::AbsTen, iso) = _partialtruncate(site, iso.u * sqrt(iso.s))
partialtruncate(iso, site::AbsTen) = _partialtruncate(sqrt(iso.s) * iso.v, site)

function _partialtruncate(arg1, arg2)
    site_pt = similar_but_truncated(arg1, arg2)
    @debug "" space(site_pt)
    truncateto!(site_pt, arg1, arg2)
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

function _truncate!(pepo::GlobalUpdatablePEPO, iso::TruncationIsometries)
    @assert size(latticetensors(pepo)) == (1, 1)

    # Absorb weights into the isometries `u`.
    u = iso.u * sqrt(iso.s)

    v = sqrt(iso.s) * iso.v

    # Truncate all bonds simultaneously.

    _truncate!.(latticetensors(pepo), evolvedtensors(pepo), Ref(u), Ref(v))

    return pepo
end

function vidaltruncation!(pepo_dst, pepo_src, tno, vidal)
    truncateto!(pepo_dst, pepo_src, tno, vidal.east, vidal.west, vidal.south, vidal.north)
    return pepo_dst
end

# Truncation macros.

# East-bond truncation
function truncateto!(p::AbsTen{2,4}, q::AbsTen{2,4}, u::AbsTen{1,1})
    @tensoropt p[k b; e s w n] = q[k b; ee s w n] * u[ee; e]
    return p
end

# West bond truncation
function truncateto!(p::AbsTen{2,4}, v::AbsTen{1,1}, q::AbsTen{2,4})
    @tensoropt p[k b; e s w n] = v[w; ww] * q[k b; e s ww n]
    return p
end

# Both
function truncateto!(p::AbsTen{2,4}, q::AbsTen{2,4}, u::AbsTen{1,1}, v::AbsTen{1,1})
    @tensoropt p[k b; e s w n] = v[w; ww] * q[k b; ee s ww n] * u[ee; e]
    return p
end
function truncateto!(
    p::AbsTen{2,4},     # dst
    q::AbsTen{2,4},     # src
    l::AbsTen{4,4},     # time-stepper
    u1::AbsTen{2,1},    # east
    v1::AbsTen{2,1},    # west
    u2::AbsTen{2,1},    # south
    v2::AbsTen{2,1},    # north
)
    @tensoropt p[k b; e s w n] =
        v1[ww1 ww2; w] *
        v2[nn1 nn2; n] *
        q[kk bb; ee1 ss1 ww1 nn1] *
        l[k b kk bb; ee2 ss2 ww2 nn2] *
        u1[ee1 ee2; e] *
        u2[ss1 ss2; s]
    return p
end
