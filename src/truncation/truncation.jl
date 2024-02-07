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

const EnvironmentOrNeighbourhoodState = Union{NeighbourhoodUpdateState,FullEnvironmentState}

# Vidal gauge truncation (simmilar to SU)
struct VidalTruncationState{M<:MessageTensors,U<:VidalIsometries,S<:VidalWeights,T} <:
       AbstractTruncatorState
    messages::Lattice{Square,M,Matrix{M}} # only works for
    isometries::Lattice{Square,U,Matrix{U}}
    weights::Lattice{Square,S,Matrix{S}}
    map::T
end

# SU, FET, NTU
abstract type AbstractTruncationScheme end

function Base.show(io::IO, m::MIME"text/plain", sim::AbstractTruncationScheme)
    shownested(io, m, sim)
    return nothing
end

# TRUNCATION SCHEMES
@kwdef struct FET{BAlg<:AbstractBoundaryAlgorithm,UScheme<:AbstractUpdateScheme} <:
              AbstractTruncationScheme
    gauge::Bool = true
    ξ::Float64 = Inf
    rankreduce = true
    alg::BAlg
    updatescheme::UScheme = LULT()
end

showheader(::Type{<:FET}) = "Full environment truncation (FET)"

@kwdef struct NTU{UScheme<:AbstractUpdateScheme} <: AbstractTruncationScheme
    gauge::Bool = false
    ξ::Float64 = Inf
    rankreduce = true
    updatescheme::UScheme = LULT()
end

showheader(::Type{<:NTU}) = "Neighbourhood tensor update (NTU)"


@kwdef struct SU <: AbstractTruncationScheme
    ξ::Float64 = Inf
end

showheader(::Type{<:SU}) = "Simple update"

@kwdef struct VGU <: AbstractTruncationScheme
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

## Initalialize truncators for a full TD-PEPO
function inittruncators(
    pepo::NTuple{2,P}, ::NTuple{2,L}, truncalg::Union{FET,NTU}
) where {P<:AbstractPEPO,L}
    xenv = inittruncators(pepo[1], truncalg)
    yenv = inittruncators(pepo[2], truncalg)
    return xenv, yenv
end
function inittruncators(
    ::NTuple{2,P}, ::NTuple{2,L}, truncalg::SU
) where {P<:AbstractPEPO,L}
    return SimpleUpdateState(), SimpleUpdateState()
end

## 

function ITC.newcontraction(conalg, net::HilbertSchmidt; kwargs...)
    return newcontraction(conalg, net.data; kwargs...)
end

## Initalialize truncators for lattice of tensors
inittruncators(bulk, fet::FET) = inittruncators(bulk, fet.alg)
inittruncators(bulk, conalg::InfiniteContraction) = inittruncators(bulk, conalg.algorithm)

function inittruncators(bulk, conalg::AbstractBoundaryAlgorithm)
    return newcontraction(conalg, hilbertschmidt(bulk); store_initial=true)
end

function inittruncators(bulk, ::NTU)
    return initialize_neighbourhood(hilbertschmidt(bulk))
end

## Initalialize truncators using existing environment state
inittruncators(tensors, ::NTUState) = inittruncators(tensors, NTU())
function inittruncators(tensors, st::AbstractBoundaryRuntime)
    return inittruncators(tensors, st.alg)
end

function inittruncators(pepo, dynamical_map, ::VGU)
    map = dynamical_map.x
    st = getstate(pepo.x)

    messages = initmessages.(st, map.data)

    function useful_f1(p, t)
        pdom = domain(p)
        mdom = domain(t)

        ue = TensorMap(
            undef, promote_type(eltype(p), eltype(t)), pdom[1] * mdom[1], pdom[1]
        )
        us = TensorMap(
            undef, promote_type(eltype(p), eltype(t)), pdom[2] * mdom[2], pdom[2]
        )
        vw = TensorMap(
            undef, promote_type(eltype(p), eltype(t)), pdom[3] * mdom[3], pdom[3]
        )
        vn = TensorMap(
            undef, promote_type(eltype(p), eltype(t)), pdom[4] * mdom[4], pdom[4]
        )
        return VidalIsometries(ue, us, vw, vn)
    end
    function useful_f2(p)
        pdom = domain(p)

        ue = TensorMap(undef, eltype(p), pdom[1], pdom[1])
        us = TensorMap(undef, eltype(p), pdom[2], pdom[2])
        vw = TensorMap(undef, eltype(p), pdom[3]', pdom[3]')
        vn = TensorMap(undef, eltype(p), pdom[4]', pdom[4]')
        return VidalWeights(ue, us, vw, vn)
    end

    vidals = useful_f1.(st, map.data)
    weights = useful_f2.(st)

    return VidalTruncationState(messages, vidals, weights, map)
end

inittruncators(tensors, st::SimpleUpdateState) = st
inittruncators(st::SimpleUpdateState) = st

# Truncator renormalization
function renormalize!(st::EnvironmentOrNeighbourhoodState)
    renormalize!(st.x)
    renormalize!(st.y)
    return st
end

# function renormalize!(env::InfiniteContraction)
#     ITC.reset!(env)
#     ITC.run!(env)
#     return env
# end

function renormalize!(env::InfiniteContraction)
    ITC.restart!(env)
    ITC.run!(env)
    return env
end

renormalize!(st::NTUState) = calculate_neighbourhood!(st)
renormalize!(st::SimpleUpdateState) = st

function truncators(tensors, truncalg)
    truncs = inittruncators(tensors, truncalg)
    renormalize!(truncs)
    return truncs
end
# function renormalize(tensors, env::NTUState)
#     new_env = ntu_initenvironment(tensors)
#     ITC.calculate!(new_env)
#     return new_env
# end
# renormalize(tensors, env::SimpleUpdateState) = env

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

    @debug "Truncating bond: $bond"

    new_left, new_right = pepo[bond]
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

# Truncate a globally evolving pepo
function _truncate!(pepo::GlobalUpdatablePEPO, truncators::VidalTruncationState)
    dst = getstate(pepo)
    src = getevolved(pepo)

    tno = truncators.map
    vidals = truncators.isometries

    vidaltruncation!.(dst, src, tno.data, vidals)
    copy!.(src, dst)

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
