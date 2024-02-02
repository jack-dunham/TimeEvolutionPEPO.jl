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
    ξ::Float64 = Inf
end

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
function initialize_truncators(pepo::TimeDependantPEPO, ::DynamicalMap, truncalg::FET)
    xenv = initialize_truncators(pepo.x, truncalg)
    yenv = initialize_truncators(pepo.y, truncalg)
    return FullEnvironmentState(xenv, yenv)
end
function initialize_truncators(pepo::TimeDependantPEPO, ::DynamicalMap, truncalg::NTU)
    xenv = initialize_truncators(pepo.x, truncalg)
    yenv = initialize_truncators(pepo.y, truncalg)
    return NeighbourhoodUpdateState(xenv, yenv)
end
function initialize_truncators(::TimeDependantPEPO, ::DynamicalMap, truncalg::SU)
    return SimpleUpdateState()
end
## 

function ITC.newcontraction(conalg, net::HilbertSchmidt; kwargs...)
    return newcontraction(conalg, net.data; kwargs...)
end

## Initalialize truncators for lattice of tensors
initialize_truncators(bulk, fet::FET) = initialize_truncators(bulk, fet.alg)
function initialize_truncators(bulk, conalg::AbstractBoundaryAlgorithm)
    return newcontraction(conalg, hilbertschmidt(bulk); store_initial=true)
end

function initialize_truncators(bulk, ::NTU)
    return initialize_neighbourhood(hilbertschmidt(bulk))
end

## Initalialize truncators using existing environment state
initialize_truncators(tensors, ::NTUState) = initialize_truncators(tensors, NTU())
function initialize_truncators(tensors, st::AbstractBoundaryRuntime)
    return initialize_truncators(tensors, st.alg)
end

function initialize_truncators(pepo::TimeDependantPEPO, dynamical_map, ::VGU)
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

initialize_truncators(st::SimpleUpdateState) = st

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
    truncs = initialize_truncators(tensors, truncalg)
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

function metric(pepo::Union{AbstractPEPO,AbstractLattice}, environment, bond::Bond)
    left, right = pepo[bond]
    return _metric(left, right, environment, bond)
end

function _metric(left, right, env::InfiniteContraction, bond)
    return _metric(left, right, env.runtime, bond)
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
    old_left, old_right = getstate(pepo)[bond]

    # Truncate the left site with an isometry on the right bond
    tensor_truncate!(old_left, new_left, iso.u * sqrt(iso.s))
    # Truncate the right site with an isometry on the left bond
    tensor_truncate!(old_right, sqrt(iso.s) * iso.v, new_right)

    setweights!(pepo, normalize(iso.s), bond)

    @debug "" weight = normalize(iso.s)

    return pepo
end

partialtruncate(site::AbsTen, iso) = _partialtruncate(site, iso.u * sqrt(iso.s))
partialtruncate(iso, site::AbsTen) = _partialtruncate(sqrt(iso.s) * iso.v, site)

function _partialtruncate(arg1, arg2)
    site_pt = similar_but_truncated(arg1, arg2)
    @debug "" space(site_pt)
    tensor_truncate!(site_pt, arg1, arg2)
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
    tensor_truncate!(
        pepo_dst, pepo_src, tno, vidal.east, vidal.west, vidal.south, vidal.north
    )
    return pepo_dst
end

# Truncation macros.

function tensor_truncate!(p::AbsTen{2,4}, q::AbsTen{2,4}, u::AbsTen{1,1})
    @tensoropt p[k b; e s w n] = q[k b; ee s w n] * u[ee; e]
    return p
end
function tensor_truncate!(p::AbsTen{2,4}, v::AbsTen{1,1}, q::AbsTen{2,4})
    @tensoropt p[k b; e s w n] = v[w; ww] * q[k b; e s ww n]
    return p
end
function tensor_truncate!(p::AbsTen{2,4}, q::AbsTen{2,4}, u::AbsTen{1,1}, v::AbsTen{1,1})
    @tensoropt p[k b; e s w n] = v[w; ww] * q[k b; ee s ww n] * u[ee; e]
    return p
end
function tensor_truncate!(
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
