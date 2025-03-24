abstract type AbstractIsometryOptimisation{G<:Union{NoGauge,WTG}} end

using InfiniteTensorContractions: newrenormalization, RenormalizationProblem
function Base.show(io::IO, m::MIME"text/plain", sim::AbstractIsometryOptimisation)
    return shownested(io, m, sim)
end

@kwdef struct EVB{G} <: AbstractIsometryOptimisation{G}
    maxiter::Int = 100
    verbose::Bool = false
    pinvtol::Float64 = sqrt(eps())
    isotol::Float64 = 1e-8
    randomness::Float64 = 1e-3
    gauge::G = WTG()
end

@kwdef struct DZM{G} <: AbstractIsometryOptimisation{G}
    maxiter::Int = 100
    verbose::Bool = false
    pinvtol::Float64 = sqrt(eps())
    isotol::Float64 = 1e-8
    randomness::Float64 = 0.0
    EAT::Bool = false
    check::Bool = false
    gauge::G = NoGauge()
end

# struct TruncationIsometries{T<:AbsTen{1,1}}
#     u::T
#     s::T
#     v::T
# end
struct TruncationIsometries{TU,TS,TV}
    u::TU
    s::TS
    v::TV
end

struct BondIsometries{Geom,U,S,V}
    u::UnitCell{Geom,U,Matrix{U}}
    s::UnitCell{Geom,S,Matrix{S}}
    v::UnitCell{Geom,V,Matrix{V}}
end

struct SiteIsometries{Geom,U,V}
    ux::UnitCell{Geom,U,Matrix{U}}
    vx::UnitCell{Geom,V,Matrix{V}}
    uy::UnitCell{Geom,U,Matrix{U}}
    vy::UnitCell{Geom,V,Matrix{V}}
    function SiteIsometries(
        ux::UnitCell{Geom,U,Matrix{U}},
        vx::UnitCell{Geom,V,Matrix{V}},
        uy::UnitCell{Geom,U,Matrix{U}},
        vy::UnitCell{Geom,V,Matrix{V}},
    ) where {Geom,U,V}
        @assert size(ux) == size(vx) == size(uy) == size(vy)
        return new{Geom,U,V}(ux, vx, uy, vy)
    end
end

struct AxisView{T}
    parent::T
    axis::Int
end

unpack(iso::SiteIsometries) = ((iso.ux, iso.vx), (iso.uy, iso.vy))
viewaxis(iso::SiteIsometries, axis) = AxisView(iso, axis)

Base.iterate(iso::SiteIsometries, state=1) = iso[state], state + 1

function Base.iterate(iso::AxisView, state=1)
    next, state = iterate(iso.parent, state)
    return next[iso.axis], state
end
Base.length(view::AxisView) = length(view.parent)

unpack(iso::AxisView) = unpack(iso.parent)[iso.axis]

function Base.getindex(iso::SiteIsometries, bond::Bond)
    i1, i2 = bond

    axis = bondaxis(bond)

    if axis == 1
        u = iso.ux[i1]
        v = iso.vx[i2]
    elseif axis == 2
        u = iso.uy[i1]
        v = iso.vy[i2]
    end

    return u => v
end

function Base.getindex(iso::SiteIsometries, I::Dims)
    e = iso.ux[I]
    w = iso.vx[I]

    s = iso.uy[I]
    n = iso.vy[I]

    return e, s, w, n
end

function seperateweights(
    horizontal::UnitCell{G,Tuple{Pair{TU,TV},TS}},
    vertical::UnitCell{G,Tuple{Pair{TU,TV},TS}},
) where {G,TU,TV,TS}
    nxny = size(isometries)

    uhlat = UnitCell{G,TU}(undef, nxny)
    vhlat = UnitCell{G,TV}(undef, nxny)
    shlat = UnitCell{G,TS}(undef, nxny)

    uvlat = UnitCell{G,TU}(undef, nxny)
    vvlat = UnitCell{G,TV}(undef, nxny)
    svlat = UnitCell{G,TS}(undef, nxny)

    hind = eachindex(horizontal)
    vind = eachindex(vertical)

    if hind != vind
        throw(DimensionMismatch())
    end

    populate_isometries_weights!(uhlat, vhlat, shlat, horizontal; axis=1)
    populate_isometries_weights!(uvlat, vvlat, svlat, vertical; axis=2)

    return SiteIsometries(uhlat, vhlat, uvlat, vvlat), UnitCellBonds(shlat, svlat)
end

function populate_isometries_weights!(ulat, vlat, slat, isometries; axis)
    axisbonds = bonds(isometries, axis)

    foreach(axisbonds) do bond
        uind, vind = bond

        (u, v), s = isometries[uind]

        ulat[uind] = u
        vlat[vind] = v
        slat[vind] = s
    end

    return ulat, vlat, slat
end

function TruncationIsometries(pepo, u, s, v)
    sout = similar(s, scalartype(pepo))
    copy!(sout, s)
    return TruncationIsometries(u, sout, v)
end

abstract type AbstractTruncatorState end

# Most basic algorithm for truncating bonds
struct SimpleUpdateState <: AbstractTruncatorState end

# SU, FET, NTU
abstract type AbstractTruncationScheme end

function Base.show(io::IO, m::MIME"text/plain", sim::AbstractTruncationScheme)
    shownested(io, m, sim)
    return nothing
end

# TRUNCATION SCHEMES
@kwdef mutable struct FET{
    BAlg<:AbstractBoundaryAlgorithm,OAlg<:AbstractIsometryOptimisation,G<:Union{NoGauge,MCF}
} <: AbstractTruncationScheme
    ξ::Float64 = Inf
    rankreduce::Bool = true
    optalg::OAlg = EVB()
    gauge::G = NoGauge()
    const alg::BAlg
end

showheader(::Type{<:FET}) = "Full environment truncation (FET)"

@kwdef mutable struct NTU{OAlg<:AbstractIsometryOptimisation,G<:Union{NoGauge,MCF}} <:
                      AbstractTruncationScheme
    ξ::Float64 = Inf
    rankreduce::Bool = true
    optalg::OAlg = DZM()
    gauge::G = NoGauge()
end

showheader(::Type{<:NTU}) = "Neighbourhood tensor update (NTU)"

@kwdef mutable struct SU{G<:Union{NoGauge,MCF}} <: AbstractTruncationScheme
    gauge::G = NoGauge()
    ξ::Float64 = Inf
end

showheader(::Type{<:SU}) = "Simple update"

@kwdef mutable struct VGU <: AbstractTruncationScheme
    ξ::Float64 = Inf
end

## Initalialize truncators for a full TD-PEPO
function newtruncators(
    pepo::NTuple{2,P}, ::Vector{S}, truncalg::Union{FET,NTU}
) where {P<:AbstractPEPO,S}
    xenv = newtruncators(pepo[1], truncalg)
    yenv = newtruncators(pepo[2], truncalg)
    return xenv, yenv
end

function newtruncators(pepo::PEPO, _, truncalg::Union{FET,NTU})
    xenv = newtruncators(pepo, truncalg)
    return xenv, xenv
end
function newtruncators(::PEPO, _, truncalg::SU)
    xenv = SimpleUpdateState()
    return xenv, xenv
end
function newtruncators(pepo::AxialPEPO, _, truncalg::Union{FET,NTU})
    xenv = newtruncators(getaxis(pepo, 1), truncalg)
    yenv = newtruncators(getaxis(pepo, 2), truncalg)
    return xenv, yenv
end

function newtruncators(::NTuple{2,P}, ::Vector{S}, truncalg::SU) where {P<:AbstractPEPO,S}
    return SimpleUpdateState(), SimpleUpdateState()
end

## 

function ITC.newrenormalization(net::HilbertSchmidt; alg=alg, kwargs...)
    return newrenormalization(net.data; alg=alg, kwargs...)
end

## Initalialize truncators for lattice of tensors
newtruncators(bulk, fet::FET) = newtruncators(bulk, fet.alg)
newtruncators(bulk, conalg::RenormalizationProblem) = newtruncators(bulk, conalg.alg)

function newtruncators(bulk, conalg::AbstractBoundaryAlgorithm)
    return newrenormalization(hilbertschmidt(bulk); alg=conalg)
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

environment!(pepo, ::SimpleUpdateState, ::Bond) = weights(pepo)
environment!(pepo, ::NTUState, bond::Bond) = pair -> ntu_total(pair, pepo, bond)

function environment!(pepo, truncator::RenormalizationProblem, ::Bond)
    hs = hilbertschmidt(pepo)
    ITC.renormalize!(truncator, hs)
    return truncator
end

function ITC.renormalize!(env::RenormalizationProblem, network)
    ITC.recycle!(env, network)
    ITC.renormalize!(env)
    return env
end

# function ITC.renormalize!(env::RenormalizationProblem)
#     ITC.reset!(env)
#     ITC.runcontraction!(env)
#     return env
# end

ITC.renormalize!(st::NTUState) = calculate_neighbourhood!(st)
ITC.renormalize!(st::SimpleUpdateState) = st

function ITC.renormalize(tensors, truncalg)
    truncs = newtruncators(tensors, truncalg)
    renormalize!(truncs)
    return truncs
end

# function renormalize!(truncator::NTuple{2}, buffer)
#     truncx, truncy = truncator
#
#     renormalize!(truncx, buffer)
#     renormalize!(truncy, PermuteTranpose(buffer))
#
#     return truncator
# end

"""
    metric(pepo::LocallyEvolvingPEPO, environment::AbstractEnvironment)
    metric(pepo::GloballyEvolvingPEPO, environment::AbstractEnvironment)

Construct the truncation metric calculated from `environmnet`. The first method returns the metric corresponding to 
truncation around `currentbond(pepo)`. The second method returns metrics for all bonds on the lattice.

"""
# function metric(pepo::LocalUpdatablePEPO, environment)
#     return metric(pepo, environment, currentbond(pepo))
# end

function metric(state::Union{AbstractPEPO,AbstractUnitCell}, environment, bond::Bond)
    left, right = state[bond]
    return _metric(left, right, environment, bond)
end

function metric(
    ::Union{AbstractPEPO,AbstractUnitCell}, environment::AbstractTensorMap, ::Bond
)
    return environment
end
function metric(pair::Pair, env, bond)
    left, right = pair
    return _metric(left, right, env, bond)
end
function _metric(left, right, env::RenormalizationProblem, bond::Bond)
    return _metric(left, right, env.runtime, bond)
end
function _metric(left, right, env::NTUState, bond::Bond)
    return _metric(left, right, env.tensors, bond)
end
_metric(left, right, env::AbstractTensorMap, bond::Bond) = env
_metric(left, right, env::AbstractUnitCell, bond::Bond) = env

"""
    truncate!(pepo::AbstractUpdatablePEPO, iso::TruncationIsometries)

Truncate the state `pepo` according to truncation matrices `iso`.
"""
# function truncate!(pepo::AbstractUpdatablePEPO, iso)
#     @debug "Before truncation:" isevolved(pepo)
#
#     _truncate!(pepo, iso)
#     setisevolved!(pepo, false)
#
#     @debug "After truncation:" isevolved(pepo)
#
#     return pepo
# end

# # Truncate a locally evolving pepo
# function _truncate!(pepo::LocalUpdatablePEPO, iso::TruncationIsometries; axis...)
#     bond = currentbond(pepo)
#
#     @debug "Truncating bond $bond"
#
#     new_left, new_right = evolvedpair(pepo)
#     old_left, old_right = parentpair(pepo, bond)
#
#     # Truncate the left site with an isometry on the right bond
#     truncateto!(old_left, new_left, iso.u; axis...)
#     # Truncate the right site with an isometry on the left bond
#     truncateto!(old_right, iso.v, new_right; axis...)
#
#     setweight!(pepo, iso.s, bond)
#
#     @debug "" weight = iso.s
#
#     return pepo
# end

partialtruncate(site::AbsTen, iso) = _partialtruncate(site, iso.u)
partialtruncate(iso, site::AbsTen) = _partialtruncate(iso.v, site)

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

# function _truncate!(pepo::GlobalUpdatablePEPO, iso::TruncationIsometries)
#     @assert size(tensors(pepo)) == (1, 1)
#
#     # Truncate all bonds simultaneously.
#
#     _truncate!.(tensors(pepo), currentbuffer(pepo), Ref(iso.u), Ref(iso.v))
#
#     return pepo
# end

function vidaltruncation!(pepo_dst, pepo_src, tno, vidal)
    truncateto!(pepo_dst, pepo_src, tno, vidal.east, vidal.west, vidal.south, vidal.north)
    return pepo_dst
end

# Truncation macros.

function truncate!(dstbuff, srcbuff, isometries, bond)
    axis = bondaxis(bond)
    return truncateto!(dstbuff[bond], srcbuff[bond], isometries; axis=axis)
end

function truncateto!(
    dst::Pair{T,T}, src::Pair{T,T}, uv::Pair{S,S}; axis
) where {T<:AbsTen{2,4},S<:AbsTen{1,1}}
    p1, p2 = dst
    q1, q2 = src

    u, v = uv

    truncateto!(p1, q1, u; axis=axis)
    truncateto!(p2, v, q2; axis=axis)

    return dst
end

function truncateto!(p::AbsTen{2,4}, q::AbsTen{2,4}, u::AbsTen{1,1}; axis)
    if axis == 1
        @tensor p[k b; e s w n] = q[k b; ee s w n] * u[ee; e]
    elseif axis == 2
        @tensor p[k b; e s w n] = q[k b; e ss w n] * u[ss; s]
    end
    return p
end

function truncateto!(p::AbsTen{2,4}, v::AbsTen{1,1}, q::AbsTen{2,4}; axis)
    if axis == 1
        @tensor p[k b; e s w n] = q[k b; e s ww n] * v[w; ww]
    elseif axis == 2
        @tensor p[k b; e s w n] = q[k b; e s w nn] * v[n; nn]
    end
    return p
end

# Both
function truncateto!(p::AbsTen{2,4}, q::AbsTen{2,4}, u, v; axis=1)
    if axis == 1
        @tensor p[k b; e s w n] = v[w; ww] * q[k b; ee s ww n] * u[ee; e]
    elseif axis == 2
        @tensor p[k b; e s w n] = v[n; nn] * q[k b; e ss w nn] * u[ss; s]
    end
    return p
end

function truncate_swap!(p::AbsTen{2,4}, q::AbsTen{2,4}, u::AbsTen{1,1}, v::AbsTen{1,1})
    @tensoropt p[k b; s e n w] = v[w; ww] * q[k b; ee s ww n] * u[ee; e]
    return p
end

function truncateto!(
    p::AbsTen{2,4},     # dst
    q::AbsTen{2,4},     # src
    l::AbsTen{4,4},     # time-stepper
    u1::AbsTen{2,1},    # east
    u2::AbsTen{2,1},    # south
    v1::AbsTen{2,1},    # west
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

function truncateto!(
    p::AbsTen{2,4},     # dst
    q::AbsTen{2,4},     # src
    l::AbsTen{4,4},     # time-stepper
    u1::AbsTen{2,1},    # east
    u2::AbsTen{2,1},    # south
    v1::AbsTen{1,2},    # west
    v2::AbsTen{1,2},    # north
)
    @tensoropt p[k b; e s w n] =
        v1[w; ww1 ww2] *
        v2[n; nn1 nn2] *
        q[kk bb; ee1 ss1 ww1 nn1] *
        l[k b kk bb; ee2 ss2 ww2 nn2] *
        u1[ee1 ee2; e] *
        u2[ss1 ss2; s]
    return p
end

# function truncateto(
#     p::AbsTen{2,4},     # dst
#     q::AbsTen{2,4},     # src
#     l::SparseTensorMap{4,4},     # time-stepper
#     u1::AbsTen{1,1},    # east
#     u2::AbsTen{1,1},    # south
#     v1::AbsTen{1,1},    # west
#     v2::AbsTen{1,1},    # north
# )
#     p = sptensor(p)
#     q = sptensor(q)
#
#     u1= sptensor(u1)
#     u2= sptensor(u2)
#     v1= sptensor(v1)
#     v2= sptensor(v2)
#
#     D1, D2, D3, D4 = domain(q)
#     η1, η2, η3, η4 = domain(l)
#
#     u1p = similar(u1, D1 * η1, D1)
#     u2p = similar(u2, D2 * η2, D2)
#
#     v1p = similar(u1, D3', D3' * η3')
#     v2p = similar(u2, D4', D4' * η4')
#
#     mergeinto!(u1p,u1)
#     mergeinto!(u2p,u2)
#
#     mergeinto!(v1p,v1)
#     mergeinto!(v2p,v2)
#
#     @tensoropt p[k b; e s w n] =
#         v1p[w; ww1 ww2] *
#         v2p[n; nn1 nn2] *
#         q[kk bb; ee1 ss1 ww1 nn1] *
#         l[k b kk bb; ee2 ss2 ww2 nn2] *
#         u1p[ee1 ee2; e] *
#         u2p[ss1 ss2; s]
#
#     droptol!(p)
#
#     return p
# end

function truncateto!(
    p::AbsTen{2,4},     # dst
    q::AbsTen{2,4},     # src
    l::AbsTen{4,4},     # time-stepper
    u1::AbsTen{1,1},    # east
    u2::AbsTen{1,1},    # south
    v1::AbsTen{1,1},    # west
    v2::AbsTen{1,1},    # north
)
    D1, D2, D3, D4 = domain(q)
    η1, η2, η3, η4 = domain(l)

    u1p = similar(u1, D1 * η1, D1)
    u2p = similar(u2, D2 * η2, D2)

    v1p = similar(u1, D3', D3' * η3')
    v2p = similar(u2, D4', D4' * η4')

    mergeinto!(u1p, u1)
    mergeinto!(u2p, u2)

    mergeinto!(v1p, v1)
    mergeinto!(v2p, v2)

    @tensoropt (
        k => 2,
        b => 2,
        kk => 2,
        bb => 2,
        e => D,
        s => D,
        w => D,
        n => D,
        ee2 => D^2,
        ss2 => D^2,
        ww2 => D^2,
        nn2 => D^2,
        ee1 => D,
        ss1 => D,
        ww1 => D,
        nn1 => D,
    ) p[k b; e s w n] =
        v1p[w; ww1 ww2] *
        v2p[n; nn1 nn2] *
        q[kk bb; ee1 ss1 ww1 nn1] *
        l[k b kk bb; ee2 ss2 ww2 nn2] *
        u1p[ee1 ee2; e] *
        u2p[ss1 ss2; s]
    return p
end
function truncateto!(
    p::AbsTen{2,4},     # dst
    q::AbsTen{2,4},     # src
    l::AbsTen{4,2},     # time-stepper
    u1::AbsTen{2,1},    # east
    u2::AbsTen{1,1},    # south
    v1::AbsTen{1,2},    # west
    v2::AbsTen{1,1},    # north
)
    @tensoropt p[k b; e s w n] =
        v1[w; ww1 ww2] *
        v2[n; nn1] *
        q[kk bb; ee1 ss1 ww1 nn1] *
        l[k b kk bb; ee2 ww2] *
        u1[ee1 ee2; e] *
        u2[ss1; s]
    return p
end

function truncateto!(
    p::AbsTen{2,4},     # dst
    q::AbsTen{2,4},     # src
    u1::AbsTen{1,1},    # east
    u2::AbsTen{1,1},    # south
    v1::AbsTen{1,1},    # west
    v2::AbsTen{1,1},    # north
)
    @tensoropt p[k b; e s w n] =
        v1[w; ww] * v2[n; nn] * q[k b; ee ss ww nn] * u1[ee; e] * u2[ss; s]
    return p
end

function truncateto!(
    p::AbsTen{2,4},     # dst
    q::AbsTen{2,4},     # src
    l::AbsTen{4,2},     # time-stepper
    u1::AbsTen{2,1},    # east
    v1::AbsTen{1,2},    # west
)
    @tensoropt p[k b; e s w n] =
        v1[w; ww1 ww2] *
        q[kk bb; ee1 s ww1 n] *
        l[k b kk bb; ee2 ww2] *
        u1[ee1 ee2; e] *
        return p
end
