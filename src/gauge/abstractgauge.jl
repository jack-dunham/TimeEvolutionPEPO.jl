# Gauge fixing. Both algorithms are PEPSKit's; these wrappers write the result back into the state and
# weights a `Simulation` holds, since `gauge_fix` returns new objects.

"""
    $(TYPEDSIGNATURES)

Fix the gauge of `state` in place, updating `weights` to match, and return the residual reported by
the algorithm.

`SUGauge` runs trivial simple update with identity gates. `BPGauge` uses a belief-propagation fixed
point and needs its environment; note that PEPSKit fuses the two physical legs internally for the PEPO
case and unfuses on the way out, so the state stays a density-matrix PEPO throughout.
"""
function gaugefix!(state::InfinitePEPO, weights::SUWeight, alg::SUGauge = SUGauge())
    fixed, wts, ϵ = gauge_fix(state, alg)
    _copystate!(state, fixed)
    _copyweights!(weights, wts)
    return ϵ
end

function gaugefix!(state::InfinitePEPO, weights::SUWeight, alg::BPGauge, env::BPEnv)
    fixed, _ = gauge_fix(state, alg, env)
    _copystate!(state, fixed)
    _copyweights!(weights, SUWeight(env))
    return nothing
end

gaugefix!(sim::Simulation, args...) = gaugefix!(sim.state, sim.weights, args...)

function _copystate!(dst::InfinitePEPO, src::InfinitePEPO)
    Nr, Nc, = size(dst)
    for r in 1:Nr, c in 1:Nc
        dst[r, c] = src[r, c]
    end
    return dst
end

function _copyweights!(dst::SUWeight, src::SUWeight)
    for d in 1:2, r in axes(dst, 2), c in axes(dst, 3)
        dst[d, r, c] = src[d, r, c]
    end
    return dst
end

"""
    $(TYPEDSIGNATURES)

Belief-propagation environment of `state`, the fixed point `BPGauge` needs.
"""
function bpenvironment(state::InfinitePEPO; alg = BeliefPropagation(), kwargs...)
    env₀ = BPEnv(state; kwargs...)
    env, info = leading_boundary(env₀, InfiniteSquareNetwork(state), alg)
    return env, info
end

"""
$(TYPEDEF)

Settings for the CTMRG boundary that environment-based bond truncation is optimised against: the
boundary space and the tolerances handed to `leading_boundary`.

# Constructors

    BondEnvironment(bonddim; tol = 1e-8, maxiter = 100, verbosity = 0)

`bonddim` is a dimension or an `ElementarySpace`.
"""
Base.@kwdef struct BondEnvironment{S <: ElementarySpace}
    space::S
    tol::Float64 = 1.0e-8
    maxiter::Int = 100
    verbosity::Int = 0
end

function BondEnvironment(bonddim::Union{Integer, ElementarySpace}; kwargs...)
    return BondEnvironment(; space = _indexspace(bonddim), kwargs...)
end

"""
    $(TYPEDSIGNATURES)

CTMRG boundary of the PEPO sandwiched with its adjoint — the Hilbert-Schmidt metric that
environment-based bond truncation is optimised against. Returns the environment and the convergence
info.

`env₀` warm-starts the solve when its network legs match the state's, and is ignored otherwise.
"""
function bondenvironment(state::InfinitePEPO, alg::BondEnvironment; env₀ = nothing)
    network = InfiniteSquareNetwork(state)
    fresh = initialize_ctmrg_environment(ComplexF64, network, RandomInitialization(), alg.space)
    start = (!isnothing(env₀) && _matchesnetwork(env₀, fresh)) ? env₀ : fresh
    return leading_boundary(start, network; alg.tol, alg.maxiter, alg.verbosity)
end

function bondenvironment(
        state::InfinitePEPO, bonddim::Union{Integer, ElementarySpace}; env₀ = nothing, kwargs...
    )
    return bondenvironment(state, BondEnvironment(bonddim; kwargs...); env₀)
end

# An edge tensor carries the network's two virtual legs at positions 2 and 3.
function _matchesnetwork(env::CTMRGEnv, reference::CTMRGEnv)
    size(env.edges) == size(reference.edges) || return false
    return all(zip(env.edges, reference.edges)) do (e, r)
        return space(e, 2) == space(r, 2) && space(e, 3) == space(r, 3)
    end
end
