# User-facing driver: holds the state, its bond weights and the Trotter decomposition, and runs the
# time steps with a callback schedule.

"""
$(TYPEDEF)

Mutable state of a simulation: the density-matrix PEPO, its bond weights, and the Trotter
decomposition of the propagator. The fields are internal.

# Constructors

    Simulation(state::InfinitePEPO, model; timestep, weights = SUWeight(state))

`model` is a term or a vector of terms, one per Trotter layer, as [`trotterise`](@ref) accepts.

See also: [`simulate!`](@ref)
"""
struct Simulation{M, D}
    info::SimulationInfo
    model::M
    state::InfinitePEPO
    weights::SUWeight
    decomposition::D
    timestep::Float64
end

function Simulation(
        state::InfinitePEPO, model; timestep, weights::SUWeight = SUWeight(state),
        info::SimulationInfo = SimulationInfo()
    )
    P = only(unique(physicalspace(state)))
    return Simulation(info, model, state, weights, trotterise(model, P, timestep), timestep)
end

quantumstate(sim::Simulation) = sim.state
bondweights(sim::Simulation) = sim.weights
timestep(sim::Simulation) = sim.timestep

"""
    $(TYPEDSIGNATURES)

Named tuple of the simulation's progress: see [`SimulationInfo`](@ref).
"""
function simulationinfo(sim::Simulation)
    names = fieldnames(SimulationInfo)
    return NamedTuple{names}(map(name -> getfield(sim.info, name), names))
end

"""
    $(TYPEDSIGNATURES)

Largest relative change per unit time between two sets of bond weights, the measure of approach to a
steady state. Bonds whose space changed count as `Inf`.
"""
function weightdistance(previous::SUWeight, current::SUWeight, dt::Real)
    return maximum(zip(previous, current)) do (old, new)
        space(old) == space(new) || return Inf
        return norm(old - new) / (dt * norm(new) * dim(codomain(new)))
    end
end

"""
    simulate!([callback = identity], sim::Simulation; numsteps, kwargs...)

Run `numsteps` time steps. `callback` is passed `sim` itself.

## Optional keywords

- `maxshots::Integer = numsteps`: call `callback` at most this many times, at evenly spaced steps
- `maxtime::Float64 = Inf`: stop once this much wall-clock time has elapsed
- `breaktol::Float64 = 0.0`: stop once the change in bond weights per unit time falls below this
- `verbose::Bool = true`: log progress at `@info`
"""
simulate!(sim::Simulation; kwargs...) = simulate!(identity, sim; kwargs...)

function simulate!(
        callback, sim::Simulation; numsteps::Integer, maxshots::Integer = numsteps,
        maxtime::Float64 = Inf, breaktol::Float64 = 0.0, verbose::Bool = true
    )
    info = sim.info
    info.finished = false
    started = time()
    walltime₀, simtime₀ = info.walltime, info.simtime

    shotsat = collect(range(; start = 0, step = max(1, cld(numsteps, maxshots)), stop = numsteps))
    previous = deepcopy(sim.weights)

    info.iterations == 0 && callback(sim)

    step = 0
    while step < numsteps
        timestep!(sim.state, sim.weights, sim.decomposition)
        step += 1

        info.convergence = weightdistance(previous, sim.weights, sim.timestep)
        previous = deepcopy(sim.weights)
        info.iterations += 1
        info.simtime = simtime₀ + step * sim.timestep
        info.walltime = walltime₀ + (time() - started)

        verbose && @info "step $step of $numsteps, convergence $(info.convergence)"
        step in shotsat && callback(sim)

        if time() - started > maxtime
            verbose && @info "wall-clock budget exhausted after $step steps"
            break
        elseif info.convergence < breaktol
            info.converged = true
            verbose && @info "converged after $step steps"
            break
        end
    end

    info.finished = true
    return sim
end
