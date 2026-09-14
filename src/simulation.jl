# User-facing driver: holds the state, its bond weights and the Trotter decomposition, and runs the
# time steps with a callback schedule.

"""
$(TYPEDEF)

Mutable state of a simulation: the density-matrix PEPO, its bond weights, the Trotter decomposition
of the propagator, and the [`DampedUpdate`](@ref) that says how a step is taken. The fields are
internal.

# Constructors

    Simulation(state::InfinitePEPO, model; timestep, alg = DampedUpdate(), weights = SUWeight(state))

`model` is a [`Model`](@ref), or a term or a vector of terms as shorthand for a single-layer or
uniform model, as [`trotterise`](@ref) accepts. `alg` is the default for every [`simulate!`](@ref)
call on this simulation; a call may override it or any of its fields.

See also: [`simulate!`](@ref)
"""
struct Simulation{M, D, A}
    info::SimulationInfo
    model::M
    state::InfinitePEPO
    weights::SUWeight
    decomposition::D
    timestep::Float64
    alg::A
end

function Simulation(
        state::InfinitePEPO, model; timestep, alg::DampedUpdate = DampedUpdate(),
        weights::SUWeight = SUWeight(state), info::SimulationInfo = SimulationInfo()
    )
    P = only(unique(physicalspace(state)))
    decomposition = trotterise(model, P, timestep)

    Nr, Nc, = size(state)
    for (propagators, _) in decomposition
        size(propagators) == (1, 1) || size(propagators) == (Nr, Nc) || throw(
            DimensionMismatch(
                "a model layer of size $(size(propagators)) cannot be applied to a " *
                    "$Nr×$Nc unit cell: give a 1×1 layer or one matching the state"
            ),
        )
    end

    return Simulation(info, model, state, weights, decomposition, timestep, alg)
end

"""
    $(TYPEDSIGNATURES)

The density-matrix PEPO a simulation is evolving. Mutated in place by [`simulate!`](@ref).
"""
quantumstate(sim::Simulation) = sim.state

"""
    $(TYPEDSIGNATURES)

The bond weights of a simulation's state, one per bond of the unit cell.
"""
bondweights(sim::Simulation) = sim.weights

"""
    $(TYPEDSIGNATURES)

The [`DampedUpdate`](@ref) a simulation steps with by default.
"""
algorithm(sim::Simulation) = sim.alg

timestep(sim::Simulation) = sim.timestep

function Base.show(io::IO, m::MIME"text/plain", sim::Simulation)
    println(io, "SIMULATION")
    println(io, "≡≡≡≡≡≡≡≡≡≡")
    println(io, "size: $(size(quantumstate(sim)))")
    println(io, "timestep: $(timestep(sim))")
    println(io, "Trotter layers: $(length(sim.decomposition))")
    println(io, "MODEL")
    println(io, "≡≡≡≡≡")
    show(io, m, sim.model)
    println(io, "")
    println(io, "INFO")
    print(io, "≡≡≡≡")
    for name in fieldnames(SimulationInfo)
        println(io, "")
        print(io, name, ": ", getfield(sim.info, name))
    end
    return nothing
end

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
    simulate!([callback = identity], sim::Simulation, [alg::DampedUpdate]; numsteps, kwargs...)

Run `numsteps` time steps. `callback` is passed `sim` itself, and `alg` says how each bond is updated
and how often the gauge is re-fixed. Omitting `alg` builds one from the keywords below.

## Optional keywords

- `maxshots::Integer = numsteps`: call `callback` at most this many times, at evenly spaced steps
- `maxtime::Float64 = Inf`: stop once this much wall-clock time has elapsed
- `breaktol::Float64 = 0.0`: stop once the change in bond weights per unit time falls below this
- `verbose::Bool = true`: log progress at `@info`
- `trunc`, `truncalg`, `envalg`, `zeta`, `gaugeevery`: fields of [`DampedUpdate`](@ref), each
  defaulting to the simulation's own `alg`, so a call may override one field without restating the rest

A sweep leaves the Vidal gauge only approximately, and imaginary-time evolution amplifies the drift
until the bond spectrum collapses. Setting `gaugeevery` prevents that, and restores a state in which
it has already happened, since the drift misrepresents the state rather than destroying it. Thermal
evolution needs it; Lindblad evolution has not been observed to.
"""
simulate!(sim::Simulation, args...; kwargs...) = simulate!(identity, sim, args...; kwargs...)

function simulate!(
        callback, sim::Simulation; trunc = sim.alg.trunc, truncalg = sim.alg.truncalg,
        envalg = sim.alg.envalg, zeta = sim.alg.zeta, gaugeevery = sim.alg.gaugeevery, kwargs...
    )
    alg = DampedUpdate(; trunc, truncalg, envalg, zeta, gaugeevery)
    return simulate!(callback, sim, alg; kwargs...)
end

function simulate!(
        callback, sim::Simulation, alg::DampedUpdate; numsteps::Integer,
        maxshots::Integer = numsteps, maxtime::Float64 = Inf, breaktol::Float64 = 0.0,
        verbose::Bool = true
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
        timestep!(sim.state, sim.weights, sim.decomposition, alg)
        step += 1
        alg.gaugeevery > 0 && step % alg.gaugeevery == 0 && gaugefix!(sim.state, sim.weights)

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

    # The gauge drifts between fixes, so a run that stops mid-interval would hand back a drifted state.
    alg.gaugeevery > 0 && step % alg.gaugeevery != 0 && gaugefix!(sim.state, sim.weights)

    info.finished = true
    return sim
end
