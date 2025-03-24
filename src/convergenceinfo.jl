"""
    SimulationInfo

A mutable struct that holds information about the convergence progress of a simulation
# Fields
- `converged::Bool`: whether the simulation has converged to a steady state
- `convergence::Float64`: stores the current numerical convergence
- `iterations::Int`: stores the number of interations so far
- `finished::Bool`: whether or not the algorithm has finished running. This is always `true`
    if `simulation` exists normally.
- `simtime::Float64`: the total simulation time so far
- `walltime::Float64`: the total wall clock time so far
"""
@kwdef mutable struct SimulationInfo
    converged::Bool = false
    convergence::Float64 = Inf
    iterations::Int = 0
    finished::Bool = false
    simtime::Float64 = 0.0
    walltime::Float64 = 0.0
end

isconverged(conv::SimulationInfo) = conv.converged
convergence(conv::SimulationInfo) = conv.convergence
numiter(conv::SimulationInfo) = conv.iterations
isfinished(conv::SimulationInfo) = conv.finished

simtime(conv::SimulationInfo) = conv.simtime
walltime(conv::SimulationInfo) = conv.walltime


function reset!(conv::SimulationInfo)
    conv.finished = false
    return conv
end
