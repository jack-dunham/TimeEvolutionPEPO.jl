"""
    SimulationInfo

A mutable struct that holds information about the convergence progress of a simulation
# Fields
- `isconverged::Bool`: whether the simulation has convergeced to a steady state
- `convergence::Float64`: stores the current numerical convergence
- `iterations::Int`: stores the number of interations so far
- `finished::Bool`: whether or not the algorithm has finished running
"""
@kwdef mutable struct SimulationInfo
    isconverged::Bool = false
    convergence::Float64 = Inf
    iterations::Int = 0
    finished::Bool = false
end

isconverged(conv::SimulationInfo) = conv.isconverged
convergence(conv::SimulationInfo) = conv.convergence
numiter(conv::SimulationInfo) = conv.iterations
isfinished(conv::SimulationInfo) = conv.finished

function reset!(conv::SimulationInfo)
    conv.finished = false
    return conv
end
