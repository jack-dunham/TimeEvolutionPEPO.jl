abstract type AbstractNeighbourhood end

const AbstractEnvironment = Union{AbstractBoundary,AbstractNeighbourhood}

struct NNTU <: AbstractNeighbourhood end

# Metrics can be constructed from Boundary or Neighbourhood.
function metric(P, env::AbstractEnvironment) end
# Traces require a Boundary, not a Neighbourhood. 
function trace(P, env::AbstractBoundary) end
