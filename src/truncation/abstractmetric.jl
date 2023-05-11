abstract type AbstractEnvironment end

abstract type AbstractBoundary <: AbstractEnvironment end
abstract type AbstractNeighbourhood <: AbstractEnvironment end

struct VUMPS <: AbstractBoundary end
struct CTMRG <: AbstractBoundary end
struct FPCM <: AbstractBoundary end

struct NTU <: AbstractNeighbourhood
    T
    B
    LC
    RC
end

struct NNTU <: AbstractNeighbourhood end

# Metrics can be constructed from Boundary or Neighbourhood.
function metric(P, env::AbstractEnvironment) end
# Traces require a Boundary, not a Neighbourhood. 
function trace(P, env::AbstractBoundary) end
