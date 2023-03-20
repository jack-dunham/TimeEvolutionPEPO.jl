# The tensors that form the boundary 
# CTMRG, VUMPS, FPCM, FPCM_CTMRG, 
abstract type AbstractBoundary end
# The paramaters of the contraction boundary 
abstract type AbstractBoundaryParameters end
# The state of an algorithm of type `Alg`
abstract type AbstractBoundaryState{Alg<:AbstractBoundary} end

@doc raw"""
    ConvergenceInfo

A mutable struct that holds information about the convergence progress of an algorithm
# Fields
- `converged::Bool`: a boolean stating if an algorithm has converged or not
- `error::Float64`: stores the current error
- `iterations::Int`: stores the number of interations so far
- `finished::Bool`: whether or not the algorithm has finished running
"""
mutable struct ConvergenceInfo
    converged::Bool
    error::Float64
    iterations::Int
    finished::Bool
end

ConvergenceInfo() = ConvergenceInfo(false, Inf, 0, false)

@doc raw"""
    BoundaryParameters{S<:IndexSpace} <: AbstractBoundaryParameters

Stores the parameters of a boundary algorithm.

# Fields
- `bondspace::IndexSpace`: the vector space of the bond indices of the boundary 
    tensors. Unrelated to the bond space of the matrix product operator to be contracted by
    the algorithm. 
- `verbosity::Int`: the verbosity level logging during the algorithm's progress. A value of
    `0` denotes no logging, with increasing integers representing increasingly verbose 
    logging.
- `maxiter::Int`: maximum number of iterations before the algorithm must terminate.
- `tol::Float64`: error tolerence. The algorithm terminates after the error drops below this
    value, even if the number of iterations is less than `maxiter`.
"""
struct BoundaryParameters <: AbstractBoundaryParameters
    bondspace::IndexSpace
    verbosity::Int
    maxiter::Int
    tol::Float64
end

@doc raw"""
    BoundaryParameters(bondspace; verbosity=0, maxiter=100, tol=1e-8)

Constructor for the `BoundaryParameters` type, with defaults. Note, the bondspace of the
boundary must always be set by the user.
"""
function BoundaryParameters(bondspace; verbosity=0, maxiter=100, tol=1e-8)
    return BoundaryParameters(bondspace, verbosity, maxiter, tol)
end

function BoundaryParameters(P::InfinitePEPO)
    χ = westbond(P[1, 1])
    bondspace = χ ⊕ χ
    return BoundaryParameters(bondspace, 0, 100, 1e-10)
end

@doc raw"""
    BoundaryState{Alg<:AbstractBoundary} <: AbstractBoundaryState{Alg}

A struct representing the state of boundary algorithm of type `Alg`. The paramater `Alg`
can be one of `VUMPS`, `CTMRG`, `FPCM`, `FPCM_CTMRG`, or a user-defined algorithm. 

# Fields
- `data::Alg`: the tensors that compose the boundary, in addition to algorithm specific 
    data
- `mpo::InfMPO`: the matrix product operator whos boundary tensors are stored in `data`
- `initial::Alg`: the tensors used as the starting point of the algorithm
- `info::ConvergenceInfo`: information about the covergence progress of the algorithm
- `param::BoundaryParameters`: the paramaters used in the algorithm
"""
struct BoundaryState{Boundary<:AbstractBoundary} <: AbstractBoundaryState{Alg}
    data::Boundary
    mpo::InfiniteMPO
    initial::Boundary
    info::ConvergenceInfo
    param::BoundaryParameters
end

function calculate!(::AbstractBoundaryState) end

function reset!(b::BoundaryState)
    reset!(b.info)
    return b
end

@doc raw"""
    reset!(conv::ConvergenceInfo)

Resets `conv` to an initital state. That is, sets fields `converged`, `error`, and
`iterations` to `false`, `Inf`, and `0` respectively.
"""
function reset!(conv::ConvergenceInfo)
    conv.converged = false
    conv.error = Inf
    conv.iterations = 0
    conv.finished = false
    return conv
end

include("vumps/vumps.jl")

function initialise_tensors(
    ::Type{Alg}, m::InfiniteMPO, param::AbstractBoundaryParameters
) where {Alg<:AbstractBoundary} end

function initialise_state(
    ::Type{Alg}, m::InfiniteMPO, param::AbstractBoundaryParameters
) where {Alg<:AbstractBoundary}
    tensors = initialise_tensors(Alg, m, param)
    return initialise_state(tensors, m, param)
end

function initialise_state(
    tensors::Alg, m::InfiniteMPO, param::AbstractBoundaryParameters
) where {Alg<:AbstractBoundary}
    return BoundaryState{Alg}(tensors, m, tensors, ConvergenceInfo(), param)
end
