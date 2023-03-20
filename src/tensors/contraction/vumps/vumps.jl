# Algorithm parameters

@doc raw"""
    VUMPS <: AbstractBoundary

A struct holding the boundary tensors obtained via the VUMPS algorithm.

# Fields
- `A::MultilineInfMPS`: The VUMPS boundary matrix product state. 
- `FL::LeftEnv:` The left environment of the transfer matrix.
- `RF::RightEnv:` The right environment of the transfer matrix.
"""
struct VUMPS <: AbstractBoundary
    A::MultilineInfiniteMPS
    FL::LeftEnvironment
    FR::RightEnvironment
end

const VUMPSState = BoundaryState{VUMPS}

function initialise_tensors(::Type{VUMPS}, P, L, param)
    nx, ny = size(L)

    cod_P = codomain(P)
    cod_L = codomain(L)

    spx = fuse(cod_P[1] * cod_L[1] * cod[2]')
    spy = fuse(cod_P[2] * cod_P[2]')

    return A = MultilineInfiniteMPS(rand, ComplexF64, spx * spy, spx * spy)
end

function initialise_tensors(::Type{VUMPS}, M::InfiniteMPO, param::BoundaryParameters)
    nx, ny = size(M)

    χsp = param.bondspace
    Dsp = @. getindex(domain(M), 2)

    c = Ref(χsp) .* Dsp

    data = Tuple(
        InfiniteMPS{nx}(TensorMap.(rand, numbertype(M), c[:, y], Ref(χsp))) for y in 1:ny
    )
    A = MultilineInfiniteMPS(data...)
    FL, FR = environments(A, M)

    return VUMPS(A, FL, FR)
end

function initialise_state(::Type{VUMPS}, M::InfiniteMPO, param::BoundaryParameters)
    tensors = initialise_tensors(VUMPS, M, param)
    return BoundaryState{VUMPS}(tensors, M, tensors, ConvergenceInfo(), param)
end

function initialise_state(b::BoundaryState{VUMPS}, M::InfiniteMPO)
    return BoundaryState{VUMPS}(b.data, M, b.data, ConvergenceInfo(), b.param)
end
function randomstate(b::BoundaryState{VUMPS}, M::InfiniteMPO)
    return initialise_state(VUMPS, M, b.param)
end

calculate!(v::VUMPSState) = vumps!(v)

include("alg.jl")
