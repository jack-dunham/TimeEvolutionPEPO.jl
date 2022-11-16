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
    A::MultilineInfMPS
    FL::LeftEnv
    FR::RightEnv
end

const VUMPSState = BoundaryState{VUMPS}

function initialise_tensors(::Type{VUMPS}, P, L, param)
    nx,ny = size(L)

    cod_P = codomain(P)
    cod_L = codomain(L)

    spx = fuse(cod_P[1]*cod_L[1]*cod[2]')
    spy = fuse(cod_P[2]*cod_P[2]')

    A = MultilineInfMPS(rand, ComplexF64, spx * spy, spx * spy)
end

function initialise_tensors(::Type{VUMPS},
    M::InfMPO, param::BoundaryParameters
)
    χsp = param.bondspace
    # χsp = ⊕(fill(oneunit(physcod), χ)...)
    nx, ny = size(M)
    physcod, physdom = space(M, :n)

    A = MultilineInfMPS{nx,ny}(rand, ComplexF64, physdom, physcod, χsp)
    FL, FR = environments(A, M)

    return VUMPS(A, FL, FR)
end


function initialise_state(::Type{VUMPS},
    M::InfMPO, param::BoundaryParameters
)
    tensors = initialise_tensors(VUMPS, M, param)
    return BoundaryState{VUMPS}(tensors, M, tensors, ConvergenceInfo(), param, false)
end


include("alg.jl")
