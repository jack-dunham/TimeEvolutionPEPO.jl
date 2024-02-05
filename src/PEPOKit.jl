module PEPOKit

using Random
using TensorKit
using LinearAlgebra
using KrylovKit
using MacroTools
using TrotterMacros
using InfiniteTensorContractions
using Combinatorics

import LowRankApprox: psvd
import TensorKit.IndexTuple, TensorKit.AdjointTensorMap

const ITC = InfiniteTensorContractions

## ITC exports
export Square
export VUMPS, CTMRG
#

export delta

export cj, conjugate

export σ0, σ1, σ2, σ3, σp, σm, PAULI, LADDER

# Lattice
export AbstractLattice, Lattice, SquareLattice
# PEPO
export AbstractPEPO, PEPO, Weights
export Bond

export Model

export TEBD, WII, TNO
export SU, NTU, FET, VGU
export GateLike, MPOLike
export initialize, simulate!, simulate

export @defmodel

export TimeEvolutionMPO, MatrixProductOperator

export LindbladEquation, HamiltonianTerm, LongRangeTerm, LindbladTerm

export ham, lin

#generic 
include("model.jl")
include("utils.jl")
include("linalg.jl")
include("convergenceinfo.jl")
include("abstractlattice.jl")
include("physics/abstractmap.jl")
# MPS
include("abstractpepo.jl")
include("pepo.jl")
#

include("contract/vumps.jl")
include("contract/ctmrg.jl")

include("physics/trace.jl")
include("physics/longrange.jl")

include("beliefpropagation.jl")

include("truncation/abstractopt.jl")
include("truncation/wtg.jl")
include("truncation/clu/clu.jl")
include("truncation/truncation.jl")
include("truncation/isometries.jl")

#models
include("physics/evolution.jl")
include("physics/tebd_evolve.jl")
include("physics/tno_evolve.jl")
include("physics/dynamicalmap.jl")

# include("Models/Models.jl")

end # module
