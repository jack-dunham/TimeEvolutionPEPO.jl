module PEPOKit

using Random
using TensorKit
using LinearAlgebra
using KrylovKit
using MacroTools
using TrotterMacros
using InfiniteTensorContractions
using Combinatorics
using TupleTools

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
export ThermalState
export AbstractPEPO, PEPO, Weights
export Bond

export TEBD, DMPO
export Local, Global
export SU, NTU, FET, VGU
export Simulation, simulate!, simulate

export dynamicalmpo, MPO

export LindbladEquation, HamiltonianTerm, LongRangeTerm, LindbladTerm

export ham, lin

#generic 
include("model.jl")
include("utils.jl")
include("linalg.jl")
include("convergenceinfo.jl")
include("abstractlattice.jl")
# MPS

## PEPO OBJECTS
include("pepo/weights.jl")
include("pepo/abstractpepo.jl")
include("pepo.jl")

### UPDATABLE PEPOS
include("pepo/abstractupdatablepepo.jl")
include("pepo/localpepo.jl")
include("pepo/globalpepo.jl")

### MINIMAL CANONICAL FORM
include("pepo/mcf.jl")

include("contract/vumps.jl")
include("contract/ctmrg.jl")

include("physics/trace.jl")
include("physics/longrange.jl")

include("beliefpropagation.jl")

include("truncation/wtg.jl")
include("truncation/clu/clu.jl")
include("truncation/truncation.jl")
include("truncation/isometries.jl")

#models
include("physics/abstractevolution.jl")
include("physics/dynamicalmap.jl")
include("physics/timeevolution.jl")
# include("physics/tno_evolve.jl")
include("physics/simulation.jl")

include("physics/densitymatrix.jl")

# include("Models/Models.jl")

end # module
