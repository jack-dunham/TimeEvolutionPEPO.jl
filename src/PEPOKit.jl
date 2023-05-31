module PEPOKit

using Random
using TensorKit
import TensorKit.IndexTuple, TensorKit.AdjointTensorMap
using LinearAlgebra
using KrylovKit
using MacroTools
using DataStructures: DefaultOrderedDict
using TrotterMacros
using Parameters

import Base.*

export delta

export cj, conjugate

export σ0, σ1, σ2, σ3, σp, σm, pauli

# LatticeStyle
export Infinite, Finite, Periodic, Open
# Lattice
export AbstractLattice, Lattice, TransposedLattice, LatticeView
export AbstractOnLattice, OnLattice, OnUnitCell
# Matrix product state
export MatrixProductState
export MPS
# Matrix product Operator
export MPS, SingleMPS
# Environments
export FixedPoints
# PEPO
export AbstractPEPO, PEPO, TransposedPEPO, Weights
export Bond

#generic 
include("model.jl")
include("utils.jl")
include("linalg.jl")
include("abstractlatticelike.jl")
include("lattice.jl")
include("onlattice.jl")
include("convergenceinfo.jl")
# MPS
include("abstractpepo.jl")
include("mps/mps.jl")
include("mps/gauge.jl")
#

include("physics/trace.jl")
include("physics/liouvillian.jl")

# CONTRACTION
include("contraction/abstractcontraction.jl")
## BOUNDARY ALGORITHMS
include("contraction/boundaries/boundary.jl")
include("truncation/truncation.jl")
include("pepo.jl")
### VUMPS
include("contraction/boundaries/vumps/tensormacros.jl")
include("contraction/boundaries/vumps/transfermatrix.jl")
include("contraction/boundaries/vumps/fixedpoints.jl")
include("contraction/boundaries/vumps/vumps.jl")
### CTMRG
include("contraction/boundaries/ctmrg/ctmrg.jl")
include("contraction/boundaries/ctmrg/init.jl")
include("contraction/boundaries/ctmrg/tensormacros.jl")

include("truncation/abstractmetric.jl")
include("truncation/isometries.jl")
include("truncation/wtg.jl")
# include("truncation/truncation.jl")
include("truncation/clu/clu.jl")

#models
include("physics/trotterevolve.jl")
include("physics/tebd.jl")
include("models/models.jl")

end # module
