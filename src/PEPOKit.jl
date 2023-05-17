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

export Cotensor
export cj, conjugate

export σ0, σ1, σ2, σ3, σp, σm, pauli

# LatticeStyle
export Infinite, Finite, Periodic, Open
# Lattice
export Lattice
export OnLattice, OnUnitCell
# Matrix product state
export MatrixProductState
export MPS
# Matrix product Operator
export MPS, SingleMPS
# Environments
export FixedPoints

#generic 
include("utils.jl")
include("linalg.jl")
include("abstractlatticelike.jl")
include("lattice.jl")
include("onlattice.jl")
include("convergenceinfo.jl")
# MPS
include("mps/mps.jl")
include("mps/gauge.jl")
#

include("physics/trace.jl")

# CONTRACTION
include("contraction/abstractcontraction.jl")
## BOUNDARY ALGORITHMS
include("contraction/boundaries/boundary.jl")
### VUMPS
include("contraction/boundaries/vumps/tensormacros.jl")
include("contraction/boundaries/vumps/transfermatrix.jl")
include("contraction/boundaries/vumps/fixedpoints.jl")
include("contraction/boundaries/vumps/vumps.jl")
### CTMRG
include("contraction/boundaries/ctmrg/ctmrg.jl")
include("contraction/boundaries/ctmrg/init.jl")
include("contraction/boundaries/ctmrg/tensormacros.jl")

# include("truncation/truncation.jl")

#models
include("models/models.jl")

end # module
