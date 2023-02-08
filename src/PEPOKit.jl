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

# TensorStyle
export Generic, NotTensor
export BondMatrix, MPS, MPO, PEPO, LeftEnv, RightEnv, TruncMetric
# LatticeStyle
export Infinite, Finite, Periodic, Open
# Lattice
export OnLattice, OnUnitCell
# Matrix product state
export MatrixProductState
export InfiniteMPS
# Matrix product Operator
export MatrixProductOperator
export InfiniteMPO
export MultilineInfiniteMPS
# Environments
export LeftEnvironment, RightEnvironment

#generic 
include("utils.jl")
include("linalg.jl")
include("tensorlattice.jl")
#tensors
include("mpo.jl")
include("mps.jl")
include("states/states.jl")
include("environments.jl")
include("transfermatrix.jl") # What is this for? Fixed points?
# include("symmetrise.jl")

#algorthims
include("tensors/tensors.jl")
include("truncation/truncation.jl")

include("gauge.jl")

#physics api
include("physics/physics.jl")

#models
include("models/models.jl")

end # module
