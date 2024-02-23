module PEPOKit

import TensorKit.AdjointTensorMap
import TensorKit.IndexTuple

using Combinatorics
using InfiniteTensorContractions
using KrylovKit
using LinearAlgebra
using MacroTools
using Random
using TensorKit
using TupleTools

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
export LocalUpdate, GlobalUpdate
export SU, NTU, FET, VGU
export Simulation, simulate!, simulate

export dynamicalmpo, MPO

export LindbladEquation, HamiltonianTerm, LongRangeTerm, LindbladTerm

export DensityMatrix, ReducedDensityMatrix
export trace, partialtrace
export expval, negativity, correlator

export ham, lin

include("utils.jl")
include("tensorkitutils.jl")
include("linalg.jl")
include("convergenceinfo.jl")
include("abstractlattice.jl")

## MODEL AND EQUATION
include("model/abstractterm.jl")
include("model/hamiltonianterm.jl")
include("model/longrangeterm.jl")
include("model/lindbladterm.jl")
include("model/equation.jl")
include("model/mpo.jl")

## PEPO OBJECTS
include("pepo/weights.jl")
include("pepo/abstractpepo.jl")
include("pepo/pepo.jl")

### UPDATABLE PEPOS
include("pepo/abstractupdatablepepo.jl")
include("pepo/localpepo.jl")
include("pepo/globalpepo.jl")

### OTHER
include("pepo/mcf.jl")
include("pepo/hilbertschmidt.jl")

include("contract/vumps.jl")
include("contract/ctmrg.jl")

include("truncation/wtg.jl")
include("truncation/clu/clu.jl")
include("truncation/truncation.jl")
include("truncation/isometries.jl")

include("evolution/abstractevolution.jl")
include("evolution/dynamicalmap.jl")
include("evolution/timeevolution.jl")

include("simulation.jl")
include("densitymatrix.jl")

# include("dev/tno_evolve.jl")
# include("dev/longrange.jl")
# include("dev/beliefpropagation.jl")

end # module
