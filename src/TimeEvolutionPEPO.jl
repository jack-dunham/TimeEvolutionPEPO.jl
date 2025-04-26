module TimeEvolutionPEPO

import TensorKit.AdjointTensorMap
import TensorKit.IndexTuple

using Combinatorics
using InfiniteTensorContractions
using KrylovKit
using LinearAlgebra
using Random
using TensorKit
using TupleTools
using Zygote
using OptimKit
using VectorInterface
using ChainRulesCore
using SparseArrayKit
using Accessors
using DocStringExtensions

const ITC = InfiniteTensorContractions
const PEPOKit = TimeEvolutionPEPO

import InfiniteTensorContractions: AbsTen
import InfiniteTensorContractions: TenAbs

export PEPOKit

## Essential ITC reexports
export UnitCell, Square, SquareSymmetric
export VUMPS, CTMRG
#

export delta

export cj, conjugate

export PAULI, LADDER

# PEPO
export UnitCellBonds
export ThermalState

export AbstractPEPO, PEPO, expandspace, subpepo, unsymmetrise, weights
export Bond

export TEBD, AxialTEBD, GlobalTEBD, DMPO, GlobalTEBD, TEPEPSO
export LocalUpdate, GlobalUpdate
export SU, SVDU, NTU, FET, IterativeSU
export Simulation, simulate!, quantumstate

export EVB, DZM, WTG, MCF, NoGauge

export dynamicalmpo, MPO

export Hamiltonian, Liouvillian, Dissipator, TrotterLayer, Model, Ising

export isaxial, isexpandable, dissipator

export AbstractTerm, AbstractSingleTerm, AbstractSumOfTerms
export AbstractAxialTerm, AbstractRadialTerm, AbstractManhattanTerm
export LocalOp, ExpOp, GaussOp, GaussOp2d, PowOp, PowOp2d

export DensityMatrix, PurifiedDensityMatrix
export trace, partialtrace
export expval, negativity, correlator, correlationlength

include("utils.jl")
include("tensorkitutils.jl")
include("linalg.jl")
include("convergenceinfo.jl")
include("abstractlattice.jl")
include("edgelattice.jl")
include("updatestyle.jl")

# include("sparsetensormap.jl")

## PEPO OBJECTS
include("pepo/weights.jl")
include("pepo/abstractpepo.jl")
include("pepo/pepo.jl")
include("pepo/axialpepo.jl")

### UPDATABLE PEPOS
# include("pepo/abstractupdatablepepo.jl")
# include("pepo/localpepo.jl")
# include("pepo/globalpepo.jl")

### OTHER
include("pepo/hilbertschmidt.jl")

include("model/abstractaxis.jl")

include("contract/vumps.jl")
include("contract/ctmrg.jl")

include("gauge/abstractgauge.jl")
include("gauge/wtg.jl")
include("gauge/mcf.jl")

include("truncation/clu/clu.jl")
include("truncation/truncation.jl")
include("truncation/isometries.jl")
include("truncation/iterative.jl")

include("evolution/abstractevolution.jl")
include("evolution/timeevolution.jl")
include("evolution/timestep.jl")

include("truncation/simpleupdate.jl")

## MODEL AND EQUATION
include("model/abstractterm.jl")
# Load these in this order
include("model/terms/localop.jl")
include("model/terms/linop.jl")
include("model/terms/expop.jl")
include("model/terms/gaussop.jl")
include("model/terms/powop.jl")
include("model/terms/nnnop.jl")
# Load these in this order
include("model/hamiltonian.jl")
include("model/liouvillian.jl")
include("model/ising.jl")
include("model/trotterlayer.jl")
include("model/model.jl")

include("model/equation.jl")
include("model/mpo.jl")
include("model/pepso.jl")
include("model/fsm.jl")

include("model/heisenberg.jl")
include("model/longrangeheisenberg.jl")
include("model/axialheisenberg.jl")

include("evolution/dynamicalmap.jl")

include("simulation.jl")
include("densitymatrix.jl")

# include("dev/tno_evolve.jl")
# include("dev/longrange.jl")
# include("dev/beliefpropagation.jl")
include("dev/vidalgauge.jl")

end # module
