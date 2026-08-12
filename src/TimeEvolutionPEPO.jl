module TimeEvolutionPEPO

# Mid-refactor onto PEPSKit (see PEPSKIT_REFACTOR_PLAN.md). Only the state and observable layers are
# live; every include commented out below awaits its PEPSKit-based replacement and is unloaded, not
# deleted. Do not assume any of it works against TensorKit 0.17.

using Accessors
using DocStringExtensions
using LinearAlgebra
using PEPSKit
using Random
using TensorKit
using TupleTools

using PEPSKit: EAST, NORTH, SOUTH, WEST

const PEPOKit = TimeEvolutionPEPO

export PEPOKit

## PEPSKit reexports
export InfinitePEPO, InfinitePartitionFunction, SUWeight, CTMRGEnv
export leading_boundary, initialize_ctmrg_environment, RandomInitialization
export reduced_densitymatrix, correlation_length
#

export PAULI, LADDER, SPIN1
export cj, conjugate

export ThermalState, bloch, productpepo, bondentropy

export DensityMatrix, partialtrace, expval, tracenetwork, isconverged

export Hamiltonian, Liouvillian, Dissipator, Ising
export AbstractTerm, AbstractSingleTerm, AbstractSumOfTerms
export LocalOp, ExpOp, GaussOp, GaussOp2d, PowOp, PowOp2d
export dissipator, oneterms, isaxial, isexpandable

export generator, bondpropagator, fusephysical, vectorisedidentity

include("linalg.jl")
include("state.jl")
include("observables.jl")

## Term algebra: matrix-level code, carried over unchanged
include("model/abstractterm.jl")
include("model/terms/localop.jl")
include("model/terms/linop.jl")
include("model/terms/expop.jl")
include("model/terms/gaussop.jl")
include("model/terms/powop.jl")
include("model/terms/nnnop.jl")
include("model/hamiltonian.jl")
include("model/liouvillian.jl")
include("model/ising.jl")

include("propagator.jl")

# Awaiting replacement, in the order the plan takes them:
#
# include("utils.jl")
# include("tensorkitutils.jl")
# include("convergenceinfo.jl")
# include("abstractlattice.jl")
# include("edgelattice.jl")
# include("updatestyle.jl")
# include("sparsetensormap.jl")
#
# include("pepo/weights.jl")
# include("pepo/abstractpepo.jl")
# include("pepo/pepo.jl")
# include("pepo/axialpepo.jl")
# include("pepo/hilbertschmidt.jl")
#
# include("model/abstractaxis.jl")
#
# include("contract/vumps.jl")
# include("contract/ctmrg.jl")
#
# include("gauge/abstractgauge.jl")
# include("gauge/wtg.jl")
# include("dev/mcf.jl")
#
# include("truncation/clu/clu.jl")
# include("truncation/truncation.jl")
# include("truncation/isometries.jl")
# include("truncation/iterative.jl")
#
# include("evolution/abstractevolution.jl")
# include("evolution/timeevolution.jl")
#
# include("truncation/simpleupdate.jl")
#
# include("model/trotterlayer.jl")
# include("model/model.jl")
# include("model/equation.jl")
# include("model/mpo.jl")
# include("model/pepso.jl")
# include("model/fsm.jl")
# include("model/heisenberg.jl")
# include("model/longrangeheisenberg.jl")
# include("model/axialheisenberg.jl")
#
# include("evolution/dynamicalmap.jl")
#
# include("simulation.jl")
# include("densitymatrix.jl")
#
# include("dev/vidalgauge.jl")

end # module
