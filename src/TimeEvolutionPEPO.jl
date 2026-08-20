module TimeEvolutionPEPO

# Mid-refactor onto PEPSKit. Several files below hold their PEPSKit-based replacement under the name of
# the code they superseded, so a path is no indication of its contents. Everything under src/ is loaded.

using Accessors
using DocStringExtensions
using LinearAlgebra
using PEPSKit
using Random
using TensorKit
using TupleTools

using PEPSKit: EAST, NORTH, SOUTH, WEST
using PEPSKit: PEPOTensor, GenericMPSTensor, absorb_weight
using MatrixAlgebraKit: TruncationStrategy

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

export DensityMatrix, PurifiedDensityMatrix, partialtrace, expval, tracenetwork, isconverged
export correlationlength, correlator, negativity

export Hamiltonian, Liouvillian, Dissipator, Ising
export AbstractTerm, AbstractSingleTerm, AbstractSumOfTerms
export LocalOp, ExpOp, GaussOp, GaussOp2d, PowOp, PowOp2d, J1J2SymmetricOp
export dissipator, oneterms, isaxial, isexpandable
export TrotterLayer, scalelayer
export Model, distribute!

export generator, bondpropagator, fsmpepso
export j1j2model, fsmheisenberg
export InfinitePEPSO, LocalTruncation, compress
export reducedbond_first, reducedbond_last, undo_reducedbond_first, undo_reducedbond_last
export applypropagator, truncatebond, fusephysical
export FullEnvTruncation, ALSTruncation, BondEnvironment, bondenvironment
export bondstep!, axissweep!, timestep!
export trotterise

export Simulation, SimulationInfo, simulate!
export quantumstate, bondweights, timestep, simulationinfo, weightdistance
export isconverged, convergence, numiter, isfinished, simtime, walltime

export gaugefix!, bpenvironment
export SUGauge, BPGauge, BPEnv, BeliefPropagation, gauge_fix

include("linalg.jl")
include("pepo/pepo.jl")
include("densitymatrix.jl")

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
include("model/trotterlayer.jl")
include("model/model.jl")

# `PowOp` and `GaussOp` fit a power law as a sum of exponentials with routines defined here; only the
# `fsm*` builders in this file need porting, and they are not called at load time.
include("model/longrangeheisenberg.jl")
# The finite-state machine that turns a long-range term into a PEPO. Its bond dimension is the state
# count; only `GaussOp2d` has a rule, so the axial `PowOp{ExpOp}` has no route through it.
include("model/fsm.jl")

include("model/equation.jl")
include("truncation/isometries.jl")
include("truncation/simpleupdate.jl")
include("evolution/dynamicalmap.jl")
include("convergenceinfo.jl")
include("simulation.jl")
include("gauge/abstractgauge.jl")

# Unfinished work that does not run against the refactor lives in src/dev/ and is not included: the
# NTU cluster metric, IterativeSU, WTG, the long-range PEPO builders, and the model recipes.

end # module
