module TimeEvolutionPEPO

import TensorKit.AdjointTensorMap
import TensorKit.IndexTuple
using Accessors
using Combinatorics
using DocStringExtensions
using KrylovKit
using LinearAlgebra
using Logging
using Random
using TensorKit
using TensorRenormalizationGroups
using TupleTools
using VectorInterface

const TRGroups = TensorRenormalizationGroups
const PEPOKit = TimeEvolutionPEPO

import TensorRenormalizationGroups: AbsTen, SquareSymmetric, TenAbs

export PEPOKit

## Essential TRGroups reexports
export UnitCell, Square, SquareSymmetric
export VUMPS, CTMRG
export tensortype
#

export delta

export cj, conjugate

export PAULI, LADDER, SPIN1

# PEPO
export UnitCellBonds
export ThermalState

export AbstractPEPO, PEPO
export expandspace, subpepo, unsymmetrise, bondweights, bondentropy, physicalspace
export Bond

export TEBD, AxialTEBD, GlobalTEBD, ClusterMPO, GlobalTEBD, ClusterPEPO
export LocalUpdate, GlobalUpdate
export SimpleUpdate, SVDU, NTU, FET, IterativeSU

export Simulation, SimulationInfo
export simulate!, updatetimestep!
export quantumstate, timestep, trotterorder, simulationinfo

export EVB, DZM, WTG, MCF, NoGauge

export clustermpo, MPO

export Hamiltonian, Liouvillian, Dissipator, TrotterLayer, Model, Ising

export isaxial, isexpandable, dissipator

export AbstractTerm, AbstractSingleTerm, AbstractSumOfTerms
export AbstractAxialTerm, AbstractRadialTerm, AbstractManhattanTerm
export LocalOp, ExpOp, GaussOp, GaussOp2d, PowOp, PowOp2d

export AbstractDensityMatrix, DensityMatrix, PurifiedDensityMatrix
export trace, partialtrace
export expval, negativity, correlator, correlationlength

export permutebasis

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
include("dev/mcf.jl")

include("truncation/clu/clu.jl")
include("truncation/truncation.jl")
include("truncation/isometries.jl")
include("truncation/iterative.jl")

include("evolution/abstractevolution.jl")
include("evolution/timeevolution.jl")

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

# using PrecompileTools: @setup_workload, @compile_workload
#
# @setup_workload begin
#     X, Y, Z = PAULI
#     @compile_workload begin
#         warnlogger = ConsoleLogger(stderr, Logging.Warn)
#         oldlogger = global_logger(warnlogger)
#
#         model = Model(1.5 / 4 * LocalOp(Z, Z) + Dissipator(0.5 * (Y - im * Z)))
#
#         sim = Simulation(
#             PEPO{SquareSymmetric}((0, 0, -1), fill(ℂ^2, 2, 2), ℂ^1),
#             model,
#             TEBD();
#             timestep=1e-1,
#         )
#
#         simulate!(sim; numsteps=1)
#
#         dmc = DensityMatrix(quantumstate(sim), CTMRG(; bonddim=1))
#         dmv = DensityMatrix(quantumstate(sim), VUMPS(; bonddim=1))
#         dmp = PurifiedDensityMatrix(quantumstate(sim), VUMPS(; bonddim=1))
#
#         # `partialtrace` calls a generated function, so compile the common use cases
#         partialtrace(dmc, (1, 1))
#         partialtrace(dmc, (1, 1), (2, 1))
#         partialtrace(dmc, (1, 1), (1, 2))
#         partialtrace(dmc, (1, 1), (2, 2))
#
#         partialtrace(dmv, (1, 1))
#         partialtrace(dmv, (2, 1))
#
#         partialtrace(dmp, (1, 1))
#
#         global_logger(oldlogger)
#     end
# end

end # module
