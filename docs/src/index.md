# TimeEvolutionPEPO.jl

TimeEvolutionPEPO.jl simulates the time evolution of an open quantum system represented by an infinite
PEPO ansatz, covering both Lindblad real-time evolution and thermal-state annealing under one
interface.

The state, its bond weights and the boundary contractions are
[PEPSKit.jl](https://github.com/QuantumKitHub/PEPSKit.jl) objects: an `InfinitePEPO` holds the
density-matrix tensors, an `SUWeight` the bond weights, and a `CTMRGEnv` the boundary environment.
What this package adds on top is the open-system layer — the term algebra that builds a Liouvillian,
its Suzuki-Trotter decomposition into two-site propagators, and a bond update that applies a
propagator to the ket and the bra of both sites at once.

## Package features

This package may be of use if you wish to study quantum many-body systems that are

1. defined on a shift-invariant two-dimensional square lattice,
2. evolving according to the Lindblad equation, or described by a thermal density matrix at finite
   temperature.

Interactions are built from a term algebra (`LocalOp`, `Ising`, `Dissipator`, and the decaying terms
`ExpOp`, `GaussOp`, `PowOp`) which sums into a `Hamiltonian` or a `Liouvillian`. Terms are grouped
into Suzuki-Trotter layers, where several terms in one layer are exponentiated together rather than
Trotterised apart.

!!! note
    Only nearest-neighbour terms currently reach the evolution. `trotterise` turns each layer into a
    two-site propagator on a bond, and the long-range terms — `ExpOp`, `GaussOp`, `PowOp` and their
    two-dimensional forms — have no route to one yet. They can be constructed and manipulated, but
    not evolved.

For Lindblad evolution, dissipation is on-site only.

Bonds are truncated either by a bare SVD against the bond weights (simple update), or variationally
against a CTMRG bond environment using PEPSKit's `FullEnvTruncation` or `ALSTruncation`. The gauge can
be fixed by trivial simple update (`SUGauge`) or from a belief-propagation fixed point (`BPGauge`).
