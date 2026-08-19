# Spec: refactor TimeEvolutionPEPO onto PEPSKit.jl

Branch: `pepskit-refactor` (off `dev`). Companion document: `PEPSKIT_REFACTOR_PLAN.md` (execution
detail, not for review).

## Problem

Roughly two thirds of this package reimplements machinery that PEPSKit 0.8.1 now provides: the PEPO
ansatz itself, bond weights, simple update, full-environment truncation, gauge fixing, CTMRG and
VUMPS boundaries, reduced density matrices, correlators and correlation lengths. Keeping local
copies costs maintenance and diverges from a package that is actively developed and tested.

What PEPSKit does **not** provide, and what this package therefore exists for: Lindblad real-time
evolution of a vectorised density matrix (PEPSKit's PEPO evolution is imaginary-time only and
applies gates as `exp(-Hdt/2) ρ exp(-Hdt/2)`), the `Dissipator`/`Liouvillian` term algebra, NTU,
negativity, and the cluster-MPO/FSM construction.

## Decisions

1. **TRGroups is retired from this package.** State becomes `InfinitePEPO`; `UnitCell`,
   `UnitCellBonds`, `Bond`, `AxialPEPO` and the `Renormalization` glue leave the live tree. One
   lattice abstraction.
2. **Lindblad stays on `InfinitePEPO`**, with the gate-application layer overridden locally: a
   Liouvillian superoperator acting jointly on ket⊗bra is a 4-leg gate, which PEPSKit's
   `_su_iter!`/`_apply_gate` cannot express. Everything around it — cluster extraction, weight
   absorption, bond rotation, reduced bond tensors, bond truncation, `SUWeight` — is reused.
3. **The TensorKit 0.14 → 0.17 migration is Phase 0 of this work, on this branch.** PEPSKit 0.8.1
   requires TensorKit `0.16.5, 0.17`; this package and TRGroups sit on 0.14.11, so nothing can be
   outsourced before the migration lands.
4. **The gaps stay here, rebuilt on PEPSKit primitives.** NTU becomes another bond-environment
   source feeding `bond_truncate`; negativity, the term algebra and the FSM/cluster-MPO code stay.

## What leaves the live tree

Nothing is deleted. Code PEPSKit does not replace moves to `src/dev/` with its include commented out,
for a later refactor; code PEPSKit does replace moves there too, and stops being loaded only once the
replacement demonstrably works. The listing below is therefore what leaves the live tree, not what is
removed from the repository — and since most files mix replaced with unreplaced behaviour, they are
set aside whole, per subsystem, as each replacement lands.

`src/pepo/`, `src/edgelattice.jl`, `src/abstractlattice.jl`, `src/contract/`, `src/gauge/`,
`src/dev/vidalgauge.jl`, `src/truncation/{simpleupdate,isometries,truncation,iterative,clu}.jl`,
`src/sparsetensormap.jl`, the `LocalStepper`/`AxialStepper`/`FullStepper` zoo, and most of
`src/densitymatrix.jl` — on the order of 7.5k lines leaving the live tree, against roughly 1.5k lines
of new adapter and Lindblad-update code.

Of that, the parts with no PEPSKit counterpart move to `src/dev/` awaiting a rewrite rather than merely
superseded: `NTU` and its cluster metric under `truncation/clu/`, `IterativeSU`, `WTG`, the
product-state constructors, `bondentropy`, `negativity`, and the minimal canonical form already moved
in `src/dev/mcf.jl`.

`AxialPEPO` deserves a specific mention: PEPSKit sweeps the second axis by rotating the state
(`rotl90`), not by holding a permuted-transposed copy, so the `setaxis!` aliasing hazard disappears
with it.


## Order of work

- **Phase 0** — validate the current code against exact results first (done: `test/exact_benchmarks.jl`,
  nine benchmarks against dense linear algebra; seven agree, and three pre-existing defects were found,
  one of which is now fixed — see below). Then migrate to TensorKit 0.17.
- **Phase 1** — introduce PEPSKit, convert state and bond weights to `InfinitePEPO`/`SUWeight`.
- **Phase 2** — outsource observables to `reduced_densitymatrix`/`correlator`/`correlation_length`.
- **Phase 3** — outsource thermal annealing to `SimpleUpdate` (this is a direct swap; PEPSKit's
  `purified` flag already covers both of our thermal conventions).
- **Phase 4** — the local Lindblad update built on PEPSKit primitives.
- **Phase 5** — rebuild NTU and `IterativeSU` as bond-environment sources, and dispose of what remains.

Each of Phases 2–5 ends with two checks: the test files named for that phase pass, and the exact
benchmarks in `test/exact_benchmarks.jl` still agree. Those benchmarks check against dense linear
algebra rather than against the current implementation, so they stay valid across the refactor;
recorded trajectories would instead have preserved the leg reversal that Phase 0 found.

## Defects found while validating the current code

Recorded, not fixed, per instruction. Detail and remediation points are in the plan.

1. **Fixed.** The physical legs of the Trotter gate were reversed, so evolution ran with
   `dρ/dt = +i[H,ρ]` and with `D[conj(L)]` in place of `D[L]`. Root cause: `tebdtensor` copied the
   vectorised superoperator into its eight-leg array with `copyto!`, which fills the fastest axis
   first, whereas the composite index of the superoperator runs ket-slowest — reversing each group of
   four legs. PEPSKit fixes the convention this must obey: `PEPOTensor : P ⊗ P´ ← N ⊗ E ⊗ S ⊗ W`,
   index 1 outgoing (ket, left-multiplied), index 2 incoming (bra, right-multiplied). The vectorised
   expressions were already textbook-correct for that ordering; only the conversion was wrong.
2. **The product-state constructor prepares `ρ⁴`** instead of `ρ` (`src/pepo/pepo.jl:100-114`).
   Invisible for pure states and for `ThermalState`, which is every example in the docs and tests.
3. **Bond dimension above the gate rank produces garbage**: exact zeros in the padded bond weights
   are inverted by simple update. A one-local model exact at `D=1,2` is wrong by 2.0 at `D=4` and
   72.5 at `D=8`. The documented `PEPO(rho, ℂ^2, ℂ^4)` example is in this regime for low-rank gates.

## Risks and things you should weigh

- **There will be a stretch with no working build — decided.** Raising the TensorKit bound breaks
  TRGroups, which stays on 0.14, so tasks 0.3 and Phase 1 merge into one stretch during which the
  package does not load and no benchmark can run. TRGroups is deliberately not migrated. The
  mitigation is ordering: replace a subsystem, remove what it supersedes, and keep that stretch as short
  as possible rather than porting code that is on its way to `src/dev/`.
- **Dependency weight.** PEPSKit pulls in MPSKit, MPSKitModels, Zygote, OptimKit, MatrixAlgebraKit
  and OhMyThreads. Load and precompile time will get materially worse than the current ~12 s.
- **Index conventions differ in three ways at once** (virtual-leg order, which legs are dual, and
  row/col vs col/row indexing), so the conversion layer is where silent errors will hide. It gets
  round-trip tests before anything depends on it.
- **NTU and `IterativeSU` may move to `src/dev/` rather than be rebuilt.** If PEPSKit's `FullEnvTruncation`
  plus `bondenv_ctm` covers what NTU was for, leaving it in `src/dev/` is cheaper than porting it.
  That call is deferred to Phase 5 and is yours to make; either way the code remains in the
  repository.
- **PEPSKit's PEPO time evolution asserts a unit cell of at least 2×2**, which rules out the 1×1
  `SquareSymmetric` cells used in the current tests and docs examples. Checkerboard geometry maps
  onto PEPSKit's `bipartite` flag plus its symmetrization helpers; parity there is unproven and is
  the most likely source of scope creep.
