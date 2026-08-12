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
   `UnitCellBonds`, `Bond`, `AxialPEPO` and the `Renormalization` glue go. One lattice abstraction.
2. **Lindblad stays on `InfinitePEPO`**, with the gate-application layer overridden locally: a
   Liouvillian superoperator acting jointly on ket⊗bra is a 4-leg gate, which PEPSKit's
   `_su_iter!`/`_apply_gate` cannot express. Everything around it — cluster extraction, weight
   absorption, bond rotation, reduced bond tensors, bond truncation, `SUWeight` — is reused.
3. **The TensorKit 0.14 → 0.17 migration is Phase 0 of this work, on this branch.** PEPSKit 0.8.1
   requires TensorKit `0.16.5, 0.17`; this package and TRGroups sit on 0.14.11, so nothing can be
   outsourced before the migration lands.
4. **The gaps stay here, rebuilt on PEPSKit primitives.** NTU becomes another bond-environment
   source feeding `bond_truncate`; negativity, the term algebra and the FSM/cluster-MPO code stay.

## What gets deleted

`src/pepo/` (except product-state constructors), `src/edgelattice.jl`, `src/abstractlattice.jl`,
`src/contract/`, `src/gauge/`, `src/dev/vidalgauge.jl`, `src/dev/beliefpropagation.jl`,
`src/truncation/{simpleupdate,isometries,truncation,clu}.jl`, `src/sparsetensormap.jl`, the
`LocalStepper`/`AxialStepper`/`FullStepper` zoo, and most of `src/densitymatrix.jl`. That is on the
order of 6–7k lines, against roughly 1.5k lines of new adapter and Lindblad-update code.

`AxialPEPO` deserves a specific mention: PEPSKit sweeps the second axis by rotating the state
(`rotl90`), not by holding a permuted-transposed copy, so the `setaxis!` aliasing hazard disappears
with it.

## Order of work

- **Phase 0** — validate the current code against exact results first (done: `test/exact_benchmarks.jl`,
  seven benchmarks against dense linear algebra; four agree to machine precision, three revealed
  pre-existing defects listed below). Then migrate to TensorKit 0.17.
- **Phase 1** — introduce PEPSKit, convert state and bond weights to `InfinitePEPO`/`SUWeight`.
- **Phase 2** — outsource observables to `reduced_densitymatrix`/`correlator`/`correlation_length`.
- **Phase 3** — outsource thermal annealing to `SimpleUpdate` (this is a direct swap; PEPSKit's
  `purified` flag already covers both of our thermal conventions).
- **Phase 4** — the local Lindblad update built on PEPSKit primitives.
- **Phase 5** — rebuild NTU and `IterativeSU` as bond-environment sources; delete the rest.

Each of Phases 2–5 ends with two checks: the test files named for that phase pass, and the three
reference configurations reproduce their stored numbers within the recorded tolerance.

## Defects found while validating the current code

Recorded, not fixed, per instruction. Detail and remediation points are in the plan.

1. **The Hamiltonian in a `Liouvillian` has the wrong sign**: evolution is `dρ/dt = +i[H,ρ]`.
   Confirmed on one spin with `H = (h/2)Z` at `D=1` — reversing the sign of `H` reproduces the
   package to 4.4e-16. The thermal path is correct. Every Lindblad result so far corresponds to `-H`.
2. **The product-state constructor prepares `ρ⁴`** instead of `ρ` (`src/pepo/pepo.jl:100-114`).
   Invisible for pure states and for `ThermalState`, which is every example in the docs and tests.
3. **Bond dimension above the gate rank produces garbage**: exact zeros in the padded bond weights
   are inverted by simple update. A one-local model exact at `D=1,2` is wrong by 2.0 at `D=4` and
   72.5 at `D=8`. The documented `PEPO(rho, ℂ^2, ℂ^4)` example is in this regime for low-rank gates.

## Risks and things you should weigh

- **The one point where the test suite cannot be kept passing.** Phase 0 raises the TensorKit bound,
  which breaks TRGroups (pinned to 0.14.4) while it is still a dependency. Either TRGroups is
  migrated as well — work that is discarded once TRGroups is dropped, but only ~45 call sites, and it
  keeps every intermediate step runnable — or Phases 0 and 1 merge into a single stretch during which
  the package does not load and no test can run. The plan assumes TRGroups is migrated; say if you
  would rather accept the period with no working build.
- **Dependency weight.** PEPSKit pulls in MPSKit, MPSKitModels, Zygote, OptimKit, MatrixAlgebraKit
  and OhMyThreads. Load and precompile time will get materially worse than the current ~12 s.
- **Index conventions differ in three ways at once** (virtual-leg order, which legs are dual, and
  row/col vs col/row indexing), so the conversion layer is where silent errors will hide. It gets
  round-trip tests before anything depends on it.
- **NTU and `IterativeSU` may not survive.** If PEPSKit's `FullEnvTruncation` plus `bondenv_ctm`
  covers what NTU was for, deleting NTU is cheaper than porting it. That call is deferred to Phase 5
  and is yours to make.
- **PEPSKit's PEPO time evolution asserts a unit cell of at least 2×2**, which rules out the 1×1
  `SquareSymmetric` cells used in the current tests and docs examples. Checkerboard geometry maps
  onto PEPSKit's `bipartite` flag plus its symmetrization helpers; parity there is unproven and is
  the most likely source of scope creep.
