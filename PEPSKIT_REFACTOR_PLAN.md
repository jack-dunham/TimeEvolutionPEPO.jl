# Implementation plan: refactor onto PEPSKit.jl

Agent-facing execution detail for `PEPSKIT_REFACTOR_SPEC.md`. Branch `pepskit-refactor`.
PEPSKit reference: `~/.julia/dev/PEPSKit` (v0.8.1, branch `main`).

Conventions for every task below: use the `julia_eval` MCP tool, never `bash julia`. Format only
lines you touched (JuliaFormatter BlueStyle). ReTestItems discovers `test/*_test.jl` only. Do not
run `Pkg.test()` without asking; run the specific test files named in each task.

## 0. API mapping

| Current | PEPSKit replacement | Notes |
|:---|:---|:---|
| `PEPO`, `AbstractPEPO` | `InfinitePEPO` | 3rd array axis is the layer; must be 1 for evolution |
| `AxialPEPO`, `setaxis!`, `PermuteTranspose` | `rotl90`/`rotr90` on state, `SUWeight`, `SiteDependentTruncation` | delete outright; axis-2 sweeps rotate instead of holding a transposed copy |
| `UnitCellBonds`, `bondweights` | `SUWeight` | weights **outside** the tensors, see §1.2 |
| `Bond`, `bonds`, `eachbond`, `uniqueindices` | `Vector{CartesianIndex{2}}` site pairs, `InfiniteSquare`, `nearest_neighbours` | |
| `HilbertSchmidt` | `InfiniteSquareNetwork` | |
| `DensityMatrix`, `partialtrace` | `CTMRGEnv` + `leading_boundary`, `reduced_densitymatrix` | |
| `expval`, `correlator`, `correlationlength` | `expectation_value`, `correlator`, `correlation_length` | |
| `PurifiedDensityMatrix` | `contractions/correlator/pepo_purified.jl`, `pepo_1layer.jl` | 1-layer contraction is `Tr[ρO]` |
| `ThermalState` constructor | `infinite_temperature_density_matrix(H)` | returns D=1 identity PEPO |
| `SimpleUpdate`, `SVDU`, `trotterstep!` | `SimpleUpdate` (`purified` flag), `su_iter` | their `purified=false` is `exp(-Hdt/2) ρ exp(-Hdt/2)` |
| `FET`, `isometries`, `updatetensors` | `FullEnvTruncation`, `ALSTruncation`, `bond_truncate`, `bondenv_ctm` | |
| `WTG`, `simple_update_gauge!`, `src/dev/vidalgauge.jl` | `SUGauge` + `gauge_fix` | `SUGauge` *is* gauge fixing by trivial simple update |
| `MCF`, `src/dev/beliefpropagation.jl` | `BPGauge`, `BeliefPropagation`, `BPEnv`, `SUWeight(::BPEnv)` | |
| `truncdim`/`truncbelow`/`truncspace` args | `TruncationStrategy`: `truncrank`/`trunctol`/`truncspace`, `FixedSpaceTruncation`, `SiteDependentTruncation` | |
| `mpinv`, `safepinv` | `sdiag_pow` | |
| `_permute`, `permutedom`, `swap`, rotations | `utility/{util,indexing,rotations}.jl` | keep only what has no equivalent |
| `tebdtensor` exponentiation | `trotterize`, `gate_to_mpo` (Hamiltonians only) | Liouvillian gates stay local, §4 |

No PEPSKit equivalent — keep: `src/model/**` (term algebra, `Liouvillian`, `Dissipator`, `Ising`,
`fsm.jl`, `mpo.jl`, `pepso.jl`), `negativity`, NTU, `Simulation`/`SimulationInfo`/`simulate!`,
`bondentropy`.

## 1. Conversion rules (the correctness of every later phase depends on these)

### 1.1 Tensor layout

Current: `ket ⊗ bra' ← east ⊗ south ⊗ west' ⊗ north'` (`AbsTen{2,4}`).
PEPSKit `PEPOTensor`: `P ⊗ Pp ← N ⊗ E ⊗ S ⊗ W`, built as `f(T, P * Pp ← N * E * S * W)` with
`Sspaces = adjoint.(circshift(Nspaces, (-1,0,0)))` and `Wspaces = adjoint.(circshift(Espaces, (0,1,0)))`.

Three differences at once:

1. **Domain order** — ours `(E,S,W,N)`, theirs `(N,E,S,W)`: permute domain by `(4,1,2,3)`.
2. **Which legs are dual** — ours `W,N`; theirs `S,W`. After the permutation, `flip` the north and
   south legs.
3. **Site indexing** — ours `pepo[i,j]` with `i` the column (x) and `j` the row (y, increasing
   south); theirs `A[row, col, layer]`. So `pepo[i,j] ↔ A[j,i,1]`. Codomain `(ket, bra')` maps
   straight onto `(P, Pp)` for a single layer.

### 1.2 Bond weights

`SUWeight` requires each weight to be a **real, non-negative `DiagonalTensorMap`** with equal domain
and codomain. Layout: `wt[1,r,c]` on the x-bond between `[r,c]` and `[r,c+1]`; `wt[2,r,c]` on the
y-bond between `[r,c]` and `[r-1,c]` — i.e. the **north** bond, where `UnitCellBonds` stores the
**south** bond as `data[ind][2]`. Converting the y-weights therefore needs a one-row `circshift`
on top of the row/col transpose.

**The absorption convention differs and this is the likeliest source of a silent physics error.**
This package pre-absorbs `sqrt(w)` into the site tensors; PEPSKit keeps `state` tensors bare and
holds the weights in the `SUWeight` environment, absorbing and removing them per bond inside
`_su_iter!` (`absorb_weight(...; inv=true)` on the way out). Conversion into PEPSKit form must
therefore strip `sqrt(w)` from all four legs (`sdiag_pow(w, -1/2)`), and conversion back must
re-absorb it.

### 1.3 Task

Write `src/pepskit/convert.jl` with `to_pepskit(pepo::PEPO) -> (InfinitePEPO, SUWeight)` and
`from_pepskit(::InfinitePEPO, ::SUWeight) -> PEPO`, plus `test/convert_test.jl`:

- round trip `from_pepskit(to_pepskit(p)...) ≈ p` for `Square` 1×1, 2×2, 2×3, and a `U1Space` case;
- space and dual-ness assertions on every leg of the converted tensor;
- `⟨Z⟩` and `bondentropy` invariant across the round trip (catches a wrong `sqrt(w)` convention,
  which a pure round trip does not);
- transposed indexing checked explicitly: `to_pepskit(p)[1][j,i,1]` corresponds to `p[i,j]`.

This file is scaffolding. It is deleted at the end of Phase 5 once `PEPO` is gone.

## 2. Phase 0 — reference output, then TensorKit 0.17

### 0.1 Exact benchmarks (done; `test/exact_benchmarks.jl`, report in `test/reference/benchmarks.txt`)

Seven benchmarks compare the package against references computed independently with dense linear
algebra, so a disagreement locates an error in the package rather than a change in it. Results at
commit `ba2345c`:

| | subject | result |
|:---|:---|:---|
| B1 | one-local Lindblad, `D=1`, against `exp(t𝓛)` | agrees, 2.5e-15 |
| B2 | interacting Lindblad, `d⟨O⟩/dt` against the exact generator | disagrees, 2.30; **6.5e-6 with the Hamiltonian sign reversed** |
| B3 | unit trace, Hermiticity, positivity of the one-site RDM | agrees, 4.4e-16 |
| B4 | high-temperature limit of thermal annealing | agrees, 2.1e-4; measured/exact `zz` 1.000129, `x` 0.999793, so `β = t` |
| B5 | product-state constructor prepares the requested `ρ` | disagrees, 0.476; **1.7e-16 against `ρ⁴`** |
| B6 | Larmor precession, sign of `H` in a `Liouvillian` | disagrees, 1.13; **4.4e-16 with the sign reversed** |
| B7 | one-local model at bond dimension above the gate rank | disagrees: `D=1` 1.1e-15, `D=2` 2.1e-15, `D=4` 2.02, `D=8` 72.5 |
| B8 | `Ising` fast path against the generic path at zero coupling | agrees, 8.9e-16 (after the fix below; 1.13 before) |
| B9 | asymmetric two-site term on a three-period stripe, site ordering | agrees, 6.6e-5; 1.60 against the exchanged ordering |

These benchmarks, not trajectory comparisons, are the completion criterion for later phases: they
check against exact results rather than against the current implementation, so they remain valid
across the refactor. `test/generate_reference_output.jl` records trajectories for change detection
and is optional.

### Pre-existing defects found by task 0.1

1. **Reversed physical legs on the Trotter gate — fixed.** Evolution ran with `dρ/dt = +i[H,ρ]` and
   with `D[conj(L)]` instead of `D[L]`. `tebdtensor` built the vectorised superoperator as a matrix
   whose composite index runs ket-slowest (row major), then moved it into the eight-leg array with
   `copyto!`, which fills the fastest axis first — reversing each group of four legs, which both
   exchanges ket with bra and reverses the two sites.

   The two halves differ in severity. Exchanging ket with bra is a physics error with no symmetry
   excusing it. Reversing the two sites is a lattice inversion: applied to every bond of a
   translation-invariant model it is a relabelling, unobservable unless the state itself breaks
   inversion symmetry, and invisible outright for terms symmetric under exchanging the two sites —
   which is every term the tests and documentation use. B9 pins it with a three-period stripe, where
   a site's two neighbours along an axis hold different states; a checkerboard is not enough, since
   there both neighbours hold the same state.
   The governing convention is PEPSKit's `PEPOTensor : P ⊗ P´ ← N ⊗ E ⊗ S ⊗ W` — index 1 outgoing
   (ket, left-multiplied), index 2 incoming (bra, right-multiplied), as confirmed by
   `trace_physicalspaces` contracting `O[p p; N E S W]` and by `_apply_sitegate` applying `gate * a`
   to index 1 and `a * gate` to index 2. The vectorisations in `vectorise`/`thermalvectorise` already
   matched that ordering; only the conversion was wrong, so the fix replaces the `copyto!` with a
   reshape into the reversed leg order followed by `permutedims` back.

   Evidence: B6 moved from 1.13 to 4.4e-16 and its Hamiltonian-reversed figure from 4.4e-16 to 1.13;
   B2 from 2.30 to 2.8e-6; a jump operator with `conj(L) ≠ ±L` moved from matching `D[conj(L)]` to
   matching `D[L]` at 3.9e-15. B1, B3 and B4 were unchanged, as expected — their operators are
   insensitive to the reversal, which is why the defect survived.

   The `Ising` fast path had the same reversal at `src/model/heisenberg.jl:207`, where
   `TensorMap(L * op * L, space, space)` reshapes the matrix leg-1-fastest. B8 covers it and both
   conversions now go through `vectorised_tensormap` in `src/tensorkitutils.jl`.

   Two sites keep the defect and are deliberately untouched, both unreachable:
   `tebdtensor(model::Ising, p, method, dt)` at `src/model/heisenberg.jl:222` returns
   `TensorMap(data, space, D1 * D2')` with `data` never assigned, so it raises `UndefVarError` if
   called; and `tebdtensor(model::AbstractMatrix, …)` at `src/evolution/dynamicalmap.jl:401` is
   reached only through `timestepper`, which has no live callers.
2. **`pepositedata` prepares `ρ⁴`, not `ρ`** (`src/pepo/pepo.jl:100-114`). It writes the input matrix
   into all four directional tensors and contracts them in a chain through the physical index. Pure
   states are projectors and `ThermalState` is the identity, so every example in the docs and tests
   is unaffected; any mixed initial state is silently replaced by `ρ⁴/tr(ρ⁴)`. Fix in Phase 1, where
   the constructors are rewritten onto PEPSKit's product-state builders anyway.
3. **Bond dimension above the rank the gate can generate produces garbage** (B7). Padded bond weights
   contain exact zeros, which simple update inverts. The failure is silent and severe: a one-local
   model that is exact at `D=1,2` is wrong by 2.0 at `D=4` and by 72.5 at `D=8`. The documented
   example `PEPO(rho, ℂ^2, ℂ^4)` is in this regime for any gate of rank below 4. PEPSKit guards the
   same operation with `sdiag_pow(s, pow; tol)`, so Phase 3/4 adoption resolves it; until then, keep
   `D` at or below the gate rank reported by the compression log.

### Trajectory recording (optional, `test/generate_reference_output.jl`)

Add `test/generate_reference_output.jl`. It runs the three configurations below against the
unmodified `dev` code and writes one plain-text file per configuration under `test/reference/`,
committed to the branch. Each file holds one row per time step containing `⟨X⟩`, `⟨Z⟩`,
`correlationlength` and `bondentropy`. Each configuration fixes `Random.seed!` so that runs are
repeatable.

Later phases re-run the same script and compare the new numbers to the stored ones elementwise. That
comparison is the only available evidence that a phase changed the implementation without changing
the computed results.

The three configurations:

- **R1** — Lindblad `TEBD` + `SimpleUpdate`, the model from the tutorial (`1.5/4*LocalOp(Z,Z) +
  Dissipator(0.5*(Y - im*Z))`), 2×2 cell, `D=4`, `dt=0.025`, 40 steps.
- **R2** — thermal annealing on the same lattice (Hamiltonian only, no dissipator), 40 steps.
- **R3** — `AxialTEBD` + `FET` at small boundary bond dimension, 10 steps, covering the
  environment-based truncation path at low cost.

Then run the script twice on unmodified `dev` and record the largest elementwise difference between
the two invocations in `test/reference/tolerance.txt`. That figure is the threshold for every later
comparison: a deviation below it is run-to-run variation, a deviation above it is a change in the
computed results.

### 0.2 TRGroups migration — declined

TRGroups stays on TensorKit 0.14. Raising the bound therefore breaks it while it is still a
dependency, so task 0.3 and Phase 1 merge into a single stretch during which the package does not
load and no benchmark can run. Two consequences that shape the work:

- **Delete rather than port.** Anything the plan deletes later must not be migrated to TensorKit 0.17
  first, since that work would be done blind and then thrown away. NTU, FET, `IterativeSU`,
  `AxialPEPO`, the stepper zoo and the cluster/FSM code are all in that category.
- **The exit criterion is the benchmark suite.** The stretch ends when the package loads with no
  TRGroups dependency and B1, B3, B5, B6, B7, B9 run again. B2, B4 and B8 need the gate pipeline and
  come back later in Phase 4.

Dependency resolution was checked against the local PEPSKit clone and succeeds: TensorKit 0.17.1,
MPSKit 0.13.13, MatrixAlgebraKit 0.6.9, KrylovKit 0.10.4, TensorOperations 5.7.0, Zygote 0.7.12 on
Julia 1.12.5. Note KrylovKit moves 0.9.5 to 0.10.4, so it is a second migration surface.

### 0.2b (superseded) TRGroups to TensorKit 0.17

This work is discarded once TRGroups is dropped, but it keeps every later task runnable (~45 call
sites: 24 `TensorMap(`, 13 `tsvd`, 5 truncation, 3 orth). Same mechanical edits as 0.3. If it is
skipped, task 0.3 and Phase 1 merge into a single stretch during which the package does not load.

### 0.3 This package to TensorKit 0.17

Mechanical, file by file, `Project.toml` compat `TensorKit = "0.17"` plus `MatrixAlgebraKit`:

- `TensorMap(f, cod, dom)` → `f(T, cod ← dom)` (70 sites)
- `tsvd`/`tsvd!` → `svd_trunc`/`svd_trunc!`, `svd_compact` where untruncated (36 sites)
- `truncdim`/`truncbelow`/`truncspace` → `truncrank`/`trunctol`/`truncspace` strategies (23 sites)
- `leftorth`/`rightorth` → `left_orth`/`right_orth` (20 sites)
- `pinv` on diagonal weights → `sdiag_pow(w, -1)` (33 sites, mostly `src/truncation/isometries.jl`
  and `src/dev/vidalgauge.jl` — check which of those files are deleted later and port the minimum)

Verify: the package loads; `pepo_test.jl`, `trotterise_test.jl` and `densitymatrix_test.jl` pass; all
three reference configurations reproduce their stored numbers within the tolerance from 0.1. **Any
deviation above that tolerance at this stage is an error in the TensorKit migration, not an intended
change** — stop and fix it before continuing.

## 3. Phase 1 — introduce PEPSKit, swap the state

1. `Pkg.develop(path="~/.julia/dev/PEPSKit")`, compat `PEPSKit = "0.8"`.
2. Land `src/pepskit/convert.jl` and its tests (§1.3).
3. Re-express `PEPO`'s product-state constructors (Bloch vector, density matrix, `ThermalState`) as
   builders returning `InfinitePEPO` + `SUWeight` directly. `ThermalState` → check
   `infinite_temperature_density_matrix` covers it before writing anything.
4. Keep `PEPO` alive as a deprecated alias until Phase 5 so the suite keeps running.

Verify: `pepo_test.jl` + `convert_test.jl`.

## 4. Phase 2 — observables

Rebuild `src/densitymatrix.jl` (1360 lines → target under 300):

- `DensityMatrix(pepo, alg)` becomes a thin wrapper over `leading_boundary` + `CTMRGEnv`, keeping
  the current normalise-on-construction behaviour and the `isconverged` accessor.
- `partialtrace` → `reduced_densitymatrix`; keep the non-adjacent-index support, which is the one
  feature worth checking parity on.
- `expval`/`correlator`/`correlationlength` → `expectation_value`/`correlator`/`correlation_length`.
- `Tr[ρO]` for the Lindblad (non-purified) case goes through the 1-layer PEPO contractions in
  `algorithms/contractions/correlator/pepo_1layer.jl`; the purified/thermal case through
  `pepo_purified.jl`.
- `negativity` stays, rewritten on top of whatever `reduced_densitymatrix` returns.
- Delete `src/contract/{ctmrg,vumps}.jl` (`_metric`, `onelocalcontract`, `truncmetric!`).

Verify: `densitymatrix_test.jl` passes; the observable columns of reference configurations R1 and R2
reproduce within tolerance.

## 5. Phase 3 — thermal annealing onto `SimpleUpdate`

Direct swap: `SimpleUpdate(; trunc, imaginary_time=true, purified=…)` plus `TimeEvolver`/`timestep`.
Our two thermal conventions map onto their `purified` flag exactly (`true`:
`|ρ(t+dt)⟩ = exp(-Hdt/2)|ρ(t)⟩`; `false`: `ρ → exp(-Hdt/2) ρ exp(-Hdt/2)`).

`Simulation`/`simulate!` keeps its callback/shot/wall-clock behaviour and drives their iterator:
`Base.iterate(::TimeEvolver{<:SimpleUpdate})` yields an `SUState` with `iter`, `t`, `psi`, `env`.

Two constraints in `_timeevol_sanity_check` and `_get_dt` that the current tests and documentation
examples violate:

- unit cell must be **at least 2×2** — the current 1×1 `SquareSymmetric` tests and docs examples
  will trip this. Either move to 2×2 or bypass their check in our own driver.
- PEPO must have exactly one layer.

`SquareSymmetric`/checkerboard maps onto `alg.bipartite` plus `utility/symmetrization.jl`
(`ReflectDepth`, `Rotate`, `RotateReflect`). Verify against reference configuration R2 before
deleting the geometry code; this is the task most likely to grow beyond its estimate.

Delete on success: `src/truncation/simpleupdate.jl`, `src/gauge/**`, `src/dev/vidalgauge.jl`,
`src/dev/beliefpropagation.jl`.

## 6. Phase 4 — the local Lindblad update

This is the part with no upstream equivalent. Build it as a mirror of PEPSKit's `_su_iter!`
(`algorithms/time_evolution/simpleupdate.jl:80-117`), reusing everything except gate application.

**Why an override is needed.** Their `_su_iter!` loops `for gate_ax in 1:2`, applying the same
2-site `NNGate = AbstractTensorMap{T,S,2,2}` to the ket leg and then the bra leg, breaking after the
first when `purified`. A Liouvillian acts on `(ket₁,bra₁,ket₂,bra₂)` jointly and does not factorise
that way.

**Design.**

1. `src/evolution/liouvillegate.jl`: gate type carrying the vectorised superoperator as a 2-2 map on
   *fused* `ket⊗bra` physical spaces, so it satisfies the `NNGate` shape on the fused leg.
2. Replace `trotterise`/`broadcastmodel` with a builder emitting
   `LocalCircuit{LiouvilleGate}(lattice, [siteA, siteB] => gate, …)`. `LocalCircuit`'s operator type
   is a free parameter (`gates::Vector{Pair{Vector{CartesianIndex{2}}, O}}`), so no piracy is
   needed. Keep the existing exponentiation (`tebdtensor`, `Ising`, `ExpOp`, `PowOp`) and the
   `Model`/`TrotterLayer` semantics — the layer structure is this package's design, not PEPSKit's.
3. `src/evolution/lindblad_su.jl`: `_su_iter_lindblad!` doing, in order — `_get_cluster`,
   `_absorb_weight!`, `_nn_bondrev` + `_bond_rotation`, fuse each site's `(ket,bra)` into one
   physical leg, `bond_tensor_first`/`bond_tensor_last`, apply the fused gate, truncate via
   `bond_truncate(a, b, benv, alg)` (`FullEnvTruncation`/`ALSTruncation`) or `svd_trunc!` for plain
   simple update, unfuse, `undo_bond_tensor_*`, `_bond_rotation(…; inv=true)`,
   `absorb_weight(…; inv=true)`, then write back and `normalize!(…, Inf)`.
4. Own iterator (`LindbladEvolver`) rather than `TimeEvolver`, since `_get_dt` asserts real-time
   evolution requires an `InfinitePEPS`. Yield the same shape as `SUState` so `simulate!` does not
   care which evolver it is driving.

Verify: reference configuration R1 within the tolerance from 0.1, plus a new `test/lindblad_test.jl` checking one gate
application against a dense reference at `D=1` (exactly representable, so this is an equality test
up to floating point) plus trace preservation over 5 steps.

Delete on success: the stepper zoo in `src/evolution/abstractevolution.jl`,
`src/evolution/{timeevolution,dynamicalmap}.jl`, `src/updatestyle.jl`,
`src/model/abstractaxis.jl`'s `UpdateStyle` traits.

## 7. Phase 5 — the remainder

- NTU as a bond-environment source: build the neighbourhood environment as a `BondEnv`
  (`AbstractTensorMap{T,S,2,2}`) and hand it to `bond_truncate`, so `src/truncation/clu/` and most
  of `isometries.jl` go. **Decide first whether NTU is still wanted** given `FullEnvTruncation` +
  `bondenv_ctm`; deleting is cheaper than porting.
- `IterativeSU` likewise, or delete.
- Delete `src/{abstractlattice,edgelattice,sparsetensormap}.jl`, `src/pepo/**`,
  `src/truncation/{truncation,isometries}.jl`, `src/pepskit/convert.jl`, the `*hook!` block at the
  end of `src/simulation.jl`, and the dead `_timestep!`.
- Prune `src/{utils,tensorkitutils,linalg}.jl` against `PEPSKit.utility/**`.
- Rewrite `docs/src/man/{states,models,tutorial}.md` and `docs/src/interfaces.md`; the documented
  extension points (`updatetensors` methods A/B/C, `environment!`, `newtruncators`) cease to exist
  and are replaced by PEPSKit's `TruncationStrategy`/`BondEnv` interfaces.
- Update `CLAUDE.md`: the pipeline, conventions and dead-code sections all describe the old
  architecture.

## 8. Known hazards

- **`su_iter` deepcopies state and weights every iteration** (`simpleupdate.jl:126`), where the
  current code mutates in place. Expect a per-step allocation regression; measure before assuming it
  matters, and prefer `_su_iter!` on a pre-copied state in the Lindblad driver.
- The local PEPSKit clone is on `main`, ahead of the 0.8.1 release. Pin compat to `"0.8"` and
  re-check any internal function this plan names (`_su_iter!`, `_get_cluster`, `_bond_rotation`,
  `bond_tensor_first`) at the start of each phase — they are unexported internals and may move.
- A phase is complete only when three conditions hold: the test files it names pass, the reference
  configurations reproduce within tolerance, and the deletions listed for that phase have been made.
  Do not defer deletions to a later phase; a phase that leaves the superseded code path in place is
  not finished.
