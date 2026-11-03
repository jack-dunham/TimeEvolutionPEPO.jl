# Superseded tests

These target the API that PEPSKit replaced and no longer load. They are kept for reference, not
deleted, and their filenames drop the `_test` suffix so ReTestItems does not discover them.

- `pepo.jl`, `densitymatrix.jl`, `trotterise.jl` — the original suite, written against `PEPO`,
  `Simulation(pepo, model, TEBD(); …)` and the TRGroups boundary algorithms.
- `exact_benchmarks.jl` — the nine benchmarks as first run against the pre-refactor code, whose
  results are recorded in `../reference/benchmarks.txt` and in `PEPSKIT_REFACTOR_PLAN.md`. Their
  successors live in `../benchmarks_test.jl` and in the per-subsystem test files.
- `generate_reference_output.jl` — trajectory recording, superseded by those benchmarks: they check
  against dense linear algebra rather than against the implementation, so they stay valid across the
  refactor, whereas recorded trajectories would have preserved the leg reversal Phase 0 found.

Also stale and never discovered even before the refactor: `../model.jl`, `../truncation.jl`,
`../isometries.jl`, `../simulation.jl`, `../fsm.jl`.
