# Spec: the Model layer on the PEPSKit architecture

## Problem

`trotterise` takes a term, or a vector of terms giving one Trotter layer each, and turns each layer
into a single two-site propagator applied to every bond of a sweep. That covers uniform models only.
The parked `Model`/`TrotterLayer` (`src/model/{model,trotterlayer}.jl`) carried a term *per site* and
a rule for spreading one-local terms across layers; neither survived the refactor.

Restoring site dependence is not only a container question. `axissweep!` applies one propagator to
every bond, and `twolocal` (`src/model/terms/localop.jl:168`) symmetrises a one-local term across a
bond's two endpoints, so a bond propagator currently cannot represent a field that differs between
its two sites.


## Decisions

**Terms are site-anchored.** `TrotterLayer.data[r, c]` is anchored at site `(r, c)`: its two-local
part is the coupling between `(r, c)` and its neighbour in the `+axis` direction — east for axis 1,
south for axis 2 — and its one-local part belongs to `(r, c)` alone. A `1×1` matrix means uniform and
broadcasts to any unit cell; any other size must equal the state's unit cell exactly, checked in
`Simulation`, so a 2×2 model against a 3×3 state is an error rather than a silent tiling.

The alternative, separate matrices for couplings and fields, removes the ambiguity about what an
index means but costs two containers and diverges from the parked API for no gain in expressiveness.

**One-local terms are placed per endpoint.** A new `generator` method takes the two endpoints'
one-local terms separately rather than symmetrising one term across the bond. Each site is touched
twice per axis sweep, so each occurrence carries `1/2`, and the halving `trotterise` already applies
across the two axes brings it to `1/4` per touch — one full application per time step. A uniform
model must therefore produce the same propagators as today.

**`push!` and `distribute!` are separate.** `push!` always appends a layer. `distribute!(model, term)`
spreads a term over the existing layers with a `1/nlayers` prefactor, and throws on a model with no
layers. The parked code triggered distribution by type — a bare `LocalOp{1}` or `Dissipator` was
distributed, the same term wrapped in a `Hamiltonian` was not — which made the behaviour depend on
something invisible at the call site, and the old manual had to document the distinction as a
caveat.

**`Model` keeps its type parameter.** `Model{Hamiltonian}` rejects a `Dissipator` or `Liouvillian`
with an error naming `Model{Liouvillian}` as the fix. The parked version raised this by catching a
`convert` `MethodError` and inspecting its arguments; the check is made directly instead.


## What this does not do

Expandable terms. `expandterm(::PowOp)` returns `ExpOp`s, which are long-range axial terms coupling
all pairs `k > j`, so an expanded layer has no more route to a two-site propagator than the
unexpanded one — `trotterise` throws either way. Whether a `PowOp` is expanded or passed through
whole only changes the result once the finite-state-machine builder that turns a long-range term into
a PEPO exists, so `expand` belongs with that work. `isexpandable` and `expandterm` are left untouched;
`trotterise` gains an error naming the term type it cannot exponentiate.

Second-order Trotter, and site-dependent *axes* (a layer still carries one axis for the whole
lattice).


## Order of work

1. `generator` gains the per-endpoint one-local method; the existing symmetric path is expressed
   through it so a uniform model is unchanged.
2. `TrotterLayer` and `Model`, with `push!`, `distribute!`, the type check and `show`.
3. `trotterise` accepts a `Model` and returns `(propagators::Matrix, axis)` per layer.
4. `axissweep!` indexes the propagator matrix with `mod1`; `Simulation` validates unit-cell sizes.
5. Documentation: `docs/src/man/models.md` and `docs/src/lib/models.md`.

Each step keeps the existing suite passing; the uniform path is the regression check throughout.


## Verification

The load-bearing test is a **staggered field at D = 1**: a 2×2 unit cell carrying a one-local
Lindbladian on one sublattice and nothing on the other. Every site then evolves independently, so
both sublattices are checked against dense `exp(tL)`. This pins the prefactor and which `kron` factor
of the propagator is the first site — the detail most likely to be wrong, and the one the existing
uniform benchmarks cannot see.

Alongside: `distribute!` over `n` layers reproducing `h/n` written into each layer by hand; the
`Model{Hamiltonian}` plus `Dissipator` error; a two-local coupling set to zero on some bonds leaving
those bonds unentangled; and `benchmarks_test.jl` passing unchanged.
