# Model layer implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore site-dependent terms and Trotter-layer bookkeeping to the model layer, so a model can vary across the unit cell instead of being one term applied to every bond.

**Architecture:** A `TrotterLayer` holds a `Matrix` of terms anchored at unit-cell sites plus an axis; a `Model` holds a vector of layers. `trotterise` turns each layer into a `Matrix` of two-site propagators, and `axissweep!` indexes that matrix per bond. One-local terms are placed on the endpoint they belong to rather than symmetrised across the bond.

**Tech Stack:** Julia 1.12, TensorKit 0.17, PEPSKit 0.8.1, ReTestItems.

**Spec:** `MODEL_SPEC.md`

## Global constraints

- Work in the worktree `.worktrees/pepskit-refactor-agent`, branch `pepskit-refactor-agent`.
- Use the `julia_eval` MCP tool, **not** `bash julia`. Test env path: `.worktrees/pepskit-refactor-agent/test`.
- Only `test/*_test.jl` files are discovered by ReTestItems. Add `@testitem`s to existing files where one fits.
- Formatting is JuliaFormatter BlueStyle. Format only lines you touched; never reformat a file you did not otherwise edit.
- Comments: two lines maximum, impersonal, no first person. Name the concrete mechanism, never a vague label.
- Docstrings are for users. Implementation rationale goes in a code comment or the commit message, never in a docstring.
- Do not run the full suite (`Pkg.test`) — run the named test files only.
- A `PEPOTensor` is `P ⊗ P' ← N ⊗ E ⊗ S ⊗ W`; index 1 is the ket, index 2 the bra.
- In a two-site operator, `kron(A, B)` puts `A` on the **first** site. This is fixed by the passing benchmark B9 — do not change it.

---

### Task 1: Per-endpoint placement of one-local terms

The bond propagator currently symmetrises a one-local term across both endpoints via `twolocal`, so it cannot represent a field that differs between the two sites. Introduce a per-slot embedding and express the symmetric case through it, leaving uniform models bit-identical.

**Files:**
- Modify: `src/model/terms/localop.jl:168`
- Modify: `src/model/liouvillian.jl:36-38`
- Modify: `src/propagator.jl:26-44`
- Modify: `src/TimeEvolutionPEPO.jl` (export nothing new; `generator` is already exported)
- Test: `test/propagator_test.jl`

**Interfaces:**
- Consumes: `vectorise`, `thermalvectorise`, `oneterms`, `terms`, `dissipator`.
- Produces:
  - `onsite(f, slot::Int)` — returns `H -> 1/2 * f(kron(...))` embedding a one-site operator into the two-site space at `slot ∈ (1, 2)`.
  - `generator(anchor::AbstractTerm, neighbour::AbstractTerm, P::IndexSpace) -> Matrix{ComplexF64}`
  - `generator(model::AbstractTerm, P::IndexSpace)` unchanged in behaviour, now defined as `generator(model, model, P)`.

- [ ] **Step 1: Write the failing test**

Append to `test/propagator_test.jl`, inside the existing `@testitem`:

```julia
@testset "one-local terms are placed per endpoint" begin
    X, Y, Z = PAULI
    P = ComplexSpace(2)

    # Splitting a uniform model across its two endpoints must reproduce the symmetric path.
    uniform = 1.5 / 4 * LocalOp(Z, Z) + LocalOp(X) + Dissipator(0.5 * (Y - im * Z))
    @test generator(uniform, uniform, P) ≈ generator(uniform, P)

    # A field on the first site only must equal kron(X, I) in the vectorised generator, at half
    # weight, and must differ from the same field on the second site.
    first_only = zero(Hamiltonian) + LocalOp(X)
    empty = zero(Hamiltonian)
    g1 = generator(first_only, empty, P)
    g2 = generator(empty, first_only, P)
    @test !(g1 ≈ g2)

    # Both halves together are the symmetric term.
    @test g1 + g2 ≈ generator(first_only, first_only, P)
end
```

- [ ] **Step 2: Run it and confirm it fails**

```julia
using ReTestItems; runtests("test/propagator_test.jl")
```

Expected: `MethodError: no method matching generator(::Hamiltonian, ::Hamiltonian, ::ComplexSpace)`.

- [ ] **Step 3: Add the per-slot embedding**

In `src/model/terms/localop.jl`, immediately below the existing `twolocal` definition at line 168, add `onsite` and redefine `twolocal` through it. Keep the existing comment above `twolocal` unchanged — it is human-written.

```julia
onsite(f, slot::Int) = H -> 1 / 2 * f(slot == 1 ? kron(H, one(H)) : kron(one(H), H))
twolocal(f) = H -> onsite(f, 1)(H) + onsite(f, 2)(H)
```

- [ ] **Step 4: Hoist the dissipator vectorisers**

In `src/model/liouvillian.jl`, the two closures inside `vectorise(dissipator, locality)` are needed per-slot. Lift them to file scope above that function and have it call them:

```julia
_anticomm(L) = kron(L' * L, one(L)) + kron(one(L), transpose(L' * L))
_dissipatorvec(L) = kron(L, transpose(L')) - 1 / 2 * _anticomm(L)
```

Then inside `vectorise`, replace the `anticomm`/`vec_local` closure definitions with uses of `_dissipatorvec`: `f = locality == 1 ? _dissipatorvec : twolocal(_dissipatorvec)`, leaving the `locality` validation and the `isempty` check as they are.

- [ ] **Step 5: Rewrite `generator`**

Replace the body of `generator` in `src/propagator.jl:26-44` with the two-term form. Keep the existing docstring and extend its first line to name both arguments.

```julia
_vectorisation(::Hamiltonian) = thermalvectorise
_vectorisation(::Liouvillian) = vectorise
_vectorisation(t) = throw(ArgumentError("no generator defined for a model of type $(typeof(t))"))

function generator(anchor::AbstractTerm, neighbour::AbstractTerm, P::IndexSpace)
    d = dim(P)
    data = zeros(ComplexF64, d^4, d^4)

    vecf = _vectorisation(anchor)
    vecf === _vectorisation(neighbour) || throw(
        ArgumentError(
            "a bond cannot mix a thermal and a Lindblad generator: " *
                "got $(typeof(anchor)) and $(typeof(neighbour))"
        ),
    )

    # Two-local terms belong to the bond and come from the anchor; one-local terms belong to a site.
    length(terms(anchor)) > 0 && (data += sum(vecf, terms(anchor)))

    for (slot, site) in ((1, anchor), (2, neighbour))
        for h in oneterms(site)
            data += onsite(vecf(LocalOp{2}), slot)(Matrix{ComplexF64}(h))
        end
        jumps = dissipator(site)
        isnothing(jumps) && continue
        for L in terms(jumps)
            data += onsite(_dissipatorvec, slot)(L)
        end
    end

    return data
end

generator(model::AbstractTerm, P::IndexSpace) = generator(model, model, P)
```

- [ ] **Step 6: Run the test and the propagator suite**

```julia
using ReTestItems; runtests("test/propagator_test.jl")
```

Expected: PASS, all items.

- [ ] **Step 7: Run the exact benchmarks as the uniform regression**

```julia
using ReTestItems; runtests("test/benchmarks_test.jl")
```

Expected: PASS (11 tests). These check against dense linear algebra, so they catch a changed prefactor or a swapped slot.

- [ ] **Step 8: Commit**

```bash
git add src/model/terms/localop.jl src/model/liouvillian.jl src/propagator.jl test/propagator_test.jl
git commit -m "feat: place one-local terms per bond endpoint

generator takes an anchor and a neighbour term; the two-local coupling comes
from the anchor and each site's one-local part is embedded at its own slot by
onsite. twolocal is now the sum of the two slots, so a uniform model produces
the same generator as before."
```

---

### Task 2: `generator` for `Ising`

`Ising` is exported and documented but `generator` throws for it, so the type cannot be evolved. Give it a plain generator by expressing it as the equivalent `Hamiltonian`/`Liouvillian`. The low-rank exponentiation the no-longer-loaded `equation.jl` provided is **not** part of this task.

**Files:**
- Modify: `src/model/ising.jl` (append)
- Test: `test/propagator_test.jl`

**Interfaces:**
- Consumes: `generator(anchor, neighbour, P)` from Task 1.
- Produces: `_equation(t::Ising) -> Union{Hamiltonian, Liouvillian}`; `generator` accepts `Ising`.

- [ ] **Step 1: Write the failing test**

Append to `test/propagator_test.jl` inside the existing `@testitem`:

```julia
@testset "Ising reaches a generator" begin
    X, Y, Z = PAULI
    P = ComplexSpace(2)
    a = 1.5 / 4

    ising = Ising(a * LocalOp(Z, Z), [LocalOp(X)], Dissipator(0.5 * (Y - im * Z)))
    equivalent = a * LocalOp(Z, Z) + LocalOp(X) + Dissipator(0.5 * (Y - im * Z))

    @test generator(ising, P) ≈ generator(equivalent, P)
    @test size(bondpropagator(ising, P, 0.05)) == size(bondpropagator(equivalent, P, 0.05))
end
```

- [ ] **Step 2: Run it and confirm it fails**

```julia
using ReTestItems; runtests("test/propagator_test.jl")
```

Expected: `ArgumentError: no generator defined for a model of type Ising{Liouvillian}`.

- [ ] **Step 3: Implement**

Append to `src/model/ising.jl`:

```julia
# `Ising` stores its coupling as a single matrix with C == B; the equivalent term is a * C ⊗ C.
_equation(t::Ising{Hamiltonian}) = Hamiltonian((t.a * LocalOp(t.CB, t.CB),), t.oneterms)
function _equation(t::Ising{Liouvillian})
    return Liouvillian(Hamiltonian((t.a * LocalOp(t.CB, t.CB),), t.oneterms), t.dissipator)
end

generator(t::Ising, P::IndexSpace) = generator(_equation(t), P)
generator(a::Ising, b::Ising, P::IndexSpace) = generator(_equation(a), _equation(b), P)
```

Also add, for the route check Task 3 introduces — `terms(::Ising)` does not exist, and an `Ising`
coupling is nearest-neighbour by construction, so the check is vacuous for it:

```julia
_checkroute(::Ising) = nothing
```

Task 3 defines `_checkroute` in `src/trotterdecomposition.jl`; if Task 3 has not run yet, add this
line there as part of Task 3 instead and leave a note in your report.

If `Ising` is loaded before `Hamiltonian`/`Liouvillian`, move these definitions to the bottom of `src/propagator.jl` instead — check the include order in `src/TimeEvolutionPEPO.jl:61-77` before choosing.

- [ ] **Step 4: Run the test**

```julia
using ReTestItems; runtests("test/propagator_test.jl")
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/model/ising.jl test/propagator_test.jl
git commit -m "fix: give Ising a generator

Ising was exported and documented but generator threw for it, so the type could
not be evolved. It is expanded to the equivalent Hamiltonian or Liouvillian; the
low-rank exponentiation the unloaded equation.jl provided is not restored here."
```

---

### Task 3: Promote bare terms to layers

`trotterise([LocalOp(Z, Z), LocalOp(X, X)], …)` fails with `no method matching scalelocal(::LocalOp{2}, ::Float64)` — the multi-layer form only works if every layer is already a `Hamiltonian` or `Liouvillian`. Promote each layer on the way in.

**Files:**
- Modify: `src/trotterdecomposition.jl`
- Test: `test/propagator_test.jl`

**Interfaces:**
- Produces: `layerterm(t::AbstractTerm) -> Union{Hamiltonian, Liouvillian, Ising}`.

- [ ] **Step 1: Write the failing test**

Append to `test/propagator_test.jl` inside the existing `@testitem`:

```julia
@testset "bare terms are promoted to layers" begin
    X, Y, Z = PAULI
    P = ComplexSpace(2)

    @test length(trotterise(LocalOp(Z, Z), P, 0.1)) == 2          # one layer, both axes
    @test length(trotterise([LocalOp(X, X), LocalOp(Y, Y)], P, 0.1)) == 4
    @test length(trotterise([LocalOp(X, X) => 1, LocalOp(Y, Y) => 2], P, 0.1)) == 2

    # A bare Dissipator is one-local, so it must promote to a Liouvillian, not a Hamiltonian.
    @test first(first(trotterise(Dissipator(0.5 * (Y - im * Z)), P, 0.1))) isa AbstractTensorMap
end
```

- [ ] **Step 2: Run it and confirm it fails**

```julia
using ReTestItems; runtests("test/propagator_test.jl")
```

Expected: `MethodError: no method matching scalelocal(::LocalOp{2}, ::Float64)`.

- [ ] **Step 3: Implement**

Add to the top of `src/trotterdecomposition.jl`, and call it on `term` inside the loop of `trotterise` immediately after the `layer isa Pair` destructuring:

```julia
"""
    $(TYPEDSIGNATURES)

A term as the sum type a Trotter layer holds. A bare single term becomes a `Hamiltonian`, or a
`Liouvillian` when it is dissipative; `Hamiltonian`, `Liouvillian` and `Ising` pass through.
"""
layerterm(t::Union{Hamiltonian, Liouvillian, Ising}) = t
layerterm(t::Dissipator) = convert(Liouvillian, t)
layerterm(t::AbstractTerm) = convert(Hamiltonian, t)
```

In `trotterise`, the destructuring line becomes:

```julia
        raw, axis = layer isa Pair ? (first(layer), last(layer)) : (layer, 0)
        term = layerterm(raw)
```

- [ ] **Step 4: Add the error for terms with no propagator route**

Define this as a **named helper**, not inline — Task 7 rewrites `trotterise`'s loop and must be able to
call it rather than duplicate it. `isaxial` is true for long-range axial terms too, so test for the
concrete single-term types that `generator` can vectorise:

```julia
"""
    $(TYPEDSIGNATURES)

Throw unless every term of `t` can be exponentiated into a two-site propagator.
"""
function _checkroute(t::AbstractTerm)
    unroutable = filter(x -> !(x isa LocalOp), collect(terms(t)))
    isempty(unroutable) || throw(
        ArgumentError(
            "no propagator route for $(typeof(first(unroutable))): only nearest-neighbour " *
                "terms can be exponentiated into a two-site propagator"
        ),
    )
    return nothing
end

# An Ising coupling is nearest-neighbour by construction, and `terms` is not defined for it.
_checkroute(::Ising) = nothing
```

Call `_checkroute(term)` in `trotterise`'s loop before building the propagator.

- [ ] **Step 5: Run the test**

```julia
using ReTestItems; runtests("test/propagator_test.jl")
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/trotterdecomposition.jl test/propagator_test.jl
git commit -m "fix: promote bare terms to Trotter layers

trotterise([LocalOp(X,X), LocalOp(Y,Y)], ...) threw on scalelocal, so the
multi-layer form only accepted terms already wrapped in a Hamiltonian. Each
layer is now promoted, dissipative terms to a Liouvillian. A layer holding a
long-range term now names the type it cannot exponentiate."
```

---

### Task 4: Disambiguate `productpepo`

`productpepo(rho::AbstractMatrix, bond)` is taken as a *matrix of local states*, so a single density matrix broadcasts over its scalar entries and fails with `no method matching _localdensitymatrix(::ComplexF64)`.

**Files:**
- Modify: `src/state.jl:41-58`
- Test: `test/state_test.jl`

**Interfaces:**
- Produces: `productpepo(rho::AbstractMatrix{<:Number}, bond)` treats `rho` as one local density matrix; a unit cell is any `AbstractMatrix` whose elements are not numbers.

- [ ] **Step 1: Write the failing test**

Append to `test/state_test.jl` inside the existing `@testitem`:

```julia
@testset "a bare density matrix is one site, not a unit cell" begin
    rho = bloch((0.0, 0.0, -1.0))
    @test size(productpepo(rho, 4)) == (1, 1, 1)
    @test convert(Array, productpepo(rho, 4)[1, 1])[:, :, 1, 1, 1, 1] ≈ rho

    # A unit cell of density matrices still works.
    cell = [bloch((0, 0, 1)) bloch((0, 0, -1)); bloch((0, 0, -1)) bloch((0, 0, 1))]
    @test size(productpepo(cell, 2)) == (2, 2, 1)
end
```

- [ ] **Step 2: Run it and confirm it fails**

```julia
using ReTestItems; runtests("test/state_test.jl")
```

Expected: `MethodError: no method matching _localdensitymatrix(::ComplexF64)`.

- [ ] **Step 3: Implement**

In `src/state.jl`, the generic single-state method is `productpepo(init, bond) = productpepo(fill(init, 1, 1), bond)`, which a numeric matrix never reaches because `productpepo(inits::AbstractMatrix, bond)` is more specific. Restrict the unit-cell method and add a numeric-matrix method above it:

```julia
productpepo(rho::AbstractMatrix{<:Number}, bond) = productpepo(fill(rho, 1, 1), bond)
```

Leave `productpepo(inits::AbstractMatrix, bond)` as it is — a `Matrix{<:Matrix}` or `Matrix{<:Tuple}` still selects it, and `AbstractMatrix{<:Number}` is strictly more specific for the bare case.

Extend the docstring's first paragraph to say that a matrix of numbers is one local density matrix, while a matrix of matrices, tuples or `ThermalState`s is a unit cell.

- [ ] **Step 4: Run the test**

```julia
using ReTestItems; runtests("test/state_test.jl")
```

Expected: PASS (32 existing plus the new ones).

- [ ] **Step 5: Commit**

```bash
git add src/state.jl test/state_test.jl
git commit -m "fix: read a bare density matrix as one site in productpepo

productpepo(rho::AbstractMatrix, bond) selected the unit-cell method, so a single
density matrix was broadcast over its scalar entries. A matrix of numbers is now
one local state; a matrix of matrices, tuples or ThermalStates is a unit cell."
```

---

### Task 5: `TrotterLayer`

**Files:**
- Rewrite in place: `src/model/trotterlayer.jl` — its current contents are the superseded TRGroups-based version, which uses the removed `broadcastmodel` and does not load. Replace them; **do not `git mv` the file anywhere.** Git history is the record of the old version.
- Modify: `src/TimeEvolutionPEPO.jl` (add `include("model/trotterlayer.jl")` after `include("model/ising.jl")`; export `TrotterLayer`; remove its commented-out include)
- Test: `test/model_test.jl` (new file)

**Interfaces:**
- Consumes: `layerterm` from Task 3.
- Produces:
  - `TrotterLayer(data::Matrix, axis::Int = 0)` and `TrotterLayer(term, axis::Int = 0)` (wraps into a `1×1`).
  - Fields `data::Matrix{T}`, `axis::Int`.
  - `Base.size(l::TrotterLayer)`, `Base.getindex(l::TrotterLayer, r, c)` with periodic wrapping via `mod1`.
  - `scalelayer(l::TrotterLayer, a) -> TrotterLayer` scaling only the one-local part of every entry.

- [ ] **Step 1: Write the failing test**

Create `test/model_test.jl`:

```julia
@testsetup module ModelSetup
using Reexport
@reexport using TimeEvolutionPEPO
@reexport using TensorKit
@reexport using LinearAlgebra
@reexport using Logging

export quiet
quiet(f) = with_logger(f, ConsoleLogger(stderr, Logging.Error))
end # ModelSetup

@testitem "Trotter layers" setup = [ModelSetup] begin
    X, Y, Z = PAULI

    @testset "a bare term is a 1x1 layer" begin
        layer = TrotterLayer(LocalOp(Z, Z))
        @test size(layer) == (1, 1)
        @test layer.axis == 0
        @test layer[1, 1] isa Hamiltonian
        # A 1x1 layer wraps periodically, so it serves any unit cell.
        @test layer[3, 7] === layer[1, 1]
    end

    @testset "a matrix of terms keeps its shape" begin
        data = [LocalOp(Z, Z) LocalOp(X, X); LocalOp(X, X) LocalOp(Z, Z)]
        layer = TrotterLayer(data, 1)
        @test size(layer) == (2, 2)
        @test layer.axis == 1
        @test layer[3, 3] === layer[1, 1]
    end

    @testset "a non-axial term cannot be pinned to an axis" begin
        @test_throws ArgumentError TrotterLayer(Dissipator(Z), 1)
    end
end
```

- [ ] **Step 2: Run it and confirm it fails**

```julia
using ReTestItems; runtests("test/model_test.jl")
```

Expected: `UndefVarError: TrotterLayer not defined`.

- [ ] **Step 3: Implement**

Replace the entire contents of `src/model/trotterlayer.jl` with the following. Do not move the file and
do not keep the old definitions — they reference `TRGroups` and `broadcastmodel`, neither of which
exists in the live tree.

```julia
# One layer of a Suzuki-Trotter decomposition: a term per unit-cell site, and the axis it acts on.

"""
$(TYPEDEF)

One layer of a Suzuki-Trotter decomposition.

`data[r, c]` is anchored at site `(r, c)`: its two-local part is the coupling between `(r, c)` and its
neighbour in the `+axis` direction — east for axis 1, south for axis 2 — and its one-local part
belongs to `(r, c)` alone. Indexing wraps periodically, so a `1×1` layer is uniform and serves any
unit cell.

`axis` is 1 for the column direction, 2 for the row direction, or 0 to apply the layer on both.

# Constructors

    TrotterLayer(data::AbstractMatrix, axis::Int = 0)
    TrotterLayer(term, axis::Int = 0)
"""
struct TrotterLayer{T <: AbstractTerm}
    data::Matrix{T}
    axis::Int
    function TrotterLayer(data::AbstractMatrix, axis::Int = 0)
        axis in (0, 1, 2) || throw(ArgumentError("axis must be 0, 1 or 2, got $axis"))
        promoted = map(layerterm, data)
        if axis != 0 && any(!isaxial, promoted)
            throw(ArgumentError("a non-axial term cannot be pinned to axis $axis"))
        end
        T = mapreduce(typeof, promote_type, promoted)
        return new{T}(convert(Matrix{T}, promoted), axis)
    end
end

TrotterLayer(term, axis::Int = 0) = TrotterLayer(fill(term, 1, 1), axis)
TrotterLayer(layer::TrotterLayer) = layer

Base.size(l::TrotterLayer) = size(l.data)
Base.getindex(l::TrotterLayer, r::Int, c::Int) = l.data[mod1(r, size(l, 1)), mod1(c, size(l, 2))]
Base.size(l::TrotterLayer, d::Int) = size(l.data, d)

"""
    $(TYPEDSIGNATURES)

Scale the one-local part of every entry of a layer, leaving the two-local couplings alone.
"""
scalelayer(l::TrotterLayer, a) = TrotterLayer(map(t -> scalelocal(t, a), l.data), l.axis)
```

Note `promote_type` needs a rule so a `Hamiltonian` and a `Liouvillian` in one layer promote to
`Liouvillian`. Add to `src/model/liouvillian.jl`:

```julia
Base.promote_rule(::Type{<:Hamiltonian}, ::Type{<:Liouvillian}) = Liouvillian
```

- [ ] **Step 5: Wire the include and export**

In `src/TimeEvolutionPEPO.jl`, add `include("model/trotterlayer.jl")` after `include("model/ising.jl")` and before `include("propagator.jl")`, add `TrotterLayer` to the model export line, and delete the commented `# include("model/trotterlayer.jl")` from the block of unloaded includes.

- [ ] **Step 6: Run the test**

```julia
using ReTestItems; runtests("test/model_test.jl")
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/model/trotterlayer.jl src/model/liouvillian.jl src/TimeEvolutionPEPO.jl test/model_test.jl
git commit -m "feat: TrotterLayer carries a term per unit-cell site

data[r, c] is anchored at site (r, c): its two-local part is the coupling to the
neighbour in the +axis direction, its one-local part belongs to that site alone.
Indexing wraps periodically so a 1x1 layer is uniform. Replaces the superseded
TRGroups-based version in place; git history holds the old one."
```

---

### Task 6: `Model`

**Files:**
- Rewrite in place: `src/model/model.jl` — its current contents are the superseded version, which does not load. Replace them; **do not `git mv` the file anywhere.**
- Modify: `src/TimeEvolutionPEPO.jl` (include after `trotterlayer.jl`; export `Model`, `distribute!`)
- Test: `test/model_test.jl`

**Interfaces:**
- Consumes: `TrotterLayer`, `scalelayer` from Task 5.
- Produces:
  - `Model{T}` with field `layers::Vector{TrotterLayer{T}}`; constructors `Model(terms...)`, `Model(terms::Vector)`, `Model{T}()`.
  - `Base.push!(model, layer)`, `Base.length`, `Base.iterate`, `Base.show`.
  - `distribute!(model::Model, term) -> Model`.

- [ ] **Step 1: Write the failing test**

Append to `test/model_test.jl`:

```julia
@testitem "Model container" setup = [ModelSetup] begin
    X, Y, Z = PAULI

    @testset "layers are appended, never distributed" begin
        model = Model(LocalOp(X, X), LocalOp(Y, Y))
        @test length(model) == 2
        push!(model, LocalOp(Z, Z))
        @test length(model) == 3
        # push! appends whether or not the term is one-local.
        push!(model, LocalOp(X))
        @test length(model) == 4
    end

    @testset "distribute! spreads over existing layers" begin
        model = Model(LocalOp(X, X), LocalOp(Y, Y))
        distribute!(model, LocalOp(Z))
        # The layer count is unchanged; the numerical share is checked in the next testitem, once
        # trotterise accepts a Model.
        @test length(model) == 2
        @test length(oneterms(model[1][1, 1])) == 1
    end

    @testset "distributing over no layers is an error" begin
        @test_throws ArgumentError distribute!(Model{Hamiltonian}(), LocalOp(Z))
    end

    @testset "a Hamiltonian model rejects dissipation" begin
        model = Model{Hamiltonian}()
        push!(model, LocalOp(X, X))
        @test_throws ArgumentError push!(model, Dissipator(Z))
    end
end
```

Every testset here must pass at the end of this task. The numerical check on `distribute!`'s `1/n`
share needs `trotterise(::Model, …)` and so lives in Task 7.

- [ ] **Step 2: Run it and confirm it fails**

```julia
using ReTestItems; runtests("test/model_test.jl")
```

Expected: `UndefVarError: Model not defined`.

- [ ] **Step 3: Implement**

Replace the entire contents of `src/model/model.jl` with the following. Do not move the file.

```julia
# A model is an ordered list of Suzuki-Trotter layers, swept in order once per time step.

"""
$(TYPEDEF)

An ordered list of [`TrotterLayer`](@ref)s, exponentiated and swept in order once per time step.
Terms in the same layer are exponentiated together rather than Trotterised apart.

The type parameter is `Hamiltonian`, `Liouvillian` or `Ising`; a `Model{Hamiltonian}` rejects
dissipative terms.

# Constructors

    Model(layers...)
    Model(layers::AbstractVector)
    Model{T}()

See also: [`distribute!`](@ref), [`trotterise`](@ref)
"""
struct Model{T <: AbstractTerm}
    layers::Vector{TrotterLayer{T}}
end

function Model(layers::AbstractVector)
    wrapped = map(TrotterLayer, layers)
    T = mapreduce(l -> eltype(l.data), promote_type, wrapped)
    return Model{T}([convert(TrotterLayer{T}, l) for l in wrapped])
end

Model(layers...) = Model(collect(layers))
Model{T}() where {T} = Model{T}(TrotterLayer{T}[])

function Base.convert(::Type{TrotterLayer{T}}, l::TrotterLayer) where {T}
    return TrotterLayer(map(t -> convert(T, t), l.data), l.axis)
end

Base.length(model::Model) = length(model.layers)
Base.iterate(model::Model, state = 1) = iterate(model.layers, state)
Base.eltype(::Model{T}) where {T} = T
Base.getindex(model::Model, i) = model.layers[i]

function Base.show(io::IO, ::MIME"text/plain", model::Model{T}) where {T}
    n = length(model)
    return print(io, "Model{$T} with $n Trotter layer", n == 1 ? "" : "s")
end

function Base.push!(model::Model{T}, layer) where {T}
    l = TrotterLayer(layer)
    eltype(l.data) <: T || throw(
        ArgumentError(
            "cannot add a term of type $(eltype(l.data)) to a $(typeof(model)); " *
                "construct a Model{$(promote_type(T, eltype(l.data)))} instead"
        ),
    )
    push!(model.layers, convert(TrotterLayer{T}, l))
    return model
end

"""
    $(TYPEDSIGNATURES)

Add `term` to every existing layer of `model`, scaled by `1 / length(model)`, so that a full time
step applies it once in total. Use [`push!`](@ref) instead to give the term a layer of its own.
"""
function distribute!(model::Model, term)
    n = length(model)
    n > 0 || throw(ArgumentError("cannot distribute a term over a model with no layers"))
    share = (1 / n) * term
    for (i, layer) in enumerate(model.layers)
        model.layers[i] = TrotterLayer(map(t -> t + share, layer.data), layer.axis)
    end
    return model
end
```

- [ ] **Step 5: Wire the include and export**

In `src/TimeEvolutionPEPO.jl`, add `include("model/model.jl")` after `include("model/trotterlayer.jl")`, add `Model` and `distribute!` to the model export line, and delete the commented `# include("model/model.jl")`.

- [ ] **Step 6: Run the test**

```julia
using ReTestItems; runtests("test/model_test.jl")
```

Expected: PASS, every testset.

- [ ] **Step 7: Commit**

```bash
git add src/model/model.jl src/TimeEvolutionPEPO.jl test/model_test.jl
git commit -m "feat: Model holds an ordered list of Trotter layers

push! always appends a layer; distribute! spreads a term over the existing
layers with a 1/nlayers prefactor and throws on an empty model. The superseded
version triggered distribution by argument type, which made the behaviour
depend on whether the term was wrapped. Replaces the superseded version in place."
```

---

### Task 7: `trotterise` over a `Model`

**Files:**
- Modify: `src/trotterdecomposition.jl`
- Modify: `src/propagator.jl` (add one `bondpropagator` method beside the existing one — move nothing out)
- Test: `test/model_test.jl`

**Interfaces:**
- Consumes: `Model`, `TrotterLayer`, `layerterm`, `_checkroute`.
- Produces: `trotterise(model, P, dt) -> Vector{Tuple{Matrix{<:AbstractTensorMap}, Int}}`. Every element's first entry is a `Matrix` of propagators indexed as the layer's `data` is; a uniform layer yields a `1×1` matrix.

- [ ] **Step 1: Write the failing test**

Append to `test/model_test.jl`:

```julia
@testitem "Trotter decomposition of a model" setup = [ModelSetup] begin
    X, Y, Z = PAULI
    P = ComplexSpace(2)

    @testset "a uniform layer yields a 1x1 propagator matrix" begin
        decomposition = trotterise(LocalOp(Z, Z), P, 0.1)
        @test length(decomposition) == 2                   # one layer over both axes
        propagators, axis = first(decomposition)
        @test size(propagators) == (1, 1)
        @test axis == 1
    end

    @testset "a site-dependent layer yields one propagator per site" begin
        data = [LocalOp(Z, Z) 0.5 * LocalOp(Z, Z); 0.5 * LocalOp(Z, Z) LocalOp(Z, Z)]
        decomposition = trotterise(Model([TrotterLayer(data)]), P, 0.1)
        propagators, = first(decomposition)
        @test size(propagators) == (2, 2)
        @test !(propagators[1, 1] ≈ propagators[1, 2])
    end

    @testset "distribute! gives each layer a 1/n share" begin
        # Moved here from Task 6: the check needs trotterise to accept a Model.
        distributed = Model(LocalOp(X, X), LocalOp(Y, Y))
        distribute!(distributed, LocalOp(Z))
        byhand = Model(LocalOp(X, X) + 0.5 * LocalOp(Z), LocalOp(Y, Y) + 0.5 * LocalOp(Z))
        got = trotterise(distributed, P, 0.1)
        want = trotterise(byhand, P, 0.1)
        @test length(got) == length(want)
        @test all(first(g)[1, 1] ≈ first(w)[1, 1] for (g, w) in zip(got, want))
    end

    @testset "the one-local part is halved across the two axes" begin
        # A layer applied on both axes must halve its one-local part; pinned to one axis it must not.
        both = first(first(trotterise(zero(Hamiltonian) + LocalOp(X), P, 0.1)))[1, 1]
        pinned = first(first(trotterise([(zero(Hamiltonian) + LocalOp(X)) => 1], P, 0.1)))[1, 1]
        @test !(both ≈ pinned)
    end
end
```

- [ ] **Step 2: Run it and confirm it fails**

```julia
using ReTestItems; runtests("test/model_test.jl")
```

Expected: failure on `size(propagators)` — the current `trotterise` returns a bare tensor, not a matrix.

- [ ] **Step 3: Implement**

Rewrite `src/trotterdecomposition.jl`'s `trotterise` to build a propagator matrix per layer. A propagator for the bond anchored at `(r, c)` takes its two-local coupling and slot-1 one-local part from `data[r, c]`, and its slot-2 one-local part from the neighbour in the `+axis` direction.

```julia
function trotterise(model::Model, P::IndexSpace, dt::Number)
    decomposition = Tuple{Matrix{AbstractTensorMap}, Int}[]

    for layer in model
        if layer.axis == 0
            halved = scalelayer(layer, 0.5)
            push!(decomposition, (_layerpropagators(halved, P, dt, 1), 1))
            push!(decomposition, (_layerpropagators(halved, P, dt, 2), 2))
        else
            push!(decomposition, (_layerpropagators(layer, P, dt, layer.axis), layer.axis))
        end
    end

    return decomposition
end

function _layerpropagators(layer::TrotterLayer, P::IndexSpace, dt::Number, axis::Int)
    Nr, Nc = size(layer)
    return AbstractTensorMap[
        begin
            anchor = layer[r, c]
            neighbour = axis == 1 ? layer[r, c + 1] : layer[r + 1, c]
            _checkroute(anchor)
            bondpropagator(anchor, neighbour, P, dt)
        end for r in 1:Nr, c in 1:Nc
    ]
end
```

`bondpropagator(anchor, neighbour, P, dt)` is a **new method added beside the existing one in
`src/propagator.jl`** — do not move any code out of that file. It is the existing body with
`generator(model, P)` replaced by `generator(anchor, neighbour, P)`:

```julia
function bondpropagator(
        anchor::AbstractTerm, neighbour::AbstractTerm, P::IndexSpace, dt::Number
    )
    matrix = exp(dt * generator(anchor, neighbour, P))
    spaces = P ⊗ P ⊗ P' ⊗ P'
    t = vectorised_tensormap(matrix, spaces, spaces)
    return permute(t, ((1, 3, 2, 4), (5, 7, 6, 8)))
end
```

`_checkroute` already exists as a named helper from Task 3 — call it, do not redefine it.

Keep the vector and single-term methods as thin wrappers:

```julia
trotterise(layers::AbstractVector, P::IndexSpace, dt::Number) = trotterise(Model(layers), P, dt)
trotterise(layer, P::IndexSpace, dt::Number) = trotterise(Model([layer]), P, dt)
```

- [ ] **Step 4: Run the model and propagator tests**

```julia
using ReTestItems; runtests("test/model_test.jl")
using ReTestItems; runtests("test/propagator_test.jl")
```

Expected: `model_test.jl` fully PASS, including the `distribute!` testset left red in Task 6.

- [ ] **Step 5: Commit**

```bash
git add src/trotterdecomposition.jl test/model_test.jl
git commit -m "feat: trotterise builds one propagator per unit-cell site

Each layer becomes a Matrix of propagators indexed as its term matrix is; the
bond anchored at (r, c) takes its coupling and first-slot one-local part from
data[r, c] and its second-slot part from the neighbour in the +axis direction.
A uniform layer yields a 1x1 matrix."
```

---

### Task 8: Sweep over a propagator matrix

**Files:**
- Modify: `src/simpleupdate.jl:109-141` (`axissweep!`) and `:153-170` (`timestep!`)
- Modify: `src/simulation.jl:27-33` (`Simulation` constructor)
- Test: `test/simpleupdate_test.jl`

**Interfaces:**
- Consumes: the decomposition from Task 7.
- Produces: `axissweep!(state, weights, propagators::AbstractMatrix, axis; …)`; the existing single-tensor method stays as `axissweep!(…, propagator::AbstractTensorMap, …)` wrapping in a `1×1`.

- [ ] **Step 1: Write the failing test**

Append to `test/simpleupdate_test.jl` inside the existing `@testitem`:

```julia
@testset "a propagator matrix is indexed per bond" begin
    X, Y, Z = PAULI
    P = ComplexSpace(2)
    state = productpepo(fill((0, 0, -1), 2, 2), 2)
    weights = SUWeight(state)

    # A 1x1 matrix must behave exactly as the bare tensor did.
    prop = first(first(trotterise(LocalOp(Z, Z), P, 0.05)))
    @test size(prop) == (1, 1)
    ϵ = axissweep!(state, weights, prop, 1)
    @test isfinite(ϵ)
end
```

- [ ] **Step 2: Run it and confirm it fails**

```julia
using ReTestItems; runtests("test/simpleupdate_test.jl")
```

Expected: `MethodError` on `axissweep!` with a `Matrix` third argument.

- [ ] **Step 3: Implement**

In `src/simpleupdate.jl`, change `axissweep!`'s third parameter to `propagators::AbstractMatrix` and index it inside the loop. The `_swapsites` call stays exactly where it is — it is applied per bond now:

```julia
    for r in 1:Nr, c in 1:Nc
        site1, site2 = if axis == 1
            CartesianIndex(r, c), CartesianIndex(r, c + 1)
        else
            CartesianIndex(r + 1, c), CartesianIndex(r, c)
        end
        p = propagators[mod1(r, size(propagators, 1)), mod1(c, size(propagators, 2))]
        prop = axis == 1 ? p : _swapsites(p)
        ϵ = max(ϵ, bondstep!(state, weights, prop, [site1, site2]; trunc, truncalg, env))
    end
```

Delete the `prop = axis == 1 ? propagator : _swapsites(propagator)` line above the loop. Add the compatibility method below `axissweep!`:

```julia
function axissweep!(
        state::InfinitePEPO, weights::SUWeight, propagator::AbstractTensorMap, axis::Int; kwargs...
    )
    return axissweep!(state, weights, fill(propagator, 1, 1), axis; kwargs...)
end
```

- [ ] **Step 4: Validate unit-cell sizes in `Simulation`**

In `src/simulation.jl`, after `trotterise` is called in the constructor, check every layer's propagator matrix against the state:

```julia
    Nr, Nc, = size(state)
    decomposition = trotterise(model, P, timestep)
    for (propagators, _) in decomposition
        size(propagators) == (1, 1) || size(propagators) == (Nr, Nc) || throw(
            DimensionMismatch(
                "a model layer of size $(size(propagators)) cannot be applied to a " *
                    "$Nr×$Nc unit cell: give a 1×1 layer or one matching the state"
            ),
        )
    end
```

- [ ] **Step 5: Run the sweep and simulation tests**

```julia
using ReTestItems; runtests("test/simpleupdate_test.jl")
using ReTestItems; runtests("test/simulation_test.jl")
using ReTestItems; runtests("test/envtruncation_test.jl")
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/simpleupdate.jl src/simulation.jl test/simpleupdate_test.jl
git commit -m "feat: sweep a propagator per bond

axissweep! takes a Matrix of propagators and indexes it with mod1, so a 1x1
matrix serves any unit cell and a full-size one varies per bond. Simulation
rejects a layer whose size is neither 1x1 nor the state's unit cell rather
than tiling it silently."
```

---

### Task 9: Staggered-field acceptance test

The load-bearing verification: it pins the one-local prefactor and which `kron` factor is the first site, neither of which a uniform model can see.

**Files:**
- Test: `test/model_test.jl`

- [ ] **Step 1: Write the test**

Append to `test/model_test.jl`:

```julia
@testitem "staggered field against exp(tL)" setup = [ModelSetup] begin
    X, Y, Z = PAULI
    P = ComplexSpace(2)

    # A one-local Lindbladian on one sublattice only. At D = 1 with no coupling every site evolves
    # independently, so both sublattices are checked against dense linear algebra.
    jump = 0.5 * (Y - im * Z)
    v0 = (0.0, 0.0, -1.0)
    dt, nsteps = 0.02, 5

    active = zero(Hamiltonian) + LocalOp(X) + Dissipator(jump)
    idle = zero(Hamiltonian) + LocalOp(X)   # hamiltonian-only sublattice, no dissipation

    layer = TrotterLayer([active idle; idle active])
    sim = quiet(() -> Simulation(productpepo(fill(v0, 2, 2), 1), Model([layer]); timestep = dt))
    quiet(() -> simulate!(sim; numsteps = nsteps, verbose = false))

    dense(H, jumps) = begin
        d = 2
        Id = Matrix{ComplexF64}(I, d, d)
        G = -im * (kron(Id, H) - kron(transpose(H), Id))
        for L in jumps
            LdL = L' * L
            G += kron(conj(L), L) - (kron(Id, LdL) + kron(transpose(LdL), Id)) / 2
        end
        G
    end

    evolve(G) = reshape(exp(nsteps * dt * G) * vec(bloch(v0)), 2, 2)
    want_active = evolve(dense(Matrix(X), [jump]))
    want_idle = evolve(dense(Matrix(X), Matrix{ComplexF64}[]))

    dm = quiet(() -> DensityMatrix(quantumstate(sim), 4))
    got_active = reshape(convert(Array, partialtrace(dm, (1, 1))), 2, 2)
    got_idle = reshape(convert(Array, partialtrace(dm, (1, 2))), 2, 2)

    @test got_active ≈ want_active atol = 1e-8
    @test got_idle ≈ want_idle atol = 1e-8
    # The two sublattices must actually differ, or the test proves nothing.
    @test norm(want_active - want_idle) > 0.1
end
```

- [ ] **Step 2: Run it**

```julia
using ReTestItems; runtests("test/model_test.jl")
```

Expected: PASS. If `got_active` and `got_idle` are swapped, the slot assignment in `onsite` (Task 1, Step 3) or the anchor/neighbour choice in `_layerpropagators` (Task 7, Step 3) is reversed — fix there, not in the test.

- [ ] **Step 3: Run every test file**

```julia
using ReTestItems; runtests("..")
```

Expected: all items PASS.

- [ ] **Step 4: Commit**

```bash
git add test/model_test.jl
git commit -m "test: staggered field on one sublattice against exp(tL)

At D = 1 with no coupling each site evolves independently, so both sublattices
are checked against dense linear algebra. This pins the one-local prefactor and
which kron factor of the propagator is the first site, neither of which a
uniform model can distinguish."
```

---

### Task 10: Documentation

**Files:**
- Modify: `docs/src/man/models.md`
- Modify: `docs/src/man/states.md` (the `productpepo(rho, bonddim)` example fixed by Task 4)
- Modify: `docs/src/lib/models.md` (add `Model`, `TrotterLayer`, `distribute!`)
- Modify: `CLAUDE.md` (in the main checkout, `/Users/jdunham/.julia/dev/TimeEvolutionPEPO/CLAUDE.md` — it is gitignored and absent from the worktree)

- [ ] **Step 1: Rewrite the layer section of `docs/src/man/models.md`**

Replace the "Trotter layers" section with one that uses `Model`, `push!` and `distribute!`, and add a short section on site-dependent terms showing a `TrotterLayer` built from a matrix. Every `@repl` block must be one that actually runs.

- [ ] **Step 2: Add the new bindings to `docs/src/lib/models.md`**

```markdown
## Models and layers

```@docs
Model
TrotterLayer
distribute!
trotterise
```
```

- [ ] **Step 3: Build the docs**

```bash
julia --project=docs docs/make.jl
```

Expected: no errors.

- [ ] **Step 4: Check the rendered examples for silent failures**

A `@repl` block that throws renders its error rather than failing the build, so grep the rendered HTML:

```bash
python3 - <<'PY'
import re, html, glob
for f in sorted(glob.glob('docs/build/**/index.html', recursive=True)):
    t = html.unescape(re.sub(r'<[^>]+>', '', open(f).read()))
    for m in re.finditer(r'(ERROR:[^\n]{0,120})', t):
        print(f, '|', m.group(1))
PY
```

Expected: no output.

- [ ] **Step 5: Update `CLAUDE.md`**

In the "State of the refactor" section, remove the `Model`/`TrotterLayer` bullet from the not-done list. In "The time-step pipeline", record that a decomposition element is a `(Matrix{propagator}, axis)` pair and that `data[r, c]` is site-anchored. Keep the long-range and second-order-Trotter gaps.

- [ ] **Step 6: Commit**

```bash
git add docs CLAUDE.md 2>/dev/null; git add docs
git commit -m "docs: Model, TrotterLayer and site-dependent terms"
```

---

## Self-review

**Spec coverage.** Site-anchored terms → Tasks 5, 7. Per-endpoint one-local placement → Task 1. `push!`/`distribute!` split → Task 6. `Model` type parameter → Task 6. Propagator matrices through the sweep → Tasks 7, 8. Unit-cell validation → Task 8, Step 4. Deferred expansion, with an error naming the offending type → Task 3, Step 4. Verification → Task 9 plus the regression runs in Tasks 1, 8. Documentation → Task 10.

**Beyond the spec**, added after probing found them broken: `Ising` had no `generator` (Task 2), `trotterise` rejected bare terms (Task 3), `productpepo` misread a bare density matrix (Task 4). All three were pre-existing and confirmed by direct probe.

**Type consistency.** `layerterm` (Task 3) is used by `TrotterLayer` (Task 5). `scalelayer` (Task 5) is used by `trotterise` (Task 7). `_layerpropagators` (Task 7) is consumed by nothing else. `_swapsites` already exists in `src/simpleupdate.jl`. `_checkroute` is a named helper from Task 3 that Task 7 calls, never redefines.

**No code moves between files.** Tasks 5 and 6 rewrite `src/model/{trotterlayer,model}.jl` in place rather than relocating the superseded versions to `src/dev/`; Task 7 adds a `bondpropagator` method beside its sibling in `src/propagator.jl` rather than extracting a helper elsewhere. A relocation makes the diff unreadable, and rearranging files is a separate change.

**Task 5's `promote_type` behaviour, measured** (so the implementer does not have to guess):

```
promote_type(Hamiltonian{Tuple{LocalOp{2}}}, Hamiltonian{Tuple{}})  ==  Hamiltonian
promote_type(Hamiltonian{Tuple{LocalOp{2}}}, Liouvillian{Tuple{LocalOp{2}}})  ==  AbstractSumOfTerms
mapreduce(typeof, promote_type, [a b; b a])  ==  Hamiltonian        # a, b both Hamiltonian
```

So no extra rule is needed for two `Hamiltonian`s — that already yields `Hamiltonian`. The
`Base.promote_rule(::Type{<:Hamiltonian}, ::Type{<:Liouvillian}) = Liouvillian` line in Step 4 is
**required**, not optional: without it a mixed layer becomes `TrotterLayer{AbstractSumOfTerms}`, which
defeats the type parameter `Model{T}` relies on to reject dissipation.
