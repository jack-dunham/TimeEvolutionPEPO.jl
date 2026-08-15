```@meta
CurrentModule = TimeEvolutionPEPO
ShareDefaultModule = true
```
```@setup
using TimeEvolutionPEPO, TensorKit
```
# State representation

The state is a PEPSKit `InfinitePEPO` whose tensors are local density matrices, paired with an
`SUWeight` holding one weight per bond of the unit cell. Both types come from PEPSKit and are
re-exported here; see the [PEPSKit documentation](https://quantumkithub.github.io/PEPSKit.jl/stable/)
for their full interface.

## Construction

`productpepo` builds the PEPO of a product state from a local density matrix and a bond dimension.
The density matrix can be given directly, or as a spin-1/2 Bloch vector.

```@repl
bonddim = 4
state = productpepo((0, 0, -1), bonddim);       # all spins pointing down
rho = bloch((0, 0, -1))                          # the same state, written out
state[1, 1] ≈ productpepo(rho, bonddim)[1, 1]
```

A non-trivial unit cell comes from passing a matrix of local states, one entry per site.

```@repl
up, down = (0, 0, 1), (0, 0, -1)
size(productpepo([down up; up down], bonddim))
```

`ThermalState()` selects the infinite-temperature density matrix, the usual starting point for
annealing.

```@repl
convert(Array, productpepo(ThermalState(), 1)[1, 1])[:, :, 1, 1, 1, 1]
```

The bond argument is a dimension or a `TensorKit` space; an integer is shorthand for a trivial
`ComplexSpace`.

```@repl
productpepo(rho, ComplexSpace(4))[1, 1] ≈ productpepo(rho, 4)[1, 1]
```

Only the first basis vector of each virtual space carries the product state; the rest is padded with
zeros, so the bond dimension is available for the evolution to fill.

!!! note
    `productpepo` builds trivial (non-symmetric) physical and virtual spaces. Constructing a state on
    a symmetric space, for instance to preserve total spin or particle number, is not supported by
    this constructor.

## Index conventions

Tensors follow PEPSKit's `PEPOTensor` convention: physical indices in the codomain, virtual indices in
the domain, ordered north, east, south, west.

```@repl
space(state[1, 1])
```

Index 1 is the ket and index 2 the bra, so that an operator on index 1 left-multiplies the density
matrix and one on index 2 right-multiplies it. The vectorisation that follows from this ordering is
described in [Conventions and the `TensorMap` type](@ref).

## Bond weights

The bond weights are held outside the tensors, as PEPSKit's `SUWeight`, rather than absorbed into
them. `SUWeight(state)` gives the trivial weights.

```@repl
weights = SUWeight(state);
size(weights)
```

For a state under evolution the weights are the Schmidt spectra of the bonds, so their entropy
measures how much entanglement each bond carries.

```@repl
bondentropy(weights)
bondentropy(weights; normalized = true)
```

A product state has Schmidt spectrum `(1, 0, …)` on every bond. The trivial weights are not that
spectrum — they are invertible, which the spectrum is not — so they act as a gauge choice that the
first bond update replaces.

A `Simulation` carries both, reachable with `quantumstate` and `bondweights`.
