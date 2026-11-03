# Quick start

```@meta
CurrentModule = TimeEvolutionPEPO
ShareDefaultModule = true
```

To get started, load TimeEvolutionPEPO.jl and TensorKit.jl.

```@repl
using TimeEvolutionPEPO
using TensorKit
```

## Setting up a simulation

A simulation needs an initial state, a model, and a time step.

The state is an `InfinitePEPO` whose tensors are density matrices. `productpepo` builds one from a
local density matrix per unit-cell site, given here as spin-1/2 Bloch vectors, together with the
virtual bond dimension.

```@repl
state = productpepo(fill((0, 0, -1), 2, 2), 4);
size(state)
```

The model is a term, or a vector of terms giving the Suzuki-Trotter layers. As an example, take the
nearest-neighbour Ising model

```math
H = \frac{J}{4} \sum_{\langle i,j \rangle} \sigma^{z}_i \sigma^{z}_j
```

with transverse dissipation ``L_j = \frac{\sqrt{\gamma}}{2}(\sigma^y_j - i\sigma^z_j)``, at
``\gamma = 1``.

```@repl
X, Y, Z = PAULI
model = J -> J / 4 * LocalOp(Z, Z) + Dissipator(0.5 * (Y - im * Z))
```

`Simulation` holds the state, its bond weights and the Trotter decomposition of the propagator.

```@repl
sim = Simulation(state, model(3.0); timestep = 0.05);
```

## Running it

`simulate!` advances the simulation in place and optionally calls a callback on a schedule. The
callback is passed the `Simulation` itself.

```@repl
magnetisation(sim; bonddim = 8) = real(expval(partialtrace(DensityMatrix(quantumstate(sim), bonddim), (1, 1)), PAULI[1]))
magn, times = Float64[], Float64[];
callback = sim -> begin
    push!(magn, magnetisation(sim))
    push!(times, simulationinfo(sim).simtime)
end;
simulate!(callback, sim; numsteps = 20, maxshots = 10, verbose = false);
magn[end]
```

`simulate!` can be called again on the same object to continue from where it stopped, for instance to
sample less often once the fast transient is over.

```@repl
simulate!(callback, sim; numsteps = 40, maxshots = 10, verbose = false);
simulationinfo(sim)
```

Passing `breaktol` stops early once the change in bond weights per unit time drops below it, which is
how a steady state or a converged thermal state is detected.

## Computing observables

Expectation values are taken in a reduced density matrix, which requires contracting the boundary
around the sites left untraced. `DensityMatrix` runs that boundary contraction once, at the given
boundary bond dimension, and is worth keeping rather than rebuilding for every observable.

```@repl
dm = DensityMatrix(quantumstate(sim), 8);
isconverged(dm)
rdm = partialtrace(dm, (1, 1));
real(expval(rdm, Z))
```

`partialtrace` takes any number of sites, which need not be adjacent, and `expval` then takes one
operator per site in the order the sites were given.

```@repl
pair = partialtrace(dm, (1, 1), (2, 2));
real(expval(pair, X, X)) - real(expval(pair, X, one(X))) * real(expval(pair, one(X), X))
```

`correlator` is the same computation written as a two-point function, and `negativity` measures the
entanglement of a two-site reduced density matrix.

```@repl
real(correlator(dm, X, (1, 1), X, (1, 2)))
negativity(partialtrace(dm, (1, 1), (1, 2)))
```

The correlation length is read straight off the boundary environment and needs no partial trace.

```@repl
correlationlength(dm)
```

## Plotting the trajectory

```@repl
using Makie, CairoMakie # Make sure this is `add`ed to your environment!
scatterlines(times, magn)
save("quick-start-magn-plot.svg", current_figure()); nothing # hide
```

![Magnetisation plot](quick-start-magn-plot.svg)
