```@meta
CurrentModule = TimeEvolutionPEPO
ShareDefaultModule = true
```
```@setup
using TimeEvolutionPEPO, TensorKit
```
# Constructing models

One of the strengths of tensor network methods is that, by working directly with state vectors and matrix representations of time-evolution operators, they do not require deriving specific equations for different models.
For this reason, TimeEvolutionPEPO aims to have very flexible support for implementing different models.
!!! note
    The set of models that can be constructed in this package is much larger than the set of models that a tensor network ansatz can support. 
    You may find that for many models, too much entanglement is generated and the simulation breaks down. 

## Terms, Hamiltonians, and Liouvillians

A model is a term that is a subtype of `AbstractTerm`, or a vector of them giving the layers of a
Suzuki-Trotter decomposition.
Several terms summed into one layer are exponentiated together rather than Trotterised apart.

There are a large number of possible terms that can be represented, however the most common are nearest-neighbour and one-local terms.
Both cases are represented using the `LocalOp` structure:
```@repl
X,Y,Z = PAULI # A convenience `Tuple` containing the 3 Pauli matrices.
X0 = LocalOp(X) # One-local term 
ZZ = LocalOp(Z, Z) # Nearest neighbour term
```
The terms defined above are objects in the following type hierarchy:
```@repl
typeof(X0) <: LocalOp{1} <: AbstractSingleTerm <: AbstractTerm
typeof(ZZ) <: LocalOp{2} <: AbstractSingleTerm <: AbstractTerm
```
Terms can be scaled,
```@repl
0.5 * ZZ
```
and added together to create `Hamiltonian` objects:
```@repl
H = 1.5 / 4 * ZZ - X0
```
which can also be scaled added to other terms
```@repl
H = 0.5 * H + LocalOp(X,Y) + LocalOp(Y,X)
```
Adding a `Dissipator` to a `Hamiltonian` constructs a `Liouvillian` object:
```@repl
L = H + Dissipator(Z, 0.5 * (X - im * Y))
```
A `Dissipator` can be constructed with any number of Lindblad jump operators as positional arguments to the constructor.

## Trotter layers

Passing a vector of terms rather than a single term puts each element on its own Suzuki-Trotter
layer, which are then swept in order.
As an example, consider the spin-``1/2`` Heisenberg Hamiltonian, written one coupling per layer:
```@repl
XX = LocalOp(X,X)
YY = LocalOp(Y,Y)
model_A = [XX, YY, ZZ] # ZZ defined in the previous section
```
and the same model summed into a single layer:
```@repl
model_B = XX + YY + ZZ
```
The difference shows up in the Trotter decomposition. `model_A` gives the time evolution operator
```math
e^{\tau H_{A}} \approx e^{\tau X \otimes X} e^{\tau Y \otimes Y} e^{\tau Z \otimes Z}
```
whereas `model_B` gives
```math
e^{\tau H_{B}} \approx e^{\tau \left( X \otimes X + Y \otimes Y + Z \otimes Z \right)}
```
The first commits a Trotter error the second does not, and costs three truncation steps per sweep
instead of one; the second exponentiates an operator of higher rank.
[`trotterise`](@ref) performs this decomposition, and a `Simulation` calls it on construction:
```@repl
P = ComplexSpace(2)
length(trotterise(model_A, P, 0.1))
length(trotterise(model_B, P, 0.1))
```
Each entry is a `(propagator, axis)` pair. A layer given as a bare term is applied on both axes, with
its one-local part halved so that a full time step applies it once per site rather than twice; a layer
written as `term => axis` is restricted to that axis alone.
```@repl
last.(trotterise([XX => 1, YY => 2], P, 0.1))
```

## Specific model components

In some cases, terms can have analytical expressions when exponentiated, benefit from more bespoke optimisation beyond simple low-rank compression, or are can be put in forms with specific properties that may be desirable.
The model interface allows for this. 
For example, an Ising term coupled with optional local operators and a optional dissipation is specified with the `Ising` type, which defines a specific, optimised exponentiation which can be used instead of the more general `LocalOp` constructor.
```@repl
model = Ising(X, Dissipator(Z))
state = productpepo((0, 0, -1), 2);
Simulation(state, model; timestep = 0.1);
```
