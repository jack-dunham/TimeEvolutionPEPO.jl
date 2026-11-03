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

## Trotter layers and models

A [`Model`](@ref) is an ordered list of [`TrotterLayer`](@ref)s, swept in order once per time step.
Terms placed on the same layer are exponentiated together rather than Trotterised apart. Build one
with `push!`, which always appends a new layer:
```@repl
XX = LocalOp(X,X)
YY = LocalOp(Y,Y)
model_A = Model{Hamiltonian}()
push!(model_A, XX)
push!(model_A, YY)
push!(model_A, ZZ) # ZZ defined in the previous section
```
or sum several terms into a single layer, which puts them on one exponential rather than one
Trotter step apiece:
```@repl
model_B = Model{Hamiltonian}()
push!(model_B, XX + YY + ZZ)
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
Each entry is a `(propagators, axis)` pair, where `propagators` is a matrix of two-site propagators
(a `1×1` matrix for a uniform layer). A layer with no axis is applied on both, with its one-local
part halved so that a full time step applies it once per site rather than twice; passing an axis to
[`TrotterLayer`](@ref) restricts it to that axis alone:
```@repl
last.(trotterise([TrotterLayer(XX, 1), TrotterLayer(YY, 2)], P, 0.1))
```
To add a term to every existing layer instead of giving it a layer of its own, use
[`distribute!`](@ref). It scales the term by `1 / length(model)` so that a full time step applies it
once in total — useful for a local field that should act alongside every coupling:
```@repl
model_C = Model{Hamiltonian}()
push!(model_C, XX)
push!(model_C, YY)
distribute!(model_C, X0)
```
`Model`'s type parameter tracks whether dissipation is present: a `Model{Hamiltonian}` refuses a
dissipative term, so a purely unitary model cannot pick one up by accident.
```@repl
try
    push!(model_C, Dissipator(Z))
catch e
    println(e)
end
```

## Site-dependent terms

A [`TrotterLayer`](@ref) can hold a different term at every site of the unit cell instead of one
term shared by all of them. Build it from a matrix: `data[r, c]` is anchored at site `(r, c)`, its
two-local part coupling `(r, c)` to its neighbour in the `+axis` direction (east for axis 1, south
for axis 2), and its one-local part belonging to `(r, c)` alone.
```@repl
staggered = TrotterLayer([XX 0.5*YY; YY XX], 1)
model_D = Model{Hamiltonian}()
push!(model_D, staggered)
size(first(only(trotterise(model_D, P, 0.1))))
```
A `1×1` layer is uniform and serves any unit cell, since indexing into a `TrotterLayer` wraps with
`mod1`. Passing a matrix that matches neither `(1, 1)` nor the state's own unit cell size is an
error raised by [`Simulation`](@ref) when the decomposition is built, not by `TrotterLayer` itself.

## Specific model components

In some cases, terms can have analytical expressions when exponentiated, benefit from more bespoke optimisation beyond simple low-rank compression, or are can be put in forms with specific properties that may be desirable.
The model interface allows for this. 
For example, an Ising term coupled with optional local operators and a optional dissipation is specified with the `Ising` type, which defines a specific, optimised exponentiation which can be used instead of the more general `LocalOp` constructor.
```@repl
model = Ising(LocalOp(X, X), Dissipator(Z))
state = productpepo((0, 0, -1), 2);
Simulation(state, model; timestep = 0.1);
```
