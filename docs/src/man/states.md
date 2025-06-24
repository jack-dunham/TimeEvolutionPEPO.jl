```@meta
CurrentModule = TimeEvolutionPEPO
ShareDefaultModule = true
```
```@setup
using TimeEvolutionPEPO, TensorKit, TensorRenormalizationGroups
```
# State representation using the `PEPO` type

In the TimeEvolutionPEPO.jl package, the tensor network representation of the state is represented computationally using an object of the type `PEPO`.
For more details, and a full list of constructors of `PEPO` objects, please see the relevant [library documentation]().

## Construction

A `PEPO` can be construct directly by defining the local density matrices on each lattice site in the unit cell.
Currently, only product states can be constructed this way.
One must also choose the bond dimension for each bond in the unit cell.
```@repl
bonddim = 4
state = PEPO((0,0,-1), bonddim); # Initalise a spin-1/2 product state with all spins pointing down
rho = 0.5 * (one(PAULI[3]) - PAULI[3]) # Or construct rho from the bloch vector explictly
state == PEPO(rho, bonddim) # These are equivalent
```
We can also specify a non-trivial unit cell by passing matrices to the `PEPO` constructor: 
```@repl
rhoA = (0,0,-1)
rhoB = (0,0,+1)
size(PEPO([rhoA rhoB; rhoB rhoA], bonddim)) == (2, 2)
```
Passing an integer as the bond dimension to the `PEPO` constructor is a convenient shortcut for constructing a tensor network with a trivial (non-symmetric) local and bond space, i.e. one of type `ComplexSpace`.
```@repl
bondspace = ComplexSpace(bonddim)
bondspace === ℂ^bonddim # optional unicode short-hand for the above (from TensorKit.jl)
PEPO(rho, bonddim) == PEPO(rho, bondspace)
```
In this specific case, where spaces are trivial, usually the size of the local vector space can be obtained automatically from the first argument to the `PEPO` constructor.
When this is not the case, or if one wishes to specify a non-trivial space with a symmetry, the local space must also be specified as the second argument, with the third argument now representing the bond space.
```@repl
PEPO(rho, U1Space(-1=>1, 1=>1), U1Space(0=>4));
```
Of course, not all initial states are compatible with all symmetric vector spaces. For example:
```@repl
PEPO([0 1; 1 1], U1Space(-1=>1, 1=>1), U1Space(0=>4));
```
In the above, the products state defined by the input density matrix is not U(1) symmetric, and is therefore incompatible with this representation space.

## Interface

Objects of type `PEPO` subtype the `AbstractUnitCell` abstract type from the TensorRenormalizationGroups.jl backend package, i.e.
```@repl
PEPO <: TensorRenormalizationGroups.AbstractUnitCell
```
and therefore can be indexed similarly (with periodic boundary conditions).
While it is unlikely that you will need to work with the `PEPO` object directly, there are some convenience functions that maybe be useful in this case. 
The bond space of a `PEPO` can be increased  in dimension by calling:
```@repl
state_expanded = expandspace(state, ComplexSpace(6));
space(state_expanded[1,1])
```
One can also convert a symmetric `PEPO` (one defined with respect vector spaces of type `GradedSpace`) to aa trivial one using the `unsymmetrise` function.
If `PEPO` is already trivial, then this function has no effect.
```@repl
state === unsymmetrise(state)
```

## Converting to base Julia

If one wishes to strip the tensors out of the `PEPO` objects (for example to use in a different package),
then one can just broadcast to a matrix of the correct type:
```@repl
PType = tensortype(state)
matrix_of_tensormaps = Matrix{PType}(undef, size(state))
matrix_of_tensormaps .= state;
matrix_of_tensormaps[1,1] === state[1,1]
```
The `TensorMap` elements can then be converted to standard Julia arrays by broadcasting over `matrix_of_tensormaps`.

