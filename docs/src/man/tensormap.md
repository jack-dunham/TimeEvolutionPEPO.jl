```@meta
CurrentModule = TimeEvolutionPEPO
ShareDefaultModule = true
DocTestSetup = quote
    using TimeEvolutionPEPO, TensorKit
end
```
# Conventions and the `TensorMap` type

This package uses of the `TensorMap` data structure exported by the [TensorKit](https://github.com/Jutho/TensorKit.jl) package.
You should read the relevant [TensorKit documentation](https://jutho.github.io/TensorKit.jl/stable/man/tutorial/) for details on the implementation of this data structure, however here we provide a brief summary of the essentials relevant to this packages function.
When using TimeEvolutionPEPO, you should always load TensorKit alongside:
```@example
using TimeEvolutionPEPO, TensorKit
```

## Conventions

TimeEvolutionPEPO follows PEPSKit's `PEPOTensor` convention, in which virtual indices are assigned to the domain and physical indices to the codomain, with the virtual indices ordered north, east, south, west.
Specifically, the ordering of PEPO tensors in this package can be summarized by the following code;
```@repl
ket = bra = ℂ^2; # Typical phyiscal Hilbert space dimension
north = east = south = west = ℂ^4; # Example virtual dimension
pepo = rand(ket * bra', north * east * south' * west');
space(pepo)
```
Note the dual spaces on the `bra`, `south`, and `west` vector spaces.
!!! warn
    The convention here is to order the ket space before the bra space such that 
    ```math
    A \rho B \to (A \otimes B^{\top}) | \rho \rangle\rangle
    ```
    where ``| \rho \rangle\rangle`` is the vector obtained by concatenating the *rows* together.
    Julia, being a language with column-major ordering, employs the opposite convention with respect to the `reshape` function:
    ```@repl
    ρ = ["a" "b"; "c" "d"]
    reshape(ρ, 4)
    ```
    In terms of local operator transformations:
    ```@repl
    ρ = rand(2,2); A = rand(2,2); B = rand(2,2);
    reshape(kron(B', A) * reshape(ρ, 4), 2, 2) ≈ A * ρ * B
    reshape(kron(A, B') * reshape(ρ, 4), 2, 2) ≈ A * ρ * B
    ```
    This is worth bearing in mind if you wish to import a vectorized density matrix obtained outside of this package.

## Conversion to and from a dense array

If you wish to convert a `TensorMap` to an `Array`, then it is straightforward to do so:
```@repl
s = ℂ^2
convert(Array, rand(s * s, s))
```
however this is not a free operation.
Conversely, a `TensorMap` can be constructed directly by supplying the data as an `Array` whose `size` matches the space supplied to the `TensorMap` constructor:
```@repl
data = rand(2,2,2)
TensorMap(data, s * s, s)
TensorMap(data, s, s * s)
```

## Irrep ordering in symmetric spaces

You may wish to make use of a symmetric local vector space for simulations that, for example, preserve total spin or particle number.
When constructing symmetric vector spaces, TensorKit prescribes and ordering of the basis vectors that may not align with what you expect.
For example, consider a spin-1 particle. 
You may wish to make use of a possible U(1) symmetry by using writing the basis in terms of the sectors of well-defined ``z``-axis spin projection.
The typical ordering of the basis is then ``\{| 1 \rangle, | 0 \rangle, | -1 \rangle \}`` such that the spin-``z`` operator looks like:
```math
S_z = \begin{pmatrix}
    1 & 0 & 0 \\
    0 & 0 & 0 \\
    0 & 0 & -1
\end{pmatrix}
```
However the order assumed by TensorKit is different:
```@repl
U1Space(1 => 1, 0 => 1, -1 => 1)
```
The basis of charge sectors is ordered instead like ``\{| 0 \rangle, | 1 \rangle, | -1 \rangle \}``.
When constructing a `TensorMap` with symmetric vector spaces, one should make sure the data adheres to correct charge sector ordering, permuting the rows and columns of a matrix written in the conventional basis before handing it over:
```@repl
Sz = SPIN1[3]
perm = [2, 1, 3] # swap the first two basis vectors
Sz[perm, perm]
```
