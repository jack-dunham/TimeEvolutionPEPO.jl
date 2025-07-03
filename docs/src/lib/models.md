# Models

```@meta
CurrentModule = TimeEvolutionPEPO
```
## Single terms

Terms are either one-dimensional (axial) with support on a single axis of a lattice only, or two-dimensional.
In the former case, lattice site indices are denoted with scalar sub-scripts representing the position on a lattice axis.
In the latter, vector subscripts are used.

```@docs
LocalOp
ExpOp
GaussOp
PowOp
ExpOp2d
GaussOp2d
PowOp2d
```

## Models and sums of terms

```@docs
Model
Hamiltonian
Dissipator
Liouvillian
TrotterLayer
Ising
```
