# Models

```@meta
CurrentModule = TimeEvolutionPEPO
```
## Single terms

```@docs
LocalOp
LocalOp(D::AbstractMatrix)
LocalOp(C,B,A; l)
LocalOp(CAB)
ExpOp
PowOp
PowOp2d
GaussOp
GaussOp2d
```

## Models and sums of terms

```@docs
Model
Hamiltonian
Dissipator
Liouvillian
TrotterLayer
```
