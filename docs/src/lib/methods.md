# Methods and algorithms

```@meta
CurrentModule = TimeEvolutionPEPO
```

## Propagators

```@docs
trotterise
fsmpepso
bondpropagator
generator
```

## The sweep

```@docs
DampedUpdate
timestep!
axissweep!
bondstep!
```

## Long-range layers

```@docs
IterativeSU
PEPSOLayer
iterativestep!
```

## Bond truncation

```@docs
BondEnvironment
bondenvironment
truncatebond
applypropagator
reducedbond_first
reducedbond_last
undo_reducedbond_first
undo_reducedbond_last
fusephysical
```

## Gauge fixing

```@docs
gaugefix!
bpenvironment
```
