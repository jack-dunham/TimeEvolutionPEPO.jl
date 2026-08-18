# Methods and algorithms

```@meta
CurrentModule = TimeEvolutionPEPO
```

## Propagators

```@docs
trotterise
bondpropagator
generator
```

## The sweep

```@docs
timestep!
axissweep!
bondstep!
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
