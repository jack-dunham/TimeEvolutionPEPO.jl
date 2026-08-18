# Implementation and interfaces

```@meta
CurrentModule = TimeEvolutionPEPO
```

## The time step

[`simulate!`](@ref) calls one function per time step,

```julia
timestep!(state, weights, decomposition; trunc, truncalg, envalg)
```

where `decomposition` is the vector of `(propagator, axis)` pairs [`trotterise`](@ref) produced. That
sweeps each pair in turn,

```julia
axissweep!(state, weights, propagator, axis; trunc, truncalg, env)
```

which walks every bond of the unit cell along `axis` and updates it,

```julia
bondstep!(state, weights, propagator, sites; trunc, truncalg, env)
```

A bond update absorbs the neighbouring weights, splits both site tensors into a reduced bond tensor
and an orthogonal remainder ([`reducedbond_first`](@ref), [`reducedbond_last`](@ref)), applies the
propagator to the pair ([`applypropagator`](@ref)), truncates the bond between them, and writes the
two tensors and the new weight back.

The pieces that make this an open-system update rather than PEPSKit's simple update are the reduced
bond split and the propagator application: PEPSKit's `bond_tensor_first`/`bond_tensor_last` peel off
one physical leg at a time, because a two-site gate acts on the ket and the bra in separate passes,
whereas a Liouvillian acts on both at once. Everything else — cluster extraction, weight absorption,
bond rotation and the truncations — is PEPSKit's and is reused unchanged.

## Choosing how bonds are truncated

Three keywords select the truncation, and they thread unchanged from `simulate!` down to
`bondstep!`.

| Keyword | Effect |
|:--- |:--- |
| `trunc` | A TensorKit `TruncationStrategy` for the SVD of the evolved pair. Defaults to the space the bond already has, so the bond dimension is preserved. |
| `truncalg` | A `FullEnvTruncation` or `ALSTruncation`, which optimises the truncation against a bond environment instead of taking a bare SVD. |
| `envalg` | A [`BondEnvironment`](@ref), giving the boundary bond dimension and tolerances for the CTMRG solve that builds that environment. Required with `truncalg`, and unused without it. |

Without `truncalg` the bond weights play the role of the environment and no boundary contraction is
needed — this is simple update. With it, one CTMRG boundary is solved for per axis sweep, warm-started
from the previous one, and each bond is truncated against the environment [`truncatebond`](@ref)
builds from it.

```julia
alg = FullEnvTruncation(; trunc = truncspace(ComplexSpace(4)), maxiter = 30)
simulate!(sim; numsteps = 20, truncalg = alg, envalg = BondEnvironment(16))
```

## Adding a truncation scheme

A truncation scheme is a type PEPSKit's `bond_truncate` dispatches on, so a new one is added by
defining

```julia
PEPSKit.bond_truncate(a::MPSTensor, b::MPSTensor, benv::BondEnv, alg::MyTruncation)
```

returning the updated pair, the bond weight and an info object carrying a `fid` field.
[`truncatebond`](@ref) handles the rest: it splits the evolved pair, fuses each site's ket and bra
into the single physical leg `bond_truncate` expects, calls it, and unfuses on the way out. The state
itself keeps ket and bra apart throughout; the fusion exists only for the duration of that call.

The environment `benv` is whatever [`bondenvironment`](@ref) and PEPSKit's `bondenv_ctm` produce for
the bond, an `AbstractTensorMap{T,S,2,2}`. Supplying a different environment — a cluster environment
rather than a boundary one, say — means computing that tensor instead and passing it to
`bond_truncate` yourself.

## Fixing the gauge

[`gaugefix!`](@ref) rewrites a state and its weights into a fixed gauge in place, using either trivial
simple update or a belief-propagation fixed point.

```julia
gaugefix!(sim)                                  # SUGauge
env, info = bpenvironment(quantumstate(sim))
gaugefix!(sim, BPGauge(), env)
```

Both algorithms are PEPSKit's. For the PEPO case PEPSKit fuses the two physical legs internally and
unfuses on the way out, so the state stays a density-matrix PEPO throughout.
