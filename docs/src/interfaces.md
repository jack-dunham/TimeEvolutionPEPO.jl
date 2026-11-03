# Implementation and interfaces

```@meta
CurrentModule = TimeEvolutionPEPO
```

## The time step

[`simulate!`](@ref) calls one function per time step,

```julia
timestep!(state, weights, decomposition; trunc, truncalg, envalg)
```

where `decomposition` is the vector [`trotterise`](@ref) produced: `(propagators, axis)` pairs for
nearest-neighbour layers, and a [`PEPSOLayer`](@ref) for each long-range one (see below). That sweeps
each pair in turn,

```julia
axissweep!(state, weights, propagators, axis; trunc, truncalg, env)
```

which walks every bond of the unit cell along `axis` and updates it,

```julia
bondstep!(state, weights, propagator, sites; trunc, truncalg, env)
```

`propagators` is a `Matrix` of two-site propagators, indexed as the originating layer's term matrix
was: a `1×1` matrix for a uniform layer, or one matching the state's unit cell for a site-dependent
one. `axissweep!` picks out the entry for each bond and passes that single `propagator` to
`bondstep!`.

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

Four keywords select the truncation, and they thread unchanged from `simulate!` down to
`bondstep!`.

| Keyword | Effect |
|:--- |:--- |
| `trunc` | A TensorKit `TruncationStrategy` for the SVD of the evolved pair. Defaults to the space the bond already has, so the bond dimension is preserved. |
| `truncalg` | A `FullEnvTruncation` or `ALSTruncation`, which optimises the truncation against a bond environment instead of taking a bare SVD. |
| `envalg` | A [`BondEnvironment`](@ref), giving the boundary bond dimension and tolerances for the CTMRG solve that builds that environment. Required with `truncalg`, and unused without it. |
| `zeta` | The damping parameter. After each bond is truncated its spectrum is multiplied by `Ξ(ζ)ᵢⱼ = δᵢⱼ exp(-(i-1)/ζ)` and renormalised ([`dampweight`](@ref)), suppressing the correlation length of the evolved state. `Inf` leaves the update unchanged. |

Without `truncalg` the bond weights play the role of the environment and no boundary contraction is
needed — this is simple update. With it, one CTMRG boundary is solved for per axis sweep, warm-started
from the previous one, and each bond is truncated against the environment [`truncatebond`](@ref)
builds from it.

```julia
alg = FullEnvTruncation(; trunc = truncspace(ComplexSpace(4)), maxiter = 30)
simulate!(sim; numsteps = 20, truncalg = alg, envalg = BondEnvironment(16))
```

The damping is applied inside the bond update, between the SVD and `PEPSKit.absorb_s`, so it
compounds over the sweep and feeds back through every later truncation.

```julia
simulate!(sim; numsteps = 20, trunc = truncrank(9), zeta = 100.0)
```

## Long-range layers

A layer holding a long-range term is not split into bonds. [`trotterise`](@ref) builds the whole
layer as one propagator with [`fsmpepso`](@ref), whose virtual bond dimension is the number of
finite-state-machine states, and wraps it in a [`PEPSOLayer`](@ref). `timestep!` applies it with

```julia
iterativestep!(state, weights, layer, alg)
```

the iterative simple update of the FCLS manuscript (Appendix D). The propagator is contracted onto
every site, and each bond is cut against the projectors chosen for the other legs of its two sites,
those legs weighted by their bond weights; the cuts are repeated, all bonds at once, until the
weights change by less than `tol`. The layer keeps its final projectors, and the next application
starts from them. The contractions run in parallel under PEPSKit's scheduler, so
`PEPSKit.Defaults.set_scheduler!(:serial)` runs them one at a time.

The settings are an [`IterativeSU`](@ref), passed as the `iterative` keyword of `simulate!` or field of
[`DampedUpdate`](@ref); `trunc` sets the bond dimension as for simple update.

```julia
simulate!(sim; numsteps = 400, trunc = truncrank(9),
    iterative = IterativeSU(; mu = 10.0))
```

With `mu = 0`, the default, each cut keeps the bond's top singular directions. A positive `mu` adds to
the error the cut minimises the relative change of the nearest-neighbour two-site density matrix
across the bond, as belief propagation on the traced network sees it, weighted by `mu`. This soft
cut needs spaces without symmetry.

A long-range layer takes no `truncalg` and no finite `zeta`; both raise an `ArgumentError`.

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
