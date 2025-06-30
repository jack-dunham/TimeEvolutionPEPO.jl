# Implementation and interfaces

The top level function that is called every time step during a call to [`simulate!`](@ref) is the `timestep` function with the following signature:
```julia
timestep!(pepo, truncator_list, propagator_list, truncalg; kwargs...)
```
This then calls:
```julia
trotterstep!(pepo, truncator, propagator, truncalg; kwargs...)
```
on each element of for each `truncator` and `propagator` in the lists.
Either of these methods can be specialized on to define custom functionality, however you may only need to customize small sections of the code, so various other interfaces for doing so are exposed and described in this section of the documentation.

## Customizing simple update

Simple update and similar methods (such as SVDU) can by customized by defining a subtype of `AbstractSimpleTruncationScheme` and then specializing on one of the following methods, in order of scope:

| Method                      | Signature       | Default behaviour |
|:--- |:----------------------- |:---------------------------------------------------------------------------------------- |
| A | `updatetensors(pair, truncalg::AbstractSimpleTruncationScheme, bond::Bond)` | Truncates the `bond` connecting `pair` using the truncated SVD | 
| B | `updatetensors(pair, environment, truncalg::AbstractSimpleTruncationScheme, bond::Bond)` | Assumes `environment` are the bond weights and absorbs these into `pair` before called method A. The bond weights are then removed from the new pair of tensors. |
| C | `updatetensors(pepo, environment, propagator, truncalg::AbstractSimpleTruncationScheme, bond::Bond)` | Applies `propagator` to the pair of tensors `pepo[bond]` before calling method B |

Each successive method in the above table allows you to customize the implementation at a point higher in the call stack, therefore you should specialize on the simplest method that allows you to do what you need to do.
All three methods have defaults for `AbstractSimpleTruncationScheme`, so for example you can specialize on method C and still make use of the default implementations of method A and B if desired.

For example, to define a simple update method that in addition to truncating to dimension `D` (the default), also truncates small singular values below a threshold, one can first define the struct:
```julia
struct MySimpleUpdate <: AbstractSimpleTruncationScheme
    tol::Float64
end
```
and then define the method
```julia
function updatetensors(evolvedpair, truncalg::MySimpleUpdate, bond)

    # This is the only bit of information we actually need from `bond`.
    axis = bondaxis(bond)

    Q1, R, L, Q2 = reducevirtual(evolvedpair, axis); # see `?reducevirtual`

    # The bond has vector space D * η, so D = domain(R, 1).
    D = domain(R, 1)
    
    # Truncate to D, then truncate small singular-values further (see `?TensorKit.tsvd`)
    trunc = truncdim(dim(D)) & truncbelow(truncalg.tol)

    u, s, v, _ tsvd!(normalize!(R * L); trunc=trunc)

    # Normalize to unit trace
    tracenormalize!(s)

    newpair = truncate(Q1 => Q2; u * sqrt(s) => sqrt(s) * v, axis) # see `?truncate`

    # We always need to return the new bond weight, even though it's absorbed.
    return newpair, s
end
```
which well be called whenever `MySimpleUpdate` is used as a truncation algorithm.

### Advanced customization using environments

For simple update methods, the [bond weights](@ref bondweights) of the PEPO play the role of the environment, and therefore one does not *need* to compute an environment to perform truncations. 
Nevertheless, an environment can still be specified if desired.

During the update of each bond, a call to the function `environment!` is made. 
This has the following default behaviour:
```julia
environment!(truncator, pepo, truncalg::AbstractSimpleTruncationScheme, bond::Bond) = environment(pepo, truncalg, bond)
```
The simpler, non-mutating method then defaults to:
```julia
environment(pepo, truncalg::AbstractSimpleTruncationScheme, bond::Bond) = bondweights(pepo)
```
Either of these methods can be specialized on, with the return value getting passed into method C of `updatetensors` as the `environment` argument.

If the mutating version is used, then the following default should also be overridden:
```julia
newtruncators(pepo, trotterlist, truncalg::AbstractSimpleTruncationScheme) = SimpleUpdateState()
```
This should return either the `truncator` argument to `environment!(truncator,...)` of a vector of length `length(trotterlist)` if a different `truncator` is required for each trotter step in a given time step.
This allows information to be passed between time steps that can be used to compute environments.
For example, you may wish to use the `truncator` from the previous time step as a initial guess for some optimisation procedure that constructs an environment. 
If this is not required, then the simpler `environment` method can be used.
