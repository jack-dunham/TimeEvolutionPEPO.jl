# TimeEvolutionPEPO.jl

*A Julia package implementing algorithms to simulate many-body open quantum physics on two-dimensional lattices.*

[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://jack-dunham.github.io/TimeEvolutionPEPO.jl/dev)

> **_NOTE:_** This package is in the pre-alpha stage, as such no basic functionality should be assumed.

## Installation

This package is unregistered (and currently private). 
It also depends on another unregistered package [InfiniteTensorContractions.jl](https://github.com/jack-dunham/InfiniteTensorContractions.jl), which should be installed first.
From the Pkg REPL, accessed by typing the `]` character from the Julia prompt, run
```
pkg> add https://github.com/jack-dunham/InfiniteTensorContractions.jl#dev
```
to install the development version of this package.
You should then run:
```
pkg> add https://github.com/jack-dunham/TimeEvolutionPEPO.jl#dev
```
to add the development version of *this* package.
Note this will likely require you to authenticate with Github.
You might instead want to add this packages programmatically via the Pkg package API:
```julia
julia> using Pkg; Pkg.add(url="https://github.com/jack-dunham/TimeEvolutionPEPO.jl", rev="dev")
```
Finally, you should almost certainly add the [TensorKit.jl](https://github.com/Jutho/TensorKit.jl) package as an explicit dependency to your environment.
```
pkg> add TensorKit
```
These latter two pacakages can then be loaded into your Julia session using:
```julia
using TimeEvolutionPEPO
using TensorKit
```
which can be executed in the Julia REPL, or included at the top of a script. 
Loading [InfiniteTensorContractions.jl](https://github.com/jack-dunham/InfiniteTensorContractions.jl) is optional, as TimeEvolutionPEPO.jl exports most of it's relevent functionality.

## Related packages

This package is inspired by the package (for one-dimensional systems):

- [MPSKit.jl](https://github.com/QuantumKitHub/MPSKit.jl)

and is similar to

- [PEPSKit.jl](https://github.com/QuantumKitHub/PEPSKit.jl)

which is focused more on ground state optimisation for closed systems, rather than time evolution of open systems.
I encourage you to check out either of these packages as they may be more suited to your needs. 
Both of those packages, as well as this package, make use of the [TensorKit.jl](https://github.com/Jutho/TensorKit.jl) as a backend for performing (possibly) symmetric tensor network contractions.

There are also some other large tensor network packages written in Julia such as:
- [ITensors.jl](https://github.com/ITensor/ITensors.jl) and the accompanying ecosystem of high-level code for solving quantum many body problems.
- [Tenet.jl](https://github.com/bsc-quantic/Tenet.jl)

WIP
