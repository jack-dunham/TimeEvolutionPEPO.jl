# PEPOKit.jl

> **_NOTE:_** This package is in the pre-alpha stage, as such no basic functionality should be assumed.

A Julia package implementing algorithms for the simulation of driven-dissipative many-body quantum physics in two-dimensions. Based on the excellent [TensorKit.jl](https://github.com/Jutho/TensorKit.jl) package, and inspired by the [MPSKit.jl](https://github.com/maartenvd/MPSKit.jl) project.


## Usage

As all contractions are implemented statically, this package requires a matrix product operators to be in standard from. An API exists to allow for custom MPO forms, however this requires one to implement a fair number `@tensor` macros. 

The index-structure of an MPO is encoded in its `MPOSignature`---the associated boundary MPS and left/right environment tensors will also carry this signature. All `MPOTensors` in the same `UnitCell` must have the same `MPOSignature`. For example, it is not yet possible (and likely never will be possible) to have a unit-cell where the MPO tensors do not all have the same number of indices in the codomain and domain on each side. The dimension of these indices however, may vary. 

Two MPO signatures come built-in. The `TR_SIG` signature, and the `POST_HS_SIG` signature. The former is your standard 4-index signature with the west and south indices in the codomain and the north and east indices in the domain. Any construction 2-index by 2-index construction of a `MPOTensor` will apply this signature by default. The latter is the signature used in the `FET` algorithm during the truncation step.  
