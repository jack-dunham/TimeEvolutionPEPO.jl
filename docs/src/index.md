# TimeEvolutionPEPO.jl

TimeEvolutionPEPO.jl is a high-level and domain-specific package for simulating the time-evolution of an open-quantum system represented by the iPEPO ansatz.
A variety of options are implemented under one common interface for both Lindblad real-time evolution and thermal state annealing.

## Package Features

This package may be of use if you wish to study quantum many-body systems that are
1. defined on a translational- or shift-invariant two-dimensional lattice.
2. evolving according to the Lindblad equation or described by a thermal density matrix at finite temperature.

The types of interactions that can be simulated are fairly broad including:
- nearest neighbour interactions,
- two-body interactions at an arbitrary distance on an axis,
- next-nearest neighbour interactions along the diagonal,
- non-local interactions decaying exponentially,
- approximate forms of power-law decaying interactions.

In the case of Lindblad time evolution, currently only on-site local dissipation is supported. 

Multiple methods are included. For example:
- the simple update (and related) method,
- full environment truncation 
- neighbourhood tensor update (with respect to nearest neighbours only).
