const BondSpace{L,S<:IndexSpace} = OnLattice{NotTensor,L,S}

function BondSpace(L::AbstractLattice, data::AbstractMatrix{<:IndexSpace})
    return OnLattice(NotTensor(), L, data)
end

abstract type AbstractBonds{L<:AbstractLattice,S<:IndexSpace} end

struct Bonds{L,S} <: AbstractBonds{L,S}
    west::OnLattice{NotTensor,L,S}
    south::OnLattice{NotTensor,L,S}
end

function Bonds(lattice::AbstractLattice, west, south=west)
    mat_west = spacematrix(typeof(lattice), west)
    mat_south = spacematrix(typeof(lattice), south)

    sp_west = OnLattice(NotTensor(), lattice, mat_west)
    sp_south = OnLattice(NotTensor(), lattice, mat_south)
    return Bonds(sp_west, sp_south)
end
