abstract type AbstractLattice{G,ElType,A} <: ITC.AbstractUnitCell{G,ElType,A} end

struct Lattice{G,ElType,A} <: AbstractLattice{G,ElType,A}
    data::UnitCell{G,ElType,A}
    issymmetric::Bool
end

## Fields
ITC.getdata(lattice::Lattice) = lattice.data
issymmetric(lattice::Lattice) = lattice.issymmetric)

"""
    struct Bond
"""
struct Bond
    left::CartesianIndex{2}
    right::CartesianIndex{2}
end

left(b::Bond) = b.left
right(b::Bond) = b.right

function Base.show(io::IO, ::MIME"text/plain", bond::Bond)
    print("$(Tuple(bond.left)) ⟵⟶ $(Tuple(bond.right)) ")
    return nothing
end

"""
    getaxis(bond::Bond)

Returns the axis that `bond` lies on in a lattice.
"""
function getaxis(bond::Bond)
    bond_diff = bond.right - bond.left
    if bond_diff[2] == 0
        return 1
    elseif bond_diff[1] == 0
        return 2
    else
        throw(ErrorException("Impossible `else` reached!, Bond diff: $bond_diff"))
    end
end

"""
    getbonds(L::AbstractLattice) -> H,V

Returns a matrices of `Bond` objects `H` and `V`. The `Matrix{Bond}` object `H` contains all 
the horizontal bonds of the lattice `L`, with `V` containing the vertical bonds. For non-
square lattices, horizontal and vertical bond directions are defined by the mapping to the
square lattice.
"""
function getbonds(L::AbstractLattice)
    Nx, Ny = size(L)
    rx = 1:Nx
    ry = 1:Ny

    incr = CartesianIndex((1, 0))

    func = x -> Bond(x, CartesianIndex(mod.(Tuple(x + incr), (rx, ry))))

    bonds = map(func, eachindex(L))

    return bonds
end

function nextbond(bond::Bond)
    bond_diff = bond.right - bond.left
    return Bond(bond.right, bond.right + bond_diff)
end

function Base.getindex(lattice::AbstractLattice, bond::Bond)
    @inbounds rv = lattice[bond.left], lattice[bond.right]
    return rv
end
