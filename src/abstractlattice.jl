abstract type AbstractLattice{G,ElType,A} <: ITC.AbstractUnitCell{G,ElType,A} end

struct Lattice{G,ElType,A} <: AbstractLattice{G,ElType,A}
    data::UnitCell{G,ElType,A}
    issymmetric::Bool
end

const SquareLattice = Lattice{Square}

## Fields
ITC.getdata(lattice::Lattice) = lattice.data
issymmetric(lattice::Lattice) = lattice.issymmetric

## CONSTRUCTORS

Lattice(data, issymmetric=false) = Lattice{Square}(data, issymmetric)
Lattice{G}(data, issymmetric=false) where {G} = Lattice(UnitCell{G}(data), issymmetric)

function Base.similar(lat::L) where {L<:Lattice}
    return L(similar(ITC.getdata(lat)), issymmetric(lat))
end
function Base.similar(lat::L, dims::Tuple) where {L<:Lattice}
    return L(similar(ITC.getdata(lat), dims), issymmetric(lat))
end
function Base.similar(lat::Lattice{G}, ::Type{T}) where {G,T}
    return Lattice{G}(similar(ITC.getdata(lat), T), issymmetric(lat))
end

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
function getbonds(L::AbstractLattice, dir=1)
    if dir in (1, :x, :h, :horizontal)
        incr = CartesianIndex((1, 0))
    elseif dir in (2, :y, :v, :vertical)
        incr = CartesianIndex((0, 1))
    else
        throw(
            ArgumentError(
                "Use 1 (2), :x (:y), :h :(v), or :horizontal (:vertical) to specify horizontal (vertical) bonds",
            ),
        )
    end

    Nx, Ny = size(L)
    rx = 1:Nx
    ry = 1:Ny

    func = x -> Bond(x, CartesianIndex(mod.(Tuple(x + incr), (rx, ry))))

    bonds = map(func, eachindex(L))

    return bonds
end

function nextbond(bond::Bond)
    bond_diff = bond.right - bond.left
    return Bond(bond.right, bond.right + bond_diff)
end
function prevbond(bond::Bond)
    bond_diff = bond.right - bond.left
    return Bond(bond.right, bond.right - bond_diff)
end

function Base.getindex(lattice::AbstractLattice, bond::Bond)
    @inbounds rv = lattice[bond.left], lattice[bond.right]
    return rv
end

function Base.setindex!(
    lattice::AbstractLattice{G,T}, vals::Tuple{T,T}, bond::Bond
) where {G,T}
    val1, val2 = vals
    lattice[bond.left] = val1
    lattice[bond.right] = val2
    return lattice
end
function Base.copyto!(lattice::AbstractLattice, val1, val2, bond::Bond)
    copy!(lattice[bond.left], val1)
    copy!(lattice[bond.right], val2)
    return lattice
end

abstract type AbstractLatticeStyle{G,A} <: ITC.AbstractUnitCellStyle{G,A} end
struct LatticeStyle{G,A} <: AbstractLatticeStyle{G,A} end

function Base.BroadcastStyle(::Type{Lattice{G,ElType,A}}) where {G,ElType,A}
    return LatticeStyle{G,typeof(Base.BroadcastStyle(A))}()
end

function Base.similar(
    bc::Broadcast.Broadcasted{LatticeStyle{G,A}}, ::Type{ElType}
) where {G,ElType,A}
    lattice = find_lat(bc)
    issym = false
    if lattice === nothing
        issym = false
    else
        issym = issymmetric(lattice)
    end
    return Lattice{G}(similar(Base.convert(Broadcast.Broadcasted{A}, bc), ElType), issym)
end

find_lat(bc::Base.Broadcast.Broadcasted) = find_lat(bc.args)
find_lat(args::Tuple) = find_lat(find_lat(args[1]), Base.tail(args))
find_lat(x) = x
find_lat(::Tuple{}) = nothing
find_lat(a::AbstractLattice, rest) = a
find_lat(::Any, rest) = find_lat(rest)

function uniqueindices(L::Lattice)
    Nx, Ny = size(L)
    if issymmetric(L)
        return CartesianIndices((Nx, 1))
    else
        return CartesianIndices((Nx, Ny))
    end
end
