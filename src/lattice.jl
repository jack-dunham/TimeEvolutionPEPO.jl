abstract type AbstractLattice{Nx,Ny,L<:LatticeStyle,S<:IndexSpace} <: AbstractLatticeLike{S} end

@doc raw"""
    struct Lattice{Nx,Ny,LStyle,S} <: AbstractLattice{Nx,Ny,LStyle,S}

Represents a lattice of size `Nx` by `Ny` of type `LStyle`. The physical space of the lattice 
is of type `S`

"""
struct Lattice{Nx,Ny,L,S<:IndexSpace} <: AbstractLattice{Nx,Ny,L,S}
    space::Matrix{S}
    issymmetric::Bool
end
function Lattice{Nx,Ny,B}(space::Matrix{S}, sym::Bool) where {Nx,Ny,B,S<:IndexSpace}
    return Lattice{Nx,Ny,B,S}(space, sym)
end

issymmetric(L::Lattice) = L.issymmetric
getmatrix(L::Lattice) = L.space
physpace(L::AbstractLattice) = getmatrix(L)

TensorKit.spacetype(lattice::AbstractLattice{Nx,Ny,L,S}) where {Nx,Ny,L,S} = S

function to_latticeindices(L::AbstractLattice, i::Vararg{Int,2})
    return to_latticeindices(boundarystyle(L), L, i...)
end
function to_latticeindices(::Periodic, L::AbstractLattice, i::Vararg{Int,2})
    Nx, Ny = size(L)
    return (mod(i[1], 1:Nx), mod(i[2], 1:Ny))
end
to_latticeindices(::Open, L::AbstractLattice, i::Vararg{Int,2}) = i

function Lattice{Nx,Ny,L,S}(undef::UndefInitializer) where {Nx,Ny,L,S}
    return Lattice{Nx,Ny,L}(Matrix{S}(undef, Nx, Ny))
end

function Base.similar(L::Lattice, issymmetric::Bool=issymmetric(L))
    return typeof(L)(similar(getmatrix(L)), issymmetric)
end

_similar_permutedtype(L) = _similar_permutedtype(typeof(L))
function _similar_permutedtype(LType::Type{<:Lattice{Nx,Ny,L,S}}) where {Nx,Ny,L,S}
    return Lattice{Ny,Nx,L,S}
end

function _similar_permuted(L)
    LType_new = _similar_permutedtype(L)
    return LType_new(undef)
end

Base.permutedims(l::AbstractLattice) = Base.permutedims!(_similar_permuted(l), l)
function Base.permutedims!(ldst::AbstractLattice, lsrc::AbstractLattice)
    permutedims!(ldst.space, getmatrix(lsrc), (2, 1))
    return ldst
end

# Transposing a lattice simply maps x -> y and y -> x when indexing. Used only when 
# algorithms are symmetric about an axis
struct TransposedLattice{Ny,Nx,L,S} <: AbstractLattice{Ny,Nx,L,S}
    parent::Lattice{Nx,Ny,L,S}
end

issymmetric(l::TransposedLattice) = issymmetric(l.parent)

LinearAlgebra.transpose(l::Lattice) = TransposedLattice(l)
LinearAlgebra.transpose(l::TransposedLattice) = l.parent

function to_latticeindices(t::TransposedLattice, i::Vararg{Int,2})
    return to_latticeindices(t.parent, i[2], i[1])
end
getmatrix(t::TransposedLattice) = getmatrix(t.parent)

Base.size(::Type{<:AbstractLattice{Nx,Ny}}) where {Nx,Ny} = (Nx, Ny)
Base.size(l::AbstractLattice) = size(typeof(l))

lattice(L::AbstractLattice) = L
latticestyle(::AbstractLattice{Nx,Ny,L}) where {Nx,Ny,L} = L()

getmatrix(L::AbstractLattice) = L.space

boundarystyle(l::AbstractLatticeLike) = boundarystyle(typeof(l))
boundarystyle(l::Type{<:AbstractLattice{Nx,Ny,L}}) where {Nx,Ny,L} = boundarystyle(L)

const UnitCell{Nx,Ny,S} = Lattice{Nx,Ny,Infinite,S}
const TransposedUnitCell{Nx,Ny,S} = TransposedLattice{Nx,Ny,Infinite,S}

struct SubLattice{Nx,Ny,L,S,P<:AbstractLattice} <: AbstractLattice{Nx,Ny,L,S}
    parent::P
    index::Int
end

const LatticeViewX{Nx} = SubLattice{Nx,1}
const LatticeViewY = SubLattice{1}

const UnitCellView{Nx,Ny} = SubLattice{Nx,Ny,Infinite}

function SubLattice(P::AbstractLattice{Nx,Ny,L,S}, ::Colon, i::Int) where {Nx,Ny,L,S}
    return SubLattice{Nx,1,L,S,typeof(P)}(P, i)
end
function SubLattice(P::AbstractLattice{Nx,Ny,L,S}, i::Int, ::Colon) where {Nx,Ny,L,S}
    return SubLattice{1,Ny,L,S,typeof(P)}(P, i)
end
lview(P::AbstractLattice, i1, i2) = SubLattice(P, i1, i2)
# to_matrix(L::SubLattice) = getindex(to_matrix(L.parent), latticeviewindices(L)...)

getmatrix(L::SubLattice) = getmatrix(L.parent)

# This function is for contructing the matrix view only
_to_indices(L::TransposedLattice, inds::Tuple) = to_indices(L.parent, (inds[2], inds[1]))
_to_indices(L::Lattice, inds::Tuple) = to_indices(L, inds)

function latticeviewindices(L::SubLattice{Nx,1}) where {Nx}
    return _to_indices(L.parent, (:, (L.index):(L.index)))
end
function latticeviewindices(L::SubLattice{1,Ny}) where {Ny}
    return _to_indices(L.parent, ((L.index):(L.index), :))
end
function latticeviewindices(
    L::SubLattice{1,1,LStyle,S,P}
) where {LStyle,S,Nx,P<:AbstractLattice{Nx,1}}
    return _to_indices(L.parent, (:, (L.index):(L.index)))
end
function latticeviewindices(
    L::SubLattice{1,1,LStyle,S,P}
) where {LStyle,S,Ny,P<:AbstractLattice{1,Ny}}
    return _to_indices(L.parent, ((L.index):(L.index), :))
end
function latticeviewindices(
    L::SubLattice{1,1,LStyle,S,P}
) where {LStyle,S,P<:AbstractLattice{1,1}}
    return _to_indices(L.parent, (1:1, 1:1))
end

# to_latticeindices(l::SubLattice, i::Vararg{Int,2}) = to_latticeindices(l.parent, i...)
function to_latticeindices(l::SubLattice{Nx,1}, i1::Int, i2::Int) where {Nx}
    @boundscheck checkbounds(l, i1, i2)
    return to_latticeindices(l.parent, i1, l.index)
end
function to_latticeindices(l::SubLattice{1,Ny}, i1::Int, i2::Int) where {Ny}
    @boundscheck checkbounds(l, i1, i2)
    return to_latticeindices(l.parent, l.index, i2)
end

function to_latticeindices_inbounds(l::AbstractLattice, i::Vararg{Int,2})
    return to_latticeindices(l, i...)
end
function to_latticeindices_inbounds(l::SubLattice{Nx,1}, i1::Int, i2::Int) where {Nx}
    @boundscheck checkbounds(l, i1, i2)
    return to_latticeindices(l.parent, i1, 1)
end
function to_latticeindices_inbounds(l::SubLattice{1,Ny}, i1::Int, i2::Int) where {Ny}
    @boundscheck checkbounds(l, i1, i2)
    return to_latticeindices(l.parent, 1, i2)
end
function to_latticeindices_inbounds(l::SubLattice{1,1}, i1::Int, i2::Int)
    @boundscheck checkbounds(l, i1, i2)
    return to_latticeindices(l.parent, 1, 1)
end

# Default to symmetric.
function (::Type{T})(sp::AbstractMatrix) where {T<:Lattice}
    return T(sp, true)
end
# function (::Type{T})(
#     localspace::S, sym::Bool
# ) where {Nx,Ny,T<:AbstractLattice{Nx,Ny},S<:IndexSpace}
#     return T(fill(localspace, Nx, Ny), sym)
# end
# function (::Type{T})(d::Int, sym::Bool) where {Nx,Ny,T<:AbstractLattice{Nx,Ny}}
#     return T(ℂ^d, sym)
# end
# function (::Type{T})(d::Matrix{Int}, sym::Bool) where {Nx,Ny,T<:AbstractLattice{Nx,Ny}}
#     return T(broadcast(x -> ℂ^x, d), sym)
# end

function spacematrix(L::Type{<:AbstractLattice{Nx,Ny}}, S::IndexSpace) where {Nx,Ny}
    return fill(S, Nx, Ny)
end
spacematrix(L::Type{<:AbstractLattice}, D::Int) = spacematrix(L, ℂ^D)
spacematrix(L::Type{<:AbstractLattice}, D::Matrix{Int}) = broadcast(x -> ℂ^x, D)
spacematrix(L::Type{<:AbstractLattice}, D::Matrix{<:IndexSpace}) = D
spacematrix(L::AbstractLattice, D) = spacematrix(typeof(L), D)

# boundarystyle(::Type{<:Finite{Nx,Ny,B}}) where {Nx,Ny,B} = B()
# boundarystyle(::Type{<:Infinite}) = Periodic()

@doc raw"""
    getbonds(L::AbstractLatticeLike) -> H,V

Returns a matrices of `Bond` objects `H` and `V`. The `Matrix{Bond}` object `H` contains all 
the horizontal bonds of the lattice `L`, with `V` containing the vertical bonds. For non-
square lattices, horizontal and vertical bond directions are defined by the mapping to the
square lattice.
"""
getbonds(L::AbstractLattice) = _getbonds(boundarystyle(L), L)

# function _getbonds(::Periodic, L::AbstractLattice)
#     Nx, Ny = size(L)
#     horizontal = Matrix{Bond}(undef, Nx, Ny)
#     vertical = Matrix{Bond}(undef, Ny, Nx)
#     rx = 1:Nx
#     ry = 1:Ny
#     for y in ry
#         for x in rx
#             horizontal[x, y] = Bond(CartesianIndex(x, y), CartesianIndex(mod(x + 1, rx), y))
#             # Why is this transposed?
#             vertical[y, x] = Bond(CartesianIndex(x, y), CartesianIndex(x, mod(y + 1, ry)))
#         end
#     end
#     return horizontal, vertical
# end

function _getbonds(::Periodic, L::AbstractLattice)
    Nx, Ny = size(L)
    rx = 1:Nx
    ry = 1:Ny

    incr = CartesianIndex((1, 0))

    func = x -> Bond(x, CartesianIndex(mod.(Tuple(x + incr), (rx, ry))))

    bonds = map(func, eachindex(L))

    return bonds
end
