@doc raw"""
    abstract type AbstractLatticeLike{ElType} <: AbstractMatrix{ElType}

Abstract supertype of objects that behave like lattices.
"""
abstract type AbstractLatticeLike{ElType} <: AbstractMatrix{ElType} end

@doc raw"""
    abstract type BoundaryStyle

Abstract supertype of all boundary types.
"""
abstract type BoundaryStyle end

struct Periodic <: BoundaryStyle end
struct Open <: BoundaryStyle end

abstract type LatticeStyle{B<:BoundaryStyle} end

struct Finite{B} <: LatticeStyle{B} end
struct Infinite <: LatticeStyle{Periodic} end

isperiodic(::Periodic) = true
isperiodic(::Open) = false
isperiodic(a) = BoundaryStyle(a)

boundarystyle(L::Type{<:LatticeStyle{B}}) where {B} = B()
boundarystyle(L::LatticeStyle{B}) where {B} = B()

# Fallback for matrices that are not subtypes of `AbstractLatticeLike`
getmatrix(m::AbstractMatrix) = m

# Top level function for indexing into lattices. Subtypes of `AbstractLatticeLike` should
# define methods for `getmatrix` and `to_latticeindices`.
function Base.getindex(L::AbstractLatticeLike, i::Vararg{Int,2})
    return getindex(getmatrix(L), to_latticeindices(L, i...)...)
end

Base.checkbounds(L::AbstractLatticeLike, i...) = _checkbounds(boundarystyle(L), L, i...)
_checkbounds(::Periodic, L::AbstractLatticeLike, i...) = nothing
function _checkbounds(::Open, L::AbstractLatticeLike, i...)
    sized_mat = similar(getmatrix(L), axes(L))
    return checkbounds(sized_mat, i...)
end

# struct SiteIndex <: Base.AbstractCartesianIndex{2}
#     ind::CartesianIndex{2}
# end
#
# convert(::Type{CartesianIndex}, siteind::SiteIndex) = siteind.ind
# convert(::Type{SiteIndex}, cartind::CartesianIndex{2}) = SiteIndex(cartind)
@doc raw"""
    struct Bond
"""
struct Bond
    left::CartesianIndex{2}
    right::CartesianIndex{2}
    # function Bond(left::CartesianIndex{2}, right::CartesianIndex{2})
    #     if right > left
    #         return new(left, right)
    #     else
    #         throw(
    #             ArgumentError(
    #                 "Right/up site index $right must be greater than left/down site index $left",
    #             ),
    #         )
    #     end
    # end
end

left(b::Bond) = b.left
right(b::Bond) = b.right

@doc raw"""
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

function nextbond(bond::Bond)
    bond_diff = bond.right - bond.left
    return Bond(bond.right, bond.right + bond_diff)
end

function Base.show(io::IO, ::MIME"text/plain", bond::Bond)
    print("$(Tuple(bond.left)) ⟵⟶ $(Tuple(bond.right)) ")
    return nothing
end

getbonds(L::AbstractLatticeLike) = getbonds(lattice(L))

function getsite(lattice::AbstractLatticeLike, i...)
    mat = getmatrix(lattice)
    return mat[i...]
end

function Base.getindex(lattice::AbstractLatticeLike, bond::Bond)
    @inbounds rv = lattice[bond.left], lattice[bond.right]
    return rv
end
@doc raw"""
    Base.transpose(L::AbstractLatticeLike)

Lazy transposing of a lattice-like objects. Does not permute data, but rather permutes axes when
indexing. To transpose data, use `permutedims`. Useful when algorithms are symmetric. 
"""
function LinearAlgebra.transpose(L::AbstractLatticeLike) end

function dimtospace(L::AbstractLatticeLike, D::Int)
    SType = spacetype(L)
    S = oneunit(SType)
    if D == 1
        return S::SType
    else
        return ⊕(S^D...)::SType
    end
end
