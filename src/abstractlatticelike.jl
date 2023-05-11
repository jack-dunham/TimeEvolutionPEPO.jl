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
    in::CartesianIndex{2}
    out::CartesianIndex{2}
    function Bond(in::CartesianIndex{2}, out::CartesianIndex{2})
        if out > in
            return new(in, out)
        else
            throw(
                ArgumentError(
                    "Outward site index $out must be greater than inward site index $in"
                ),
            )
        end
    end
end

@doc raw"""
    getaxis(bond::Bond)

Returns the axis that `bond` lies on in a lattice.
"""
function getaxis(bond::Bond)
    bond_diff = bond.out - bond.in
    if bond_diff[1] == 1
        return 1
    elseif bond_diff[2] == 1
        return 2
    end
end

function nextbond(bond::Bond)
    bond_diff = bond.out - bond.in
    return Bond(bond.out, bond.out + bond_diff)
end

function Base.show(io::IO, ::MIME"text/plain", bond::Bond)
    print("$(Tuple(bond.in)) ⟵⟶ $(Tuple(bond.out)) ")
    return nothing
end

getbonds(L::AbstractLatticeLike) = getbonds(lattice(L))

function getsite(lattice::AbstractLatticeLike, i...)
    mat = getmatrix(lattice)
    return mat[i...]
end

function Base.getindex(lattice::AbstractLatticeLike, bond::Bond)
    # Get the matrix of data such that we can ignore custom indexing
    # and index matrix directly.
    mat = getmatrix(lattice)
    @inbounds rv = mat[bond.in], mat[bond.out]
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
