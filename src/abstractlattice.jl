abstract type AbstractLattice{G,ElType,A} <: ITC.AbstractUnitCell{G,ElType,A} end

ischeckerboard(::AbstractLattice{G}) where {G} = ischeckerboard(G)
ischeckerboard(::Type{Square}) = false

## SIMILAR
function Base.similar(lat::L) where {L<:AbstractLattice}
    uc = similar(ITC.getdata(lat))
    T = latticetype(L, typeof(uc))
    return T(uc)
end
function Base.similar(lat::L, ::Type{S}) where {L<:AbstractLattice,S}
    uc = similar(ITC.getdata(lat), S)
    T = latticetype(L, typeof(uc))
    return T(uc)
end
function Base.similar(lat::L, dims::Dims{2}) where {L<:AbstractLattice}
    uc = similar(ITC.getdata(lat), dims)
    T = latticetype(L, typeof(uc))
    return T(uc)
end
function Base.similar(lat::L, ::Type{S}, dims::Dims{2}) where {L<:AbstractLattice,S}
    uc = similar(ITC.getdata(lat), S, dims)
    T = latticetype(L, typeof(uc))
    return T(uc)
end

struct Lattice{G,ElType,A} <: AbstractLattice{G,ElType,A}
    data::UnitCell{G,ElType,A}
end

ITC.getdata(lattice::Lattice) = lattice.data

copyindex!(lat, val, bondind) = copy!(lat[bondind], val)

const SquareLattice = Lattice{Square}

function latticetype(::Type{<:Lattice}, ::Type{UnitCell{G,ElType,A}}) where {G,ElType,A}
    return Lattice{G,ElType,A}
end

## CONSTRUCTORS

function (::Type{L})(::UndefInitializer, dims...) where {G,T,L<:AbstractLattice{G,T}}
    mat = Matrix{T}(undef, dims...)
    uc = UnitCell{G}(mat)

    LType = latticetype(L, typeof(uc))

    return LType(uc)
end

function (::Type{L})(data) where {L<:AbstractLattice}
    uc = UnitCell{Square}(data)

    LType = latticetype(L, typeof(uc))

    return LType(uc)
end
function (::Type{L})(data) where {G,L<:AbstractLattice{G}}
    uc = UnitCell{G}(data)

    LType = latticetype(L, typeof(uc))

    return LType(uc)
end

Lattice{G}(data) where {G} = Lattice(UnitCell{G}(data))

struct Bond
    data::Pair{CartesianIndex{2},CartesianIndex{2}}
    function Bond(data::Pair{CartesianIndex{2},CartesianIndex{2}})
        uind, vind = data
        if uind == vind
            throw(ArgumentError("indices in a bond must be distinct"))
        end
        return new(data)
    end
end

Bond(uind, vind) = Bond(CartesianIndex(uind) => CartesianIndex(vind))

Base.first(bond::Bond) = first(bond.data)
Base.last(bond::Bond) = last(bond.data)

Base.iterate(bond::Bond, state=1) = iterate(bond.data, state)
Base.length(bond::Bond) = 2

Base.broadcastable(bond::Bond) = Ref(bond)

astuples(bond::Bond) = mapreduce(Tuple, =>, bond)

function Base.transpose(bond::Bond)
    uind, vind = bond

    return Bond((uind[2], uind[1]), (vind[2], vind[1]))
end

increment(bond::Bond) = last(bond) - first(bond)

function Base.show(::IO, ::MIME"text/plain", bond::Bond)
    u, v = astuples(bond)
    axis = bondaxis(bond)
    if axis == 1
        s = "↔"
    elseif axis == 2
        s = "↕"
    end
    print("$u $s $v ")
    return nothing
end

"""
    bondaxis(bond::Bond)

Returns the axis that `bond` lies on in a lattice.
"""
function bondaxis(bond::Bond)
    incr = Tuple(increment(bond))

    if incr in ((1, 0), (-1, 0))
        return 1
    elseif incr in ((0, 1), (0, -1))
        return 2
    elseif incr == (0, 0)
        throw(ArgumentError("A bond on a single-site has an ambiguous direction"))
    else
        throw(ErrorException("Impossible `else` reached!, bond diff: $incr"))
    end
end

"""
    bonds(L::AbstractLattice) -> H,V

Returns a matrices of `Bond` objects `H` and `V`. The `Matrix{Bond}` object `H` contains all 
the horizontal bonds of the lattice `L`, with `V` containing the vertical bonds. For non-
square lattices, horizontal and vertical bond directions are defined by the mapping to the
square lattice.
"""
function bonds(lattice::AbstractLattice)
    hind = eachindex(lattice)
    vind = permutedims(hind)

    hbonds = map(hind) do ind
        return Bond(ind, ind + CartesianIndex(1, 0))
    end

    vbonds = map(vind) do ind
        return Bond(ind, ind + CartesianIndex(0, 1))
    end

    return hbonds, vbonds
end
bonds(lattice::AbstractLattice, axis) = bonds(lattice)[axis]

eachbond(lat) = Iterators.flatten(bonds(lat))

bondswith(::AbstractLattice{G}, ind) where {G} = bondswith(G, ind)
function bondswith(::Type{Square}, ind)
    x = CartesianIndex(1, 0)
    y = CartesianIndex(0, 1)

    e = Bond(ind, ind + x)
    s = Bond(ind, ind + y)
    w = Bond(ind - x, ind)
    n = Bond(ind - y, ind)

    return e, s, w, n
end

nextbond(bond::Bond) = begin
    _, vind = bond
    Bond(vind, vind + increment(bond))
end
prevbond(bond::Bond) = begin
    uind, _ = bond
    Bond(uind - increment(bond), uind)
end

function Base.getindex(lattice::AbstractLattice, bond::Bond)
    uind, vind = bond
    @inbounds rv = lattice[uind] => lattice[vind]
    return rv
end

function Base.setindex!(
    lattice::AbstractLattice{G,T}, vals::Union{Tuple{T,T},Pair{T,T}}, bond::Bond
) where {G,T}
    val1, val2 = vals
    uind, vind = bond

    lattice[uind] = val1
    lattice[vind] = val2

    return lattice
end

function Base.copyto!(lattice::AbstractLattice, val1, val2, bond::Bond)
    u, v = lattice[bond]

    copy!(u, val1)
    copy!(v, val2)

    return lattice
end

abstract type AbstractLatticeStyle{G,A} <: ITC.AbstractUnitCellStyle{G,A} end
struct LatticeStyle{G,A} <: AbstractLatticeStyle{G,A} end

function Base.BroadcastStyle(::Type{<:AbstractLattice{G,ElType,A}}) where {G,ElType,A}
    return LatticeStyle{G,typeof(Base.BroadcastStyle(A))}()
end

function Base.similar(
    bc::Broadcast.Broadcasted{LatticeStyle{G,A}}, ::Type{ElType}
) where {G,ElType,A}
    # lattice = find_lat(bc)
    # issym = false
    # if lattice === nothing
    #     issym = false
    # else
    #     issym = issymmetric(lattice)
    # end
    return Lattice{G}(similar(Base.convert(Broadcast.Broadcasted{A}, bc), ElType))
end

# find_lat(bc::Base.Broadcast.Broadcasted) = find_lat(bc.args)
# find_lat(args::Tuple) = find_lat(find_lat(args[1]), Base.tail(args))
# find_lat(x) = x
# find_lat(::Tuple{}) = nothing
# find_lat(a::AbstractLattice, rest) = a
# find_lat(::Any, rest) = find_lat(rest)

uniqueindices(L::SquareLattice) = CartesianIndices(size(L))
uniquebonds(L; kwargs...) = bonds(L, kwargs...)[uniqueindices(L)]

contractphysical(l::AbstractLattice) = contractphysical.(l)


function elcopy!(dst::AbstractLattice, src::AbstractLattice)
    foreach(copy!, dst, src)
    return dst
end

struct PermuteTranspose{G, ElType, A, L<:AbstractLattice{G, ElType, A}} <: AbstractLattice{G, ElType, A}
    parent::L
    perm::NTuple{4,Int}
end

PermuteTranspose(lattice) = PermuteTranspose(lattice, (2,1,4,3))

function elcopy!(dst::L, src::PermuteTranspose{L}) where {L}
    foreach(dst, PermutedDimsArray(src.parent, (2, 1))) do out, in
        permutedom!(out, in, src.perm)
    end
    return dst
end


