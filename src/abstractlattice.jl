copyindex!(lat, val, bondind) = copy!(lat[bondind], val)

using InfiniteTensorContractions: SquareSymmetric
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

Base.iterate(bond::Bond, state=1) = iterate(bond.data, state)
Base.length(bond::Bond) = 2

Base.first(bond::Bond) = bond.data[1]
Base.last(bond::Bond) = bond.data[2]

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

Returns the axis that `bond` lies on in a uc.
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
    bonds(L::AbstractUnitCell) -> H,V

Returns a matrices of `Bond` objects `H` and `V`. The `Matrix{Bond}` object `H` contains all 
the horizontal bonds of the uc `L`, with `V` containing the vertical bonds. For non-
square ucs, horizontal and vertical bond directions are defined by the mapping to the
square uc.
"""
function bonds(uc::AbstractUnitCell)
    # hind = eachindex(uc)
    hind = uniqueindices(uc)
    vind = permutedims(hind)

    hbonds = map(hind) do ind
        return Bond(ind, ind + CartesianIndex(1, 0))
    end

    vbonds = map(vind) do ind
        return Bond(ind, ind + CartesianIndex(0, 1))
    end

    return hbonds, vbonds
end

function bonds(uc::AbstractUnitCell{SquareSymmetric})
    # hind = eachindex(uc)
    hind = uniqueindices(uc)

    hbonds = map(hind) do ind
        return Bond(ind, ind + CartesianIndex(1, 0))
    end

    return hbonds, transpose.(hbonds)
end

bonds(uc::AbstractUnitCell, axis) = bonds(uc)[axis]

eachbond(lat) = Iterators.flatten(bonds(lat))

bondswith(::AbstractUnitCell{G}, ind) where {G} = bondswith(G, ind)
function bondswith(::Type{<:Union{Square,SquareSymmetric}}, ind)
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

function Base.getindex(uc::AbstractUnitCell, bond::Bond)
    uind, vind = bond
    @inbounds rv = uc[uind] => uc[vind]
    return rv
end

function Base.setindex!(
    uc::AbstractUnitCell{G,T}, vals::Union{Tuple{T,T},Pair{T,T}}, bond::Bond
) where {G<:AbstractUnitCellGeometry,T}
    val1, val2 = vals
    uind, vind = bond

    uc[uind] = val1
    uc[vind] = val2

    return uc
end

function Base.copyto!(uc::AbstractUnitCell, val1, val2, bond::Bond)
    u, v = uc[bond]

    copy!(u, val1)
    copy!(v, val2)

    return uc
end

uniqueindices(L::AbstractUnitCell{Square}) = CartesianIndices(size(L))
uniqueindices(L::AbstractUnitCell{SquareSymmetric}) = CartesianIndices((size(L, 1), 1))
uniquebonds(L; kwargs...) = bonds(L; kwargs...)[uniqueindices(L)]

contractphysical(l::AbstractUnitCell) = contractphysical.(l)

function elcopy!(dst::AbstractUnitCell, src::AbstractUnitCell)
    foreach(copy!, dst, src)
    return dst
end

struct PermuteTranspose{G,ElType,A,L<:AbstractUnitCell{G,ElType,A}} <:
       AbstractUnitCell{G,ElType,A}
    parent::L
    perm::NTuple{4,Int}
end

PermuteTranspose(uc) = PermuteTranspose(uc, (2, 1, 4, 3))

function elcopy!(dst::L, src::PermuteTranspose{L}) where {L}
    foreach(dst, PermutedDimsArray(src.parent, (2, 1))) do out, in
        permutedom!(out, in, src.perm)
    end
    return dst
end
