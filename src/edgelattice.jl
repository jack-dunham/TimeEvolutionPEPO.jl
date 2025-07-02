struct UnitCellBonds{G,ElType,A} <: AbstractUnitCell{G,ElType,A}
    data::UnitCell{G,NTuple{2,ElType},A}
end

Base.eltype(ucb::UnitCellBonds) = eltype(ucb.data)

function latticetype(
    ::Type{<:UnitCellBonds}, ::Type{UnitCell{G,NTuple{2,ElType},A}}
) where {G,ElType,A}
    return UnitCellBonds{G,ElType,A}
end

TRGroups.getdata(lat::UnitCellBonds) = lat.data

Base.similar(lat::L) where {L<:UnitCellBonds} = L(similar(TRGroups.getdata(lat)))
function Base.similar(lat::L, dims::Tuple) where {L<:UnitCellBonds}
    return L(similar(TRGroups.getdata(lat), dims))
end
function Base.similar(lat::UnitCellBonds{G}, ::Type{T}) where {G,T}
    return UnitCellBonds{G}(similar(TRGroups.getdata(lat), T))
end

function UnitCellBonds(X::AbstractUnitCell, Y::AbstractUnitCell)
    A = similar(X, NTuple{2,promote_type(eltype(X), eltype(Y))})

    A .= zip(X, Y)

    return UnitCellBonds(A)
end
function UnitCellBonds(X::AbstractMatrix, Y::AbstractMatrix)
    return UnitCellBonds(UnitCell(X), UnitCell(Y))
end

Base.getindex(mat::UnitCellBonds, bond::Bond) = mat.data[first(bond)][bondaxis(bond)]

function Base.getindex(mat::UnitCellBonds, ind::CartesianIndex{2})
    e = mat.data[ind][1]
    s = mat.data[ind][2]

    w = mat.data[ind - CartesianIndex(1, 0)][1]
    n = mat.data[ind - CartesianIndex(0, 1)][2]

    return (e, s, w, n)
end
function Base.setindex!(lat::UnitCellBonds, value, ind::CartesianIndex{2})
    lat.data[ind] = value
    return lat
end

function Base.setindex!(lat::UnitCellBonds, value, bond::Bond)
    ind = first(bond)

    newtup = TupleTools.setindex(lat.data[ind], value, bondaxis(bond))

    lat.data[ind] = newtup

    return lat
end

function Base.permutedims!(
    dst::UnitCellBonds{G,T}, src::UnitCellBonds{G,T}, perm=(2, 1)
) where {G,T}
    dst .= map(reverse, permutedims(src.data, perm))
    return dst
end

# Map the function f on all the bonds incident to the tensors indicident to `bond`
# returns U => V with the the bonds in U and V in canonical order.
function mapadjacent(f, edgelat::UnitCellBonds{Geom,T}, bond::Bond) where {Geom,T}
    uind, vind = bond

    e1, s1, w1, n1 = bondswith(Geom, uind)
    e2, s2, w2, n2 = bondswith(Geom, vind)

    axis = bondaxis(bond)

    gw = bond -> f(edgelat[bond])

    if axis == 1
        U = map(gw, (s1, w1, n1))
        V = map(gw, (e2, s2, n2))
    elseif axis == 2
        U = map(gw, (e1, w1, n1))
        V = map(gw, (e2, s2, w2))
    end

    return U => V
end

adjacent(edgelat, bond) = mapadjacent(identity, edgelat, bond)

struct UnitCellBondsStyle{G,A} <: TRGroups.AbstractUnitCellStyle{G,A} end

function Base.BroadcastStyle(::Type{UnitCellBonds{G,ElType,A}}) where {G,ElType,A}
    return UnitCellBondsStyle{G,typeof(Base.BroadcastStyle(A))}()
end

function Base.similar(
    bc::Broadcast.Broadcasted{UnitCellBondsStyle{G,A}}, ::Type{ElType}
) where {G,ElType,A}
    data = similar(Base.convert(Broadcast.Broadcasted{A}, bc), NTuple{2,ElType})

    return UnitCellBonds{G}(data)
end

Base.BroadcastStyle(::UnitCellBondsStyle, L::TRGroups.UnitCellStyle) = L

function Base.Broadcast.broadcasted(::UnitCellBondsStyle{G,A}, f, args...) where {G,A}
    data = map(TRGroups.getdata, args)

    rv = broadcast(data...) do ucs...
        h = f(broadcast(first, ucs)...)
        v = f(broadcast(last, ucs)...)

        return (h, v)
    end

    return UnitCellBonds(rv)
end
