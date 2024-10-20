struct EdgeLattice{G,ElType,A} <: AbstractLattice{G,ElType,A}
    data::UnitCell{G,NTuple{2,ElType},A}
end

function latticetype(
    ::Type{<:EdgeLattice}, ::Type{UnitCell{G,NTuple{2,ElType},A}}
) where {G,ElType,A}
    return EdgeLattice{G,ElType,A}
end

ITC.getdata(lat::EdgeLattice) = lat.data

Base.similar(lat::L) where {L<:EdgeLattice} = L(similar(ITC.getdata(lat)))
function Base.similar(lat::L, dims::Tuple) where {L<:EdgeLattice}
    return L(similar(ITC.getdata(lat), dims))
end
function Base.similar(lat::EdgeLattice{G}, ::Type{T}) where {G,T}
    return EdgeLattice{G}(similar(ITC.getdata(lat), T))
end

function EdgeLattice(X::AbstractMatrix, Y::AbstractMatrix)
    X = UnitCell(X)
    Y = UnitCell(Y)

    A = similar(X, NTuple{2,promote_type(eltype(X), eltype(Y))})

    A .= zip(X, Y)

    return EdgeLattice(A)
end

Base.getindex(mat::EdgeLattice, bond::Bond) = mat.data[first(bond)][bondaxis(bond)]

function Base.getindex(mat::EdgeLattice, ind::CartesianIndex{2})
    e = mat.data[ind][1]
    s = mat.data[ind][2]

    w = mat.data[ind - CartesianIndex(1, 0)][1]
    n = mat.data[ind - CartesianIndex(0, 1)][2]

    return (e, s, w, n)
end
function Base.setindex!(lat::EdgeLattice, value, ind::CartesianIndex{2})
    lat.data[ind] = value
    return lat
end

function Base.setindex!(lat::EdgeLattice, value, bond::Bond)
    ind = first(bond)

    newtup = TupleTools.setindex(lat.data[ind], value, bondaxis(bond))

    lat.data[ind] = newtup

    return lat
end

function Base.permutedims!(
    dst::EdgeLattice{G,T}, src::EdgeLattice{G,T}, perm=(2, 1)
) where {G,T}
    dst .= map(reverse, permutedims(src.data, perm))
    return dst
end

function mapadjacent(f, edgelat::EdgeLattice{Geom,T}, bond::Bond) where {Geom,T}
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

struct EdgeLatticeStyle{G,A} <: AbstractLatticeStyle{G,A} end

function Base.BroadcastStyle(::Type{EdgeLattice{G,ElType,A}}) where {G,ElType,A}
    return EdgeLatticeStyle{G,typeof(Base.BroadcastStyle(A))}()
end

function Base.similar(
    bc::Broadcast.Broadcasted{EdgeLatticeStyle{G,A}}, ::Type{ElType}
) where {G,ElType,A}
    data = similar(Base.convert(Broadcast.Broadcasted{A}, bc), NTuple{2,ElType})

    return EdgeLattice{G}(data)
end

Base.BroadcastStyle(::EdgeLatticeStyle, L::LatticeStyle) = L

function Base.Broadcast.broadcasted(::EdgeLatticeStyle{G,A}, f, args...) where {G,A}
    data = map(ITC.getdata, args)

    rv = broadcast(data...) do ucs...
        h = f(broadcast(first, ucs)...)
        v = f(broadcast(last, ucs)...)

        return (h, v)
    end

    return EdgeLattice(rv)
end
