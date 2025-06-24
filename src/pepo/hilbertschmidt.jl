struct HilbertSchmidt{G,PType<:CompositeTensor{2},A<:Matrix{PType}} <:
       TRGroups.AbstractNetwork{G,PType,A}
    data::UnitCell{G,PType,A}
end

TRGroups.getdata(hs::HilbertSchmidt) = hs.data

HilbertSchmidt(p::AbstractUnitCell{G,T}) where {G,T<:TensorMap} = HilbertSchmidt(p, p)

function HilbertSchmidt(
    lat1::AbstractUnitCell{G,T}, lat2::AbstractUnitCell{G,T}
) where {G,T<:TensorMap}
    rv = broadcast(lat1, lat2) do p1, p2
        return CompositeTensor((p1, p2'))
    end
    return HilbertSchmidt(rv)
end

hilbertschmidt(t) = HilbertSchmidt(t)
hilbertschmidt(t1, t2) = HilbertSchmidt(t1, t2)

hilbertschmidt(p::AbstractPEPO) = hilbertschmidt(TRGroups.getdata(p))
# hilbertschmidt(p::GlobalUpdatablePEPO) = hilbertschmidt(currentbuffer(p))
function hilbertschmidt(p::Tuple{P,P}) where {P<:AbstractPEPO}
    return hilbertschmidt(p[1]), hilbertschmidt(p[2])
end

function Base.BroadcastStyle(::Type{<:HilbertSchmidt{G,PType,A}}) where {G,PType,A}
    return TRGroups.UnitCellStyle{G,typeof(Base.BroadcastStyle(A))}()
end
