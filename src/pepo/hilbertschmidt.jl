struct HilbertSchmidt{G,PType<:TensorPair,A<:Matrix{PType}} <:
       ITC.AbstractNetwork{G,PType,A}
    data::Lattice{G,PType,A}
end

ITC.getdata(hs::HilbertSchmidt) = hs.data

HilbertSchmidt(p::AbstractLattice{G,T}) where {G,T<:TensorMap} = HilbertSchmidt(p, p)

function HilbertSchmidt(
    p1::AbstractLattice{G,T}, p2::AbstractLattice{G,T}
) where {G,T<:TensorMap}
    return HilbertSchmidt(TensorPair.(p1, p2))
end

hilbertschmidt(t) = HilbertSchmidt(t)
hilbertschmidt(t1, t2) = HilbertSchmidt(t1, t2)

hilbertschmidt(p::AbstractPEPO) = hilbertschmidt(currentbuffer(p))
hilbertschmidt(p::GlobalUpdatablePEPO) = hilbertschmidt(currentbuffer(p))
function hilbertschmidt(p::Tuple{P,P}) where {P<:AbstractPEPO}
    return hilbertschmidt(p[1]), hilbertschmidt(p[2])
end

function Base.BroadcastStyle(::Type{<:HilbertSchmidt{G,PType,A}}) where {G,PType,A}
    return LatticeStyle{G,typeof(Base.BroadcastStyle(A))}()
end
