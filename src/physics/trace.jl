abstract type AbstractBulk{G,ElType,A} <: ITC.AbstractNetwork{G,ElType,A} end

ITC.getdata(t::AbstractBulk) = t.data

Base.similar(bulk::AbstractBulk, ::Type{T}) where {T} = similar(ITC.getdata(bulk), T)

struct HilbertSchmidt{G,PType,A<:Matrix{PType}} <: AbstractBulk{G,PType,A}
    data::Lattice{G,PType,A}
end

HilbertSchmidt(p::AbstractLattice{G,T}) where {G,T<:TensorMap} = HilbertSchmidt(p, p)

function HilbertSchmidt(
    p1::AbstractLattice{G,T}, p2::AbstractLattice{G,T}
) where {G,T<:TensorMap}
    return HilbertSchmidt(TensorPair.(p1, p2))
end

hilbertschmidt(p::AbstractPEPO) = HilbertSchmidt(TensorPair.(getstate(p), getstate(p)))
function hilbertschmidt(p::GlobalUpdatablePEPO)
    return HilbertSchmidt(TensorPair.(getevolved(p), getevolved(p)))
end
hilbertschmidt(p::TimeDependantPEPO) = hilbertschmidt(p.x), hilbertschmidt(p.y)

struct Trace{G,PType<:AbsTen{0,4},A<:Matrix{PType}} <: AbstractBulk{G,PType,A}
    data::Lattice{G,PType,A}
end

trace(p::AbstractPEPO) = Trace(contractphysical.(getstate(p)))
trace(p::GlobalUpdatablePEPO) = Trace(contractphysical.(getcurrentstate(p)))

function contractphysical(tsrc::AbsTen{2,4})
    virtual_space = domain(tsrc)
    tdst = similar(tsrc, one(virtual_space), virtual_space)
    return contractphysical!(tdst, tsrc)
end
function contractphysical(tsrc_1::AbsTen{2,4}, tsrc_2::AbsTen{4,2})
    temp_space = *(map((x, y) -> x * y', domain(tsrc_1), codomain(tsrc_2))...)
    out = similar(tsrc_1, one(temp_space), temp_space)
    @tensoropt out[D1 D5 D2 D6 D3 D7 D4 D8] =
        tsrc_1[k b; D1 D2 D3 D4] * tsrc_2[D5 D6 D7 D8; k b]
    return out
end
function contractphysical!(tdst::AbsTen{0,4}, tsrc::AbsTen{2,4})
    @tensoropt tdst[1 2 3 4] = tsrc[x x; 1 2 3 4]
    return tdst
end

function contractphysical_mergevirtual(tsrc::AbsTen{2,4})
    t1 = traceobj.data
    t2 = t1'
    return contractphysical_mergevirtual(t1, t2)
end

function contractphysical_mergevirtual(tsrc_1::AbsTen{2,4}, tsrc_2::AbsTen{4,2})
    contracted_tensor = contractphysical(tsrc_1, tsrc_2)
    return mergevirtual(contracted_tensor)
end

function mergevirtual(t::AbstractTensorMap{S,0,8}) where {S}
    dom = (fuse(domain(t)[i] * domain(t)[i + 1]) for i in 1:2:7)
    return TensorMap(t.data, one(S), *(dom...))
end

function Base.BroadcastStyle(::Type{<:AbstractBulk{G,PType,A}}) where {G,PType,A}
    return LatticeStyle{G,typeof(Base.BroadcastStyle(A))}()
end
