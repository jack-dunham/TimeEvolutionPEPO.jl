struct IndexMap{N} 
    codomain::NTuple{N,Int64}
    domain::NTuple{N,Int64}
end

Base.getindex(p::IndexMap,i::Int) = getfield(p,i)
Base.getindex(p::IndexMap,i::Real) = getfield(p, convert(Int, i))
Base.transpose(p::IndexMap{N}) where {N} = IndexMap{N}(p.domain, p.codomain)

Base.firstindex(p::IndexMap) = 1
Base.lastindex(p::IndexMap) = 2

TensorKit.codomain(p::IndexMap) = p.codomain
TensorKit.domain(p::IndexMap) = p.domain

function Base.summary(p::IndexMap)
    print(codomain(p), " ← ", domain(p))
end

function Base.show(io::IO, p::IndexMap)
    if get(io, :compact, false)
        print(io, codomain(p), " ← ", domain(p))
        return
    end
    println(io, codomain(p), " ← ", domain(p))
end
