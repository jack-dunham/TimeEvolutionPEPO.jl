mutable struct SparseTensorMap{N1,N2,A<:SparseArray} <:
               AbstractTensorMap{ComplexSpace,N1,N2}
    data::A
    codom::ProductSpace{ComplexSpace,N1}
    dom::ProductSpace{ComplexSpace,N2}
    function SparseTensorMap(
        data::A, codom::ProductSpace{S,N1}, dom::ProductSpace{S,N2}
    ) where {S,N1,N2,A}
        (d1, d2) = (dim(codom), dim(dom))

        if !(
            length(data) == d1 * d2 ||
            size(data) == (d1, d2) ||
            size(data) == (dims(codom)..., dims(dom)...)
        )
            throw(DimensionMismatch())
        end

        return new{N1,N2,A}(data, codom, dom)
    end
end

VectorInterface.scalartype(st::SparseTensorMap) = scalartype(st.data)

function SparseTensorMap(data::SparseArray, V::TensorMapSpace)
    return SparseTensorMap(data, codomain(V), domain(V))
end
function SparseTensorMap(data, cod::TensorMapSpace, dom::TensorMapSpace)
    return SparseTensorMap(data, ProductSpace(cod), ProductSpace(dom))
end
function SparseTensorMap(
    ::UndefInitializer, T::Type{<:Number}, cod::TensorSpace, dom::TensorSpace
)
    shape = TupleTools.vcat(dims(cod), dims(dom))
    data = SparseArray{T}(undef, shape)

    return SparseTensorMap(data, cod, dom)
end

function Base.similar(st::SparseTensorMap, cod::TensorSpace, dom::TensorSpace)
    cod = ProductSpace(cod)
    dom = ProductSpace(dom)

    newdims = TupleTools.vcat(dims(cod), dims(dom))
    rv = similar(st.data, newdims)
    return SparseTensorMap(rv, cod, dom)
end

function Base.similar(st::SparseTensorMap, T::Type{<:Number}, V::TensorMapSpace)
    cod = codomain(V)
    dom = domain(V)

    newdims = TupleTools.vcat(dims(cod), dims(dom))
    similar_data = similar(st.data, newdims)

    return SparseTensorMap(similar_data, cod, dom)
end

function TensorKit.permute(st::SparseTensorMap, perm::TensorKit.Index2Tuple)
    p1, p2 = perm

    stp = permutedims(st.data, TupleTools.vcat(p1, p2))

    newspace = permute(space(st), perm)

    return SparseTensorMap(stp, codomain(newspace), domain(newspace))
end

function TensorKit.TO.tensoradd!(
    C::SparseTensorMap,
    pC::TensorKit.TO.Index2Tuple,
    A::SparseTensorMap,
    conjA::Symbol,
    alpha::Number,
    beta::Number,
)
    TensorKit.TO.tensoradd!(C.data, pC, A.data, conjA, alpha, beta)

    return C
end

function TensorKit.TO.tensortrace!(
    C::SparseTensorMap,
    pC::TensorKit.TO.Index2Tuple,
    A::SparseTensorMap,
    pA::TensorKit.TO.Index2Tuple,
    conjA::Symbol,
    alpha::Number,
    beta::Number,
)
    TensorKit.TO.tensortrace!(C.data, pC, A.data, pA, conjA, alpha, beta)

    return C
end

function TensorKit.TO.tensorcontract!(
    C::SparseTensorMap,
    pC::TensorKit.TO.Index2Tuple,
    A::SparseTensorMap,
    pA::TensorKit.TO.Index2Tuple,
    conjA::Symbol,
    B::SparseTensorMap,
    pB::TensorKit.TO.Index2Tuple,
    conjB::Symbol,
    alpha::Number,
    beta::Number,
)
    TensorKit.TO.tensorcontract!(
        C.data, pC, A.data, pA, conjA, B.data, pB, conjB, alpha, beta
    )

    return C
end

function TensorKit.TO.tensoralloc_contract(
    TC,
    pC::TensorKit.Index2Tuple{N1,N2},
    A::SparseTensorMap,
    pA::TensorKit.Index2Tuple,
    ::Symbol,
    B::SparseTensorMap,
    pB::TensorKit.Index2Tuple,
    ::Symbol,
    ::Bool,
) where {N1,N2}
    spacetype(A) == spacetype(B) || throw(SpaceMismatch("incompatible space types"))

    spA = space(A)
    spB = space(B)

    cA = tuple(codomain(spA)...)
    dA = tuple(domain(spA)...)

    cB = tuple(codomain(spB)...)
    dB = tuple(domain(spB)...)

    indA = pA[1]
    indB = pB[2]

    spA = TupleTools.getindices(TupleTools.vcat(cA, dA), indA)
    spB = TupleTools.getindices(TupleTools.vcat(cB, dB), indB)

    spC = TupleTools.vcat(spA, spB)

    cod = ProductSpace(TupleTools.getindices(spC, pC[1]))
    dom = ProductSpace(TupleTools.getindices(spC, pC[2]))

    data = SparseArray{TC,N1 + N2}(undef, TupleTools.vcat(dims(cod), dims(dom)))

    return SparseTensorMap(data, cod, dom)
end

TensorKit.storagetype(::Type{<:SparseTensorMap{<:Any,<:Any,A}}) where {A} = A
function TensorKit.tensormaptype(
    ::Type{ComplexSpace}, N::Int, M::Int, ::Type{A}
) where {A<:SparseArray}
    return SparseTensorMap{N,M,A}
end

TensorKit.codomain(st::SparseTensorMap) = st.codom
TensorKit.domain(st::SparseTensorMap) = st.dom

function TensorKit.blocksectors(st::SparseTensorMap)
    return TensorKit.OneOrNoneIterator(dim(st) != 0, Trivial())
end

TensorKit.block(st::SparseTensorMap, ::Trivial) = st.data

sptensor(t::SparseTensorMap) = t

function sptensor(t::Union{TensorMap,AdjointTensorMap})
    data = SparseArray(convert(Array, t))
    return SparseTensorMap(data, codomain(t), domain(t))
end

function TensorKit.blocks(st::SparseTensorMap)
    return TensorKit.SingletonDict(Trivial() => block(st, Trivial()))
end

function droptol!(t; tol=eps())
    for (key, val) = nonzero_pairs(t.data)
        if abs(val) < tol
            t.data[key] = zero(scalartype(t))
        end
    end
    return t
end

function TensorMap(sp::SparseTensorMap)
    data = convert(Array, sp.data)
    return TensorMap(data, codomain(sp), domain(sp))
end
