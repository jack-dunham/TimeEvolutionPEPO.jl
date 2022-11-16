const TO = TensorKit.TensorOperations

TO.memsize(t::AbstractPhysicalTensor) = TO.memsize(convert(AbstractTensorMap,t))

function TO.similarstructure_from_indices(
    T::Type, p1::IndexTuple, p2::IndexTuple, A::AbstractPhysicalTensor, CA::Symbol=:N
)
    return TO.similarstructure_from_indices(T, p1, p2, convert(AbstractTensorMap,A), CA)
end

function TO.similarstructure_from_indices(
    T::Type,
    poA::IndexTuple,
    poB::IndexTuple,
    p1::IndexTuple,
    p2::IndexTuple,
    A,
    B,
    CA::Symbol=:N,
    CB::Symbol=:N,
)
    return TO.similarstructure_from_indices(
        T, poA, poB, p1, p2, convert(AbstractTensorMap,A), convert(AbstractTensorMap,B), CA, CB
    )
end

TO.scalar(t::AbstractPhysicalTensor) = scalar(convert(AbstractTensorMap,t))

function TO.add!(
    α,
    tsrc,
    CA::Symbol,
    β,
    tdst,
    p1::IndexTuple,
    p2::IndexTuple,
) 
    TO.add!(α, convert(AbstractTensorMap,tsrc), CA, β, convert(AbstractTensorMap,tdst), p1, p2)
    return tdst
end

function TO.trace!(
    α,
    tsrc,
    CA::Symbol,
    β,
    tdst,
    p1::IndexTuple,
    p2::IndexTuple,
    q1::IndexTuple,
    q2::IndexTuple,
) 
    TO.trace!(α, convert(AbstractTensorMap,tsrc), CA, β, convert(AbstractTensorMap,tdst), p1, p2, q1, q2)
    return tdst
end

function TO.contract!(
    α,
    tA,
    CA::Symbol,
    tB,
    CB::Symbol,
    β,
    tC,
    oindA::IndexTuple,
    cindA::IndexTuple,
    oindB::IndexTuple,
    cindB::IndexTuple,
    p1::IndexTuple,
    p2::IndexTuple,
    syms::Union{Nothing,NTuple{3,Symbol}}=nothing,
) 
    TO.contract!(
        α,
        convert(AbstractTensorMap,tA),
        CA,
        convert(AbstractTensorMap,tB),
        CB,
        β,
        convert(AbstractTensorMap,tC),
        oindA,
        cindA,
        oindB,
        cindB,
        p1,
        p2,
        syms,
    )
    return tC
end
