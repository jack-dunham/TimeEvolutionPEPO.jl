const TO = TensorKit.TensorOperations

TO.memsize(t::AbstractPhysicalTensor) = TO.memsize(totensor(t))

function TO.similarstructure_from_indices(
    T::Type, p1::IndexTuple, p2::IndexTuple, A::AbstractPhysicalTensor, CA::Symbol=:N
)
    return TO.similarstructure_from_indices(T, p1, p2, totensor(A), CA)
end

function TO.similarstructure_from_indices(
    T::Type,
    poA::IndexTuple,
    poB::IndexTuple,
    p1::IndexTuple,
    p2::IndexTuple,
    A::AbstractPhysicalTensor,
    B::AbstractPhysicalTensor,
    CA::Symbol=:N,
    CB::Symbol=:N,
)
    return TO.similarstructure_from_indices(
        T, poA, poB, p1, p2, totensor(A), totensor(B), CA, CB
    )
end

TO.scalar(t::AbstractPhysicalTensor) = scalar(totensor(t))

function TO.add!(
    α,
    tsrc::AbstractPhysicalTensor{S},
    CA::Symbol,
    β,
    tdst::AbstractPhysicalTensor{S,T},
    p1::IndexTuple,
    p2::IndexTuple,
) where {S,N₁,N₂,T<:TensorMap{S,N₁,N₂}}
    TO.add!(α, totensor(tsrc), CA, β, totensor(tdst), p1, p2)
    return tdst
end

function TO.trace!(
    α,
    tsrc::AbstractPhysicalTensor{S},
    CA::Symbol,
    β,
    tdst::AbstractPhysicalTensor{S,T},
    p1::IndexTuple,
    p2::IndexTuple,
    q1::IndexTuple,
    q2::IndexTuple,
) where {S,N₁,N₂,T<:AbstractTensorMap{S,N₁,N₂}}
    TO.trace!(α, totensor(tsrc), CA, β, totensor(tdst), p1, p2, q1, q2)
    return tdst
end

function TO.contract!(
    α,
    tA::AbstractPhysicalTensor{S},
    CA::Symbol,
    tB::AbstractPhysicalTensor{S},
    CB::Symbol,
    β,
    tC::AbstractPhysicalTensor{S,T},
    oindA::IndexTuple,
    cindA::IndexTuple,
    oindB::IndexTuple,
    cindB::IndexTuple,
    p1::IndexTuple,
    p2::IndexTuple,
    syms::Union{Nothing,NTuple{3,Symbol}}=nothing,
) where {S,N₁,N₂,T<:AbstractTensorMap{S,N₁,N₂}}
    TO.contract!(
        α,
        totensor(tA),
        CA,
        totensor(tB),
        CB,
        β,
        totensor(tC),
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
