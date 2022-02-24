function TensorKit.TO.contract!(
    α,
    tA::AbstractPhysicalTensor,
    CA::Symbol,
    tB::AbstractPhysicalTensor,
    CB::Symbol,
    β,
    tC::AbstractPhysicalTensor,
    oindA::TensorKit.IndexTuple,
    cindA::TensorKit.IndexTuple,
    oindB::TensorKit.IndexTuple,
    cindB::TensorKit.IndexTuple,
    p1::TensorKit.IndexTuple,
    p2::TensorKit.IndexTuple,
    syms::Union{Nothing,NTuple{3,Symbol}}=nothing,
)
    return TensorKit.TO.contract!(
        α, tA.tensor, CA, tB.tensor, CB, β, tC, oindA, cindA, oindB, cindB, p1, p2, syms
    )
end

TensorKit.TO.memsize(t::AbstractPhysicalTensor) = TensorKit.TO.memsize(t.tensor)

function TensorKit.storagetype(
    ::Type{<:AbstractPhysicalTensor{<:IndexSpace,N₁,N₂,Trivial,A}}
) where {N₁,N₂,A<:DenseMatrix}
    return A
end
function TensorKit.storagetype(
    ::Type{<:AbstractPhysicalTensor{<:IndexSpace,N₁,N₂,I,<:TensorKit.SectorDict{I,A}}}
) where {N₁,N₂,I<:Sector,A<:DenseMatrix}
    return A
end

@forward(
    AbstractPhysicalTensor.tensor,
    (
        Base.getindex,
        TensorKit.domain,
        TensorKit.codomain,
        TensorKit.fusiontrees,
        TensorKit.blocks,
        TensorKit.block,
        TensorKit.blocksectors,
    )
)
