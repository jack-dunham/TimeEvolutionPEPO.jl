function TensorKit.TO.contract!(
    α,
    tA::PEPOTensor{S},
    CA::Symbol,
    tB::PEPOTensor{S},
    CB::Symbol,
    β,
    tC::PEPOTensor{S},
    oindA::TensorKit.IndexTuple,
    cindA::TensorKit.IndexTuple,
    oindB::TensorKit.IndexTuple,
    cindB::TensorKit.IndexTuple,
    p1::TensorKit.IndexTuple,
    p2::TensorKit.IndexTuple,
    syms::Union{Nothing,NTuple{3,Symbol}}=nothing,
) where {S}
    return TensorKit.TO.contract!(
        α, tA.tensor, CA, tB.tensor, CB, β, tC, oindA, cindA, oindB, cindB, p1, p2, syms
    )
end

TensorKit.TO.memsize(t::PEPOTensor) = TensorKit.TO.memsize(t.tensor)

function TensorKit.storagetype(
    pt::Type{<:PEPOTensor{<:IndexSpace,<:Sector,A}}
) where {A<:DenseMatrix}
    return A
end

TensorKit.domain(p::PEPOTensor) = TensorKit.domain(p.tensor)
TensorKit.codomain(p::PEPOTensor) = TensorKit.codomain(p.tensor)

Base.getindex(p::PEPOTensor) = Base.getindex(p.tensor) 
Base.getindex(p::PEPOTensor, f::FusionTree) = Base.getindex(p.tensor, f) 
Base.getindex(p::PEPOTensor, f1::FusionTree, f2::FusionTree) = Base.getindex(p.tensor, f1, f2) 

TensorKit.fusiontrees(p::PEPOTensor) = TensorKit.fusiontrees(p.tensor)
