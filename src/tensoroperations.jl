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
