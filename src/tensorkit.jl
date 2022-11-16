### TensorKit.jl overloads

function leftorth(
    p::AbstractPhysicalTensor,
    leftind::Tuple,
    rightind::Tuple,
    alg::OrthogonalFactorizationAlgorithm=QRpos(),
)
    return leftorth(convert(TensorMap, p), leftind, rightind, alg)
end

function rightorth(
    p::AbstractPhysicalTensor,
    leftind::Tuple,
    rightind::Tuple,
    alg::OrthogonalFactorizationAlgorithm=LQpos(),
)
    return rightorth(convert(TensorMap, p), leftind, rightind, alg)
end
