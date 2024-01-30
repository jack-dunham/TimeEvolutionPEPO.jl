struct Kagome <: ITC.AbstractUnitCellGeometry end

function kagome(D)
    sp = ℂ^2

    delta_ten = TensorMap(delta(ComplexF64, 1, 2, 2), ℂ^1, sp * sp')

    return nothing
end


