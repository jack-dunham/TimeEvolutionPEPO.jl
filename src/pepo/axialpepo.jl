mutable struct AxialPEPO{G,PType,CType} <: AbstractPEPO
    const parent::PEPO{G,PType,CType}
    const vstate::Lattice{G,PType,Matrix{PType}}
    currentaxis::Int
end

AxialPEPO(pepo) = AxialPEPO(pepo, permutetranspose(tensors(pepo)), 1)

currentaxis(axp::AxialPEPO) = axp.currentaxis

function tensors(axp::AxialPEPO, axis=currentaxis(axp))
    if axis == 1
        return tensors(axp.parent)
    elseif axis == 2
        return axp.vstate
    end
end

weights(axp::AxialPEPO) = weights(axp.parent)

function setaxis!(pepo::AxialPEPO, axis::Int)
    oldaxis = currentaxis(pepo)

    if currentaxis(pepo) == axis
        return pepo
    end

    permutetranspose!(tensors(pepo, axis), tensors(pepo, oldaxis))

    pepo.currentaxis = axis

    return pepo
end
