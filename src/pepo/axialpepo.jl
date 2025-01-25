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

currentbuffer(axp::AxialPEPO) = tensors(axp)

weights(axp::AxialPEPO) = weights(axp.parent)

function setaxis!(pepo::AxialPEPO, axis::Int)
    oldaxis = currentaxis(pepo)

    if currentaxis(pepo) == axis
        @warn "Nothing to do..."
        return pepo
    end

    permutetranspose!(tensors(pepo, axis), tensors(pepo, oldaxis))

    pepo.currentaxis = axis

    return pepo
end

function setweight!(pepo::AxialPEPO, value, bond::Bond; kwargs...)
    bond = currentaxis(pepo) == 2 ? transpose(bond) : bond
    setweight!(pepo.parent, value, bond; kwargs...)
    return pepo
end

function Base.setindex!(pepo::AxialPEPO, value, bond::Bond; kwargs...)
    
    bond = currentaxis(pepo) == 2 ? transpose(bond) : bond

    setindex!(currentbuffer(pepo), value, bond)

    return pepo
end
