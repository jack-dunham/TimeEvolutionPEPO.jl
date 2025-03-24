mutable struct AxialPEPO{G,PType,CType} <: AbstractPEPO{G,PType}
    const parent::PEPO{G,PType,CType}
    const vstate::UnitCell{G,PType,Matrix{PType}}
    currentaxis::Int
end

AxialPEPO(pepo) = AxialPEPO(pepo, permutetranspose(ITC.getdata(pepo)), 1)

currentaxis(axp::AxialPEPO) = axp.currentaxis

ITC.getdata(axp::AxialPEPO) = getaxis(axp, currentaxis(axp))
function getaxis(axp::AxialPEPO, axis)
    if axis == 1
        return ITC.getdata(axp.parent)
    elseif axis == 2
        return axp.vstate
    else
        throw(ArgumentError("argument `axis` must be either 1 or 2, got $axis"))
    end
end

weights(axp::AxialPEPO) = weights(axp.parent)

function setaxis!(pepo::AxialPEPO, axis::Int)
    oldaxis = currentaxis(pepo)

    if currentaxis(pepo) == axis
        return pepo
    end

    permutetranspose!(getaxis(pepo, axis), getaxis(pepo, oldaxis))

    pepo.currentaxis = axis

    return pepo
end

function setweight!(pepo::AxialPEPO, value, bond::Bond; kwargs...)
    bond = currentaxis(pepo) == 2 ? transpose(bond) : bond
    setweight!(pepo.parent, value, bond; kwargs...)
    return pepo
end

function Base.setindex!(
    pepo::AxialPEPO{G,T}, value::Union{Pair{T,T},Tuple{T,T}}, bond::Bond; kwargs...
) where {G,T}
    axis = currentaxis(pepo)

    # Transpose bond if on axis 2.
    bond = axis == 2 ? transpose(bond) : bond

    setindex!(getaxis(pepo, axis), value, bond)

    return pepo
end
