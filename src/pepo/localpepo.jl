mutable struct LocalUpdatablePEPO{G,PType,CType} <: AbstractUpdatablePEPO
    const parent::PEPO{G,PType,CType}
    const evolved_tensors::Lattice{G,Tuple{PType,PType},Matrix{Tuple{PType,PType}}}
    bond::Bond
    isevolved::Bool
end

currentbond(p::LocalUpdatablePEPO) = p.bond.x::Bond
function setcurrentbond!(p::LocalUpdatablePEPO, bond::Bond)
    p.bond = bond
    return p
end

function evolvedpair(p::LocalUpdatablePEPO{G,PType}, bond=currentbond(p)) where {G,PType}
    leftpair, rightpair = evolvedtensors(p)[bond]
    return leftpair[1]::PType, rightpair[2]::PType
end

function setevolvedpair!(p, val1, val2, bond)
    evolvedbuffer = evolvedtensors(p)

    leftpair, rightpair = evolvedbuffer[bond]
    newleftpair = (val1, leftpair[2])
    newrightpair = (rightpair[1], val2)

    evolvedbuffer[bond] = newleftpair, newrightpair
    return p
end

function parentpair(p::LocalUpdatablePEPO{G,PType}, bond=currentbond(p)) where {G,PType}
    leftpair, rightpair = latticetensors(p)[bond]
    return leftpair::PType, rightpair::PType
end

function _getindex(pe::LocalUpdatablePEPO, bond::Bond, ::Val{true})
    updated_bond = currentbond(pe)

    new_left, new_right = evolvedpair(pe)

    rv_left, rv_right = latticetensors(pe)[bond]

    if left(bond) == left(updated_bond)
        rv_left, rv_right = new_left, new_right
    elseif left(bond) == right(updated_bond)
        rv_left = new_right
    elseif right(bond) == left(updated_bond)
        rv_right = new_left
    end

    return rv_left, rv_right
end
