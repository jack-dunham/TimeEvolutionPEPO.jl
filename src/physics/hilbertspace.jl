const BondSpace{L,S<:IndexSpace} = Lattice{NotTensor,L,S}

function BondSpace(L::LatticeStyle, data::AbstractMatrix{<:IndexSpace})
    return Lattice(NotTensor(), L, data)
end
