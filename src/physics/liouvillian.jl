struct LiouvillianData{L,ElType,N}
    data::OnLattice{L,Array{ElType,N},Matrix{Array{ElType,N}}}
end

struct TrotterGates{L,MType}
    left::OnLattice{L,MType,Matrix{MType}}
    right::OnLattice{L,MType,Matrix{MType}}
end

function Base.getindex(gates::TrotterGates, bond::Bond)
    return (gates.left)[left(bond)], (gates.right)[right(bond)]
end

function trottergates(lattice, data, dt; trunc=truncerr(1e-8))
    data = exp.(dt * data)

    ket = lattice

    ketp = circshift(ket.space, (-1, 0))

    space = @. ket * ketp * adjoint(ket) * adjoint(ketp)

    tensors = TensorMap.(data, space, space)
    ptensors = permute.(tensors, Ref((1, 3, 5, 7)), Ref((2, 4, 6, 8)))

    out = tsvd.(ptensors; trunc=trunc)

    u = getindex.(out, 1)
    s = getindex.(out, 2)
    v = getindex.(out, 3)

    err = getindex.(out, 4)

    @info "Trotter compression error: $err with bond dim = $(dim.(codomain.(s)))"

    left = @. u * sqrt(s)
    right = circshift(sqrt.(s) .* v, (1, 0))

    return TrotterGates(left, _transpose.(right))
end

function thermalising()
    Z = Matrix{ComplexF64}([1 0; 0 -1])
    ZZ = kron(Z, Z)
    return kron(ZZ, one(ZZ)) + kron(one(ZZ), ZZ)
end
