const AbstractProjectedEntangledPairOperator{L,P<:AbsTen{3,3}} = OnLattice{PEPO,L,P}
const AbstractPEPO = AbstractProjectedEntangledPairOperator

const ProjectedEntangledPairOperator{L,P<:AbsTen{3,3}} = OnLattice{PEPO,L,P}

const InfinitePEPO{Nx,Ny,P} = ProjectedEntangledPairOperator{Infinite{Nx,Ny},P}

function Base.permutedims!(p::AbsTen{3,3}, perm::NTuple{6,Int})
    p.tensor = _permute(p.tensor, perm)
    return p
end

function bloch(a::Vector)
    σ = pauli
    mat = σ[4]
    for j in 1:3
        mat = mat + a[j] * σ[j]
    end
    return 1 / 2 * mat
end

function blochpepo_data(a::Vector, Dw::Int, Ds::Int, De::Int, Dn::Int)
    m = bloch(a)

    tw = zeros(ComplexF64, 2, Dw, 2)
    ts = zeros(ComplexF64, 2, Ds, 2)
    tw[:, 1, :] = ts[:, 1, :] = m

    te = zeros(ComplexF64, 2, 2, De)
    tn = zeros(ComplexF64, 2, 2, Dn)
    te[:, :, 1] = tn[:, :, 1] = m

    @tensor p[2 3 1; 5 6 4] := tw[s1 2; 4] * ts[s2 3; s1] * te[s3; s2 5] * tn[1; s3 6]

    return p
end

function blochpepo(localspace, a, westbond, southbond=westbond)
    L = latticestyle(localspace)
    bonds_x = OnLattice(NotTensor(), L, spacematrix(L, westbond))
    bonds_y = OnLattice(NotTensor(), L, spacematrix(L, southbond))

    bond_w = bonds_x
    bond_e = circshift(bonds_x, (0, -1))
    bond_s = bonds_y
    bond_n = circshift(bonds_y, (-1, 0))

    p_data = blochpepo_data.(Ref(a), dim.(bond_w), dim.(bond_s), dim.(bond_e), dim.(bond_n))
    p_tensors = @. TensorMap(
        p_data, bond_w * bond_s * localspace, bond_e * bond_n * localspace
    )
    return OnLattice(PEPO(), L, p_tensors)
end
