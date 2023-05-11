struct Weights{L,CType<:AbsTen{1,1},CCont}
    horizontal::OnLattice{L,CType,CCont}
    vertical::OnLattice{L,CType,CCont}
end

# Is transposed is a quasi-lazy way of swapping primary axis that does not 
# construct a new PEPO object, or change dims of data. Instead we just transpose the virtual 
# pepo bonds and set `istransposed` equal to `true`. Note the distinction to actually transpose
# the lattice.
struct PEPO{L,PType<:AbsTen{3,3},CType,PCont,CCont}
    state::OnLattice{L,PType,PCont}
    weights::Weights{L,CType,CCont}
    istransposed::Ref{Bool}
end

function transpose!(pepo::PEPO)
    tens = gettensors(pepo)
    permutedims!(tens, permute.(tens, Ref((2, 1, 3)), Ref((5, 4, 6))), (2, 1))
    pepo.istransposed[] = !(istransposed(pepo))
    return pepo
end

istransposed(pepo::PEPO) = pepo.istransposed[]

_flipaxis(istransposed::Bool, axis::Int) = mod(istransposed + axis, 1:2)
_flipaxis(istransposed::Bool, axis::Symbol) = axis

getstate(P::PEPO) = P.state

getweights(P::PEPO) = P.weights
getweights(pepo::PEPO, axis) = getfield(getweights(P), _flipaxis(istransposed(pepo), axis))

function setweight!(pepo, s, bond) end

getbonds(pepo::PEPO) = getbonds(getstate(P))

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

function pepodata(a::Vector, Dw::Int, Ds::Int, De::Int, Dn::Int)
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

function PEPO(a::AbstractVector, lattice, eastbond, southbond=eastbond)
    bond_w = circshift(eastbond, (1, 0))
    bond_e = eastbond
    bond_s = southbond
    bond_n = circshift(southbond, (0, 1))

    p_data = @. pepodata(Ref(a), dim(bond_w), dim(bond_s), dim(bond_e), dim(bond_n))
    tensors = @. TensorMap(
        p_data, bond_w * bond_s * localspace, bond_e * bond_n * localspace
    )

    return PEPO(tensors)
end

function PEPO(tensors::AbstractOnLattice{<:AbstractLattice,<:AbsTen{3,3}})
    horizontal_weights = @. id(getindex(domain(tensors), 1))
    vertical_weights = @. id(getindex(codomain(tensors), 2))

    weights = Weights(horizontal_weights, vertical_weights)

    return PEPO(tensors, weights, IsAbsorbed(true, true))
end
