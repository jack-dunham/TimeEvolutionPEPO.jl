# Linear algebra functions

const σ0 = TensorMap(Matrix{ComplexF64}([1 0; 0 1]), ℂ^2, ℂ^2) # Identity matrix
const σ1 = TensorMap(Matrix{ComplexF64}([0 1; 1 0]), ℂ^2, ℂ^2) # Pauli X matrix
const σ2 = TensorMap(Matrix{ComplexF64}([0 -im; im 0]), ℂ^2, ℂ^2) # Pauli Y matrix
const σ3 = TensorMap(Matrix{ComplexF64}([1 0; 0 -1]), ℂ^2, ℂ^2) # Pauli Z matrix

const σp = 1 / 2 * (σ1 + im * σ2) # Pauli+ matrix
const σm = 1 / 2 * (σ1 - im * σ2) # Pauli- matrix

function LinearAlgebra.tr(t::PEPOTensor{S}) where {S<:IndexSpace}
    t = t.tensor
    @tensor tTr[1 2; 3 4] := t[1 2 x; 3 4 x]
end

function LinearAlgebra.ishermitian(p::PEPOTensor; phys_only=false)
    t = p.tensor
    phys_only || return LinearAlgebra.ishermitian(t)
    tT = permute(t, (1, 2, 6), (4, 5, 3))
    for (key, block) in blocks(t)
        block ≈ conj(blocks(tT)[key]) || return false
    end
    return true
end

function unvec(p::VectorPEPOTensor)
    t = p.tensor
    u = p.unitary
    iu = inv(u)
    @tensor tt[1 2 x1 x2; 3 4] := iu[x1 x2; ph] * t[1 2 ph; 3 4]
    ttt = permute(tt, (1, 2, 3), (5, 6, 4))
    return PEPOTensor(ttt)
end

function conjugate(t::AbstractTensorMap)
    return TensorMap(conj(t.data), codomain(t), domain(t))
end

const cj = conjugate

@doc raw"""
    trace(t::PEPOTensor{S}, bond::S) where {S<:IndexSpace}

"""
function trace(t::PEPOTensor{S}, bond::S) where {S<:IndexSpace}
    @tensor pt[w s; n e] := t[w s x; n e x]
    guess = InfiniteMPS([codomain(pt)[2]], [bond])
    ts, _, _ = leading_boundary(guess, InfiniteMPO(pt), Vumps(; maxiter=400, verbose=true))
    return expectation_value(ts, pt, 1), ts
end

trace(t::PEPOTensor{ComplexSpace}, χ::Integer) = trace(t, ℂ^χ)
