# State layer on PEPSKit types. `InfinitePEPO` carries the tensors and `SUWeight` the bond weights,
# which PEPSKit keeps outside the tensors rather than absorbing their square roots.

"""
$(TYPEDEF)

Singleton selecting the infinite-temperature density matrix as an initial state.

See also: [`productpepo`](@ref).
"""
struct ThermalState end

"""
    $(TYPEDSIGNATURES)

Spin-1/2 density matrix with Bloch vector `v`. Unlike the constructor this replaces, `v` is used as
given: `norm(v) < 1` yields a mixed state and `norm(v) > 1` is rejected.
"""
function bloch(v)
    norm(v) > 1 + eps() && throw(ArgumentError("Bloch vector of length $(norm(v)) exceeds 1"))
    return (σ0 + sum(a * σ for (a, σ) in zip(v, PAULI))) / 2
end

_localdensitymatrix(rho::AbstractMatrix) = ComplexF64.(rho)
_localdensitymatrix(v::Tuple) = bloch(v)
_localdensitymatrix(::ThermalState) = ComplexF64.(σ0) / 2

_indexspace(d::Integer) = ComplexSpace(d)
_indexspace(V::IndexSpace) = V

"""
    productpepo(init, bond) -> InfinitePEPO

Density-matrix PEPO of a product state, one tensor per unit-cell site. `init` is a local density
matrix, a spin-1/2 Bloch vector, a [`ThermalState`](@ref), or a matrix of any of those giving the
unit cell; `bond` is the virtual dimension or space, padded with zeros beyond the first basis vector.
A matrix of numbers is one local density matrix, while a matrix of matrices, tuples or
`ThermalState`s is a unit cell.

Physical indices follow PEPSKit's `PEPOTensor` convention, `P ⊗ P´ ← N ⊗ E ⊗ S ⊗ W`, with index 1
outgoing (ket, left-multiplied) and index 2 incoming (bra, right-multiplied).
"""
productpepo(init, bond) = productpepo(fill(init, 1, 1), bond)
productpepo(rho::AbstractMatrix{<:Number}, bond) = productpepo(fill(rho, 1, 1), bond)

function productpepo(inits::AbstractMatrix, bond)
    V = _indexspace(bond)
    rhos = _localdensitymatrix.(inits)
    P = ComplexSpace(size(first(rhos), 1))
    all(rho -> size(rho) == (dim(P), dim(P)), rhos) ||
        throw(DimensionMismatch("local density matrices differ in size"))

    cod, dom = P ⊗ P', V ⊗ V ⊗ V' ⊗ V'
    tensors = map(rhos) do rho
        arr = zeros(ComplexF64, dim(P), dim(P), dim(V), dim(V), dim(V), dim(V))
        arr[:, :, 1, 1, 1, 1] = rho
        return TensorMap(arr, cod ← dom)
    end

    return InfinitePEPO(reshape(tensors, size(rhos)..., 1))
end

"""
    $(TYPEDSIGNATURES)

Entropy of each bond weight, normalised by `log` of its dimension when `normalized = true`.

Trivial weights for a state come from PEPSKit's `SUWeight(pepo)`. A product state has Schmidt spectrum
`(1, 0, …)` on every bond, so those weights are a gauge choice rather than the true spectrum; they are
invertible, which the spectrum is not, and the first update replaces them.
"""
function bondentropy(weights::SUWeight; normalized = false)
    return map(collect(weights)) do w
        λ = diag(convert(Array, w))
        λ = λ / sum(λ)
        entropy = sum(x -> x ≈ 0 ? zero(x) : -x * log(x), λ)
        return entropy / log(length(λ))^normalized
    end
end
