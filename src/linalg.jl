# Linear algebra functions

const σ0 = TensorMap(Matrix{ComplexF64}([1 0; 0 1]), ℂ^2, ℂ^2) # Identity matrix
const σ1 = TensorMap(Matrix{ComplexF64}([0 1; 1 0]), ℂ^2, ℂ^2) # Pauli X matrix
const σ2 = TensorMap(Matrix{ComplexF64}([0 -im; im 0]), ℂ^2, ℂ^2) # Pauli Y matrix
const σ3 = TensorMap(Matrix{ComplexF64}([1 0; 0 -1]), ℂ^2, ℂ^2) # Pauli Z matrix

const σp = 1 / 2 * (σ1 + im * σ2) # Pauli+ matrix
const σm = 1 / 2 * (σ1 - im * σ2) # Pauli- matrix

const pauli = [σ1; σ2; σ3; σ0]

function conjugate(t::AbstractTensorMap)
    return TensorMap(conj(t.data), codomain(t), domain(t))
end

const cj = conjugate
