# Linear algebra functions

const σ0 = Matrix{ComplexF64}([1 0; 0 1])
const σ1 = Matrix{ComplexF64}([0 1; 1 0])
const σ2 = Matrix{ComplexF64}([0 -im; im 0])
const σ3 = Matrix{ComplexF64}([1 0; 0 -1])

const σp = 1 / 2 * (σ1 + im * σ2) # Pauli+ matrix
const σm = 1 / 2 * (σ1 - im * σ2) # Pauli- matrix

const PAULI = (σ1,σ2,σ3)
const LADDER = (σm, σp)

function conjugate(t::AbstractTensorMap)
    return TensorMap(conj(t.data), codomain(t), domain(t))
end

const cj = conjugate
