# Linear algebra functions

const σ0 = Matrix{ComplexF64}([1 0; 0 1])
const σ1 = Matrix{ComplexF64}([0 1; 1 0])
const σ2 = Matrix{ComplexF64}([0 -im; im 0])
const σ3 = Matrix{ComplexF64}([1 0; 0 -1])

const σp = 1 / 2 * (σ1 + im * σ2) # Pauli+ matrix
const σm = 1 / 2 * (σ1 - im * σ2) # Pauli- matrix

const PAULI = (σ1, σ2, σ3)
const LADDER = (σm, σp)

const SX1 = Matrix{ComplexF64}([0 1 0; 1 0 1; 0 1 0] / sqrt(2))
const SY1 = Matrix{ComplexF64}(im * [0 -1 0; 1 0 -1; 0 1 0] / sqrt(2))
const SZ1 = Matrix{ComplexF64}([1 0 0; 0 0 0; 0 0 -1])

const SPIN1 = (SX1, SY1, SZ1)

function conjugate!(t::AbstractTensorMap)
    for (_, block) in blocks(t)
        conj!(block)
    end
    return t
end
conjugate(t::AbstractTensorMap) = conjugate!(deepcopy(t))

const cj = conjugate

lowering(space) = lowering(ComplexF64, space)
function lowering(T, cutoff)
    A = zeros(T, cutoff, cutoff)
    for i in 1:(cutoff - 1)
        A[i,i + 1] = sqrt(i)
    end
    return A
end
function lowering(T, space::IndexSpace)
    cutoff = dim(space)
    return TensorMap(lowering(T, cutoff), space, space)
end
