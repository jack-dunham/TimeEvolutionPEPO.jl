struct HamiltonianTerm{N} <: AbstractTerm{N}
    term::NTuple{N,Matrix{ComplexF64}}
    coeff::ComplexF64
end

scaleterm(a, A::HamiltonianTerm) = HamiltonianTerm(A.term, a * A.coeff)

(A::Type{<:AbstractArray})(term::HamiltonianTerm) = term_to_matrix(A, term)

ham(A) = newterm(HamiltonianTerm{1}, (A,), 1)
ham(A, B) = newterm(HamiltonianTerm{2}, (A,B), 1)

vectorise(::Type{HamiltonianTerm{1}}) = twolocal(vectorise(HamiltonianTerm{2}))

# Vectorise the superoperators; we do this the non-julian way (row major) as the rest
# of the code assumes (ket, bra) ordering (whoops).
function vectorise(::Type{HamiltonianTerm{2}})
    return H -> -im * (kron(H, one(H)) - kron(one(H), transpose(H)))
end

# We need both local terms that have support on two sites, and therefore need to divide
# by two to avoid double counting in the brickwork pattern
twolocal(f) = H -> 1 / 2 * (f(kron(one(H), H)) + f(kron(H, one(H))))
