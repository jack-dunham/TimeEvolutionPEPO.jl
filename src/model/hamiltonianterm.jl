struct HamiltonianTerm{N} <: AbstractTerm
    term::NTuple{N,Matrix{ComplexF64}}
    coeff::ComplexF64
end

scaleterm(a, A::HamiltonianTerm) = HamiltonianTerm(A.term, a * A.coeff)

(A::Type{<:AbstractMatrix})(term::HamiltonianTerm) = term_to_matrix(A, term)

ham(A) = newterm(HamiltonianTerm{1}, (A,), 1)
# ham(A, B) = newterm(HamiltonianTerm{2}, (A,B), 1)
ham(a, A, B) = HamiltonianTerm{2}((A,B), a)


vectorise(::Type{HamiltonianTerm{1}}) = twolocal(vectorise(HamiltonianTerm{2}))

function vectorise(::Type{HamiltonianTerm{2}})
    return H -> -im * (kron(H, one(H)) - kron(one(H), transpose(H)))
end
