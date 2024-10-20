struct HamiltonianTerm{N} <: AbstractTerm{N}
    term::NTuple{N,Matrix{ComplexF64}}
    coeff::ComplexF64
end

scaleterm(a, A::HamiltonianTerm) = HamiltonianTerm(A.term, a * A.coeff)

(A::Type{<:AbstractArray})(term::HamiltonianTerm) = term_to_matrix(A, term)

ham(A) = newterm(HamiltonianTerm{1}, (A,), 1)
ham(A, B) = newterm(HamiltonianTerm{2}, (A,B), 1)
