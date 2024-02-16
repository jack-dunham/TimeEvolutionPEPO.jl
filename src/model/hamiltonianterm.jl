struct HamiltonianTerm{N} <: AbstractTerm{N}
    term::NTuple{N,Matrix{ComplexF64}}
    coeff::ComplexF64
end

HamiltonianTerm(term::NTuple{N,Any}, a=1) where {N} = newterm(HamiltonianTerm{N}, term, a)
scaleterm(a, A::HamiltonianTerm) = HamiltonianTerm(A.term, a * A.coeff)

(A::Type{<:AbstractArray})(term::HamiltonianTerm) = term_to_matrix(A, term)

ham(A) = HamiltonianTerm((A,))
ham(A, B) = HamiltonianTerm((A, B))
