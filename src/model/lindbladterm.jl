struct LindbladTerm <: AbstractTerm{1}
    term::NTuple{1,Matrix{ComplexF64}}
    coeff::ComplexF64
    LindbladTerm(term::NTuple{1,Matrix{ComplexF64}}, c) = new(term, c)
end

scaleterm(a, A::LindbladTerm) = LindbladTerm(A.term, a * A.coeff)

(A::Type{<:AbstractArray})(term::LindbladTerm) = term_to_matrix(A, term)

lin(L) = newterm(LindbladTerm, (L,), 1)
