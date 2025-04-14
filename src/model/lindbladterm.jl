struct LindbladTerm <: AbstractTerm
    term::NTuple{1,Matrix{ComplexF64}}
    coeff::ComplexF64
    LindbladTerm(term::NTuple{1,Matrix{ComplexF64}}, c) = new(term, c)
end

scaleterm(a, A::LindbladTerm) = LindbladTerm(A.term, a * A.coeff)

(A::Type{<:AbstractMatrix})(term::LindbladTerm) = term_to_matrix(A, term)

# lin(L) = newterm(LindbladTerm, (L,), 1)
lin(L) = LindbladTerm((L,), 1.0)

function vectorise(::Type{<:LindbladTerm})
    anticomm = L -> kron(L' * L, one(L)) + kron(one(L), transpose(L' * L))

    return twolocal(L -> kron(L, transpose(L')) - 1 / 2 * anticomm(L))
end

thermalvectorise(term::AbstractTerm) = thermalvectorise(typeof(term))(Matrix{ComplexF64}(term))
localvectorise(term::AbstractTerm) = localvectorise(typeof(term))(Matrix{ComplexF64}(term))

function localvectorise(::Type{<:LindbladTerm})
    anticomm = L -> kron(L' * L, one(L)) + kron(one(L), transpose(L' * L))

    return L -> kron(L, transpose(L')) - 1 / 2 * anticomm(L)
end
