abstract type AbstractTerm{N} end

# This only accepts concrete types
function newterm(::Type{T}, term, args...) where {N,T<:AbstractTerm{N}}
    term = map(convert_if_identity, term)
    args = map(ComplexF64, args)
    return T(term, args...)
end

physicaldim(term::AbstractTerm) = map(x -> size(x, 1), term.term)

function Base.:(==)(t1::T, t2::T) where {T<:AbstractTerm}
    return all(name -> getfield(t1, name) == getfield(t2, name), fieldnames(T))
end

convert_if_identity(I::UniformScaling) = Matrix{ComplexF64}(I, (2, 2))
convert_if_identity(x) = x

Base.:*(a::Number, A::AbstractTerm) = scaleterm(a, A)
Base.:*(A::AbstractTerm, a::Number) = Base.:*(a, A)

function term_to_matrix(A, term)
    mat = term.coeff * kron(1, term.term...)
    return convert(A, mat)
end
