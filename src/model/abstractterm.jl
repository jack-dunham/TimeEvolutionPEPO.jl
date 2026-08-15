abstract type AbstractTerm end
abstract type AbstractSumOfTerms <: AbstractTerm end

abstract type AbstractSingleTerm <: AbstractTerm end
abstract type AbstractAxialTerm <: AbstractSingleTerm end
abstract type AbstractRadialTerm <: AbstractSingleTerm end
abstract type AbstractManhattanTerm <: AbstractSingleTerm end

Base.length(h::AbstractTerm) = length(terms(h))
Base.iterate(h::AbstractTerm, state = 1) = iterate(terms(h), state)

Base.firstindex(h::AbstractTerm) = 1
Base.lastindex(h::AbstractTerm) = length(h)

Base.getindex(h::AbstractTerm, i) = getindex(terms(h),i)

"""
    $(TYPEDSIGNATURES)

Whether a term is a compact form of a sum that has to be expanded before it can be exponentiated,
as `PowOp` is for a sum of `ExpOp`s.
"""
isexpandable(::AbstractTerm) = false

"""
    $(TYPEDSIGNATURES)

Whether every term has support along a single lattice axis, as opposed to a radial or Manhattan
distance.
"""
isaxial(::AbstractTerm) = true
isaxial(::AbstractRadialTerm) = false
isaxial(::AbstractManhattanTerm) = false

Base.:+(A::AbstractTerm, B::AbstractTerm) = +(B, A)

terms(t::AbstractSingleTerm) = (t,)

function isaxial(h::AbstractSumOfTerms)
    for term in terms(h)
        if !isaxial(term)
            # Short-circuit if we find a non-axial term
            return false
        end
    end
    return true
end

Base.zero(h::AbstractSumOfTerms) = zero(typeof(h))
# physicaldim(term::AbstractTerm) = map(x -> size(x, 1), term.term)

convert_if_identity(I::UniformScaling) = Matrix{ComplexF64}(I, (2, 2))
convert_if_identity(x) = x

function Base.:+(t1::AbstractSingleTerm, t2::AbstractSingleTerm)
    newterms = TupleTools.vcat(terms(t1), terms(t2))
    return Hamiltonian(newterms, vcat(oneterms(t1), oneterms(t2)))
end

# Commutes
Base.:*(A::AbstractTerm, a::Number) = Base.:*(a, A)
Base.:*(a::Number, A::AbstractTerm) = @set A.a *= a

Base.:-(A::AbstractTerm) = -1 * A
Base.:-(A::AbstractTerm, B::AbstractTerm) = +(A, -B)

vectorise(term::AbstractTerm) = vectorise(typeof(term))(Matrix{ComplexF64}(term))

(T::Type{<:AbstractMatrix})(A::AbstractTerm) = T(term_to_matrix(A))
