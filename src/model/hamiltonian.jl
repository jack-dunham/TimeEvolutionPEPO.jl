"""
$(TYPEDEF)

Concrete type representing a Hamiltonian. Users should construct this object by adding terms
together using `localop`, `expop`, `powop2d` etc, rather than directly.
"""
struct Hamiltonian{T<:Tuple{Vararg{AbstractSingleTerm}}} <: AbstractSumOfTerms
    terms::T
    oneterms::Vector{LocalOp{1}}
end

Hamiltonian(term::AbstractSingleTerm, oneterms) = Hamiltonian((term,),oneterms)
Hamiltonian() = Hamiltonian((), LocalOp{1}[])
Hamiltonian(h) = Hamiltonian(h, LocalOp{1}[])
Hamiltonian(h0::LocalOp{1}) = Hamiltonian((), [h0])

function Base.show(io::IO, ::MIME"text/plain", eqn::Hamiltonian)

    n1 = length(oneterms(eqn))
    n2 = length(terms(eqn))

    println(io, "Hamiltonian containing:")
    println(io, "  $n1 on-site term(s)")
    return print(io, "  $n2 local term(s)")
end

Base.convert(::Type{Hamiltonian}, term::AbstractTerm) = Hamiltonian(term)

terms(h::Hamiltonian) = h.terms
"""
    $(TYPEDSIGNATURES)

The one-local terms of a model, held separately from the two-local ones because a bond carries half
of them per site.
"""
oneterms(h::Hamiltonian) = h.oneterms

# Hamiltonians have no lindblad operators associated with them
"""
    $(TYPEDSIGNATURES)

The `Dissipator` of a model, or `nothing` for a `Hamiltonian`.
"""
dissipator(::Hamiltonian) = nothing

Base.zero(::Type{<:Hamiltonian}) = Hamiltonian()

scalelocal(h::Hamiltonian, a) = @set h.oneterms .*= a

function Base.:+(A::Hamiltonian, B::AbstractTerm)
    H1 = TupleTools.vcat(terms(A), terms(B))
    H0 = vcat(oneterms(A), oneterms(B))
    return Hamiltonian(H1, H0)
end

function Base.:*(a::Number, H::Hamiltonian) 
    H1 = map(t -> a * t, terms(H))
    H0 = map(t -> a * t, oneterms(H))
    return Hamiltonian(H1, H0)
end

function Base.hash(ham::Hamiltonian, h::UInt)
    hashes = sort(map(hash, ham))
    onehashes = map(hash, oneterms(ham))
    h = hash(hashes, h)
    h = hash(onehashes, h)
    return h
end

Base.:(==)(h1::Hamiltonian, h2::Hamiltonian) = hash(h1) == hash(h2)

isexpandable(::Hamiltonian) = false
