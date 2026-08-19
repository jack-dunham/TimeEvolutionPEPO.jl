"""
$(TYPEDEF)

Represents the dissipator in the Liouvillian super-operator.
"""
struct Dissipator <: AbstractSumOfTerms
    terms::Vector{Matrix{ComplexF64}}
end

"""
    $(FUNCTIONNAME)(ops::Vector{<:AbstractMatrix})
    $(FUNCTIONNAME)(ops::AbstractMatrix...)

Construct a `Dissipator` from a collect of onsite Lindblad operators given by `ops`.
"""
Dissipator(ops::Vararg{AbstractMatrix}) = Dissipator(collect(ops))

# Mutliplying the whole dissipator by a can be achieved by multiplying each
# Lindblad operator by sqrt(a)
Base.:*(a::Number, D::Dissipator) = Dissipator(map(t -> sqrt(a) * t, terms(D)))
Base.:+(D1::Dissipator, D2::Dissipator) = Dissipator(vcat(D1.terms, D2.terms))
Base.:+(D::Dissipator, h::AbstractTerm) = zero(Hamiltonian) + h + D

terms(D::Dissipator) = D.terms

isaxial(D::Dissipator) = isempty(terms(D))

_anticomm(L) = kron(L' * L, one(L)) + kron(one(L), transpose(L' * L))
_dissipatorvec(L) = kron(L, transpose(L')) - 1 / 2 * _anticomm(L)

function vectorise(dissipator::Dissipator, locality=2)
    if isempty(dissipator)
        throw(
            ArgumentError(
                "cannot vectorise a `Dissipator` containing no Lindblad operators"
            ),
        )
    end

    if locality == 1
        f = _dissipatorvec
    elseif locality == 2
        f = twolocal(_dissipatorvec)
    else
        throw(
            ArgumentError(
                "vectorising a `Dissipator` over $locality sites not yet supported"
            ),
        )
    end

    matrix = sum(f, dissipator)

    return matrix
end

"""
$(TYPEDEF)

Concrete type representing a Liouvillian super-operator.

## Fields

$(TYPEDFIELDS)
"""
struct Liouvillian{T} <: AbstractSumOfTerms
    hamiltonian::Hamiltonian{T}
    dissipator::Dissipator
end

dissipator(L::Liouvillian) = L.dissipator
hamiltonian(L::Liouvillian) = L.hamiltonian
terms(L::Liouvillian) = terms(L.hamiltonian)
oneterms(L::Liouvillian) = oneterms(L.hamiltonian)

# `terms(L)` only sees the Hamiltonian part, so the `AbstractSumOfTerms` fallback never inspects
# the dissipator; check it explicitly here.
isaxial(L::Liouvillian) = isaxial(L.hamiltonian) && isaxial(L.dissipator)

"""
    $(FUNCTIONNAME)(H::Hamiltonian)

Construct a `Liouvillian` with no Lindblad operators.

    $(FUNCTIONNAME)(H::Hamiltonian, D::Dissipator)
    $(FUNCTIONNAME)(H::Hamiltonian, ops::Vector{<:AbstractMatrix})
    $(FUNCTIONNAME)(H::Hamiltonian, ops::AbstractMatrix...)

Construct `Liouvillian` from `H` and dissipator `D`, or any number of Lindblad operators
`ops`.
"""
Liouvillian(H) = Liouvillian(H, [])
Liouvillian(H, ops) = Liouvillian(H, Dissipator(ops))
Liouvillian(H::Hamiltonian, ops::Vararg{<:Matrix}) = Liouvillian(H, collect(ops))

Base.convert(::Type{Liouvillian}, H::Hamiltonian) = Liouvillian(H)
Base.convert(::Type{Liouvillian}, term) = Liouvillian(convert(Hamiltonian, term))
Base.convert(::Type{Liouvillian}, term::Dissipator) = Liouvillian(zero(Hamiltonian), term)

function scalelocal(L::Liouvillian, a)
    @reset L.hamiltonian.oneterms .*= a
    @reset L.dissipator *= a

    return L
end

function Base.show(io::IO, ::MIME"text/plain", eqn::Liouvillian)
    n1 = length(oneterms(eqn))
    n2 = length(terms(eqn))

    nd = length(eqn.dissipator)

    println(io, "Liouvillian containing:")
    println(io, "  $n1 on-site term(s)")
    println(io, "  $n2 local term(s)")
    # println(io, "  $n3 non-local terms")
    return print(io, "  $nd lindblad operator(s)")
end

Base.:+(L::Liouvillian, H::AbstractTerm) = Liouvillian(L.hamiltonian + H, L.dissipator)
Base.:+(H::Hamiltonian, D::Dissipator) = Liouvillian(H, D)
Base.:+(L::Liouvillian, D::Dissipator) = Liouvillian(L.hamiltonian, dissipator(L) + D)

Base.promote_rule(::Type{<:Hamiltonian}, ::Type{<:Liouvillian}) = Liouvillian
