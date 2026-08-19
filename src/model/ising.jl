"""
$(TYPEDEF)

Represents an Ising-like term of the form
```math
    \\hat{H}_{jk} = a \\hat{X}_j \\hat{X}_k + \\hat{D}_j
```
together with an (optional) on-site dissipator ``\\hat{\\mathcal{D}}_j``.
This type allows for certain exact low-rank truncations to take place when the term is exponentiated
into a two-site propagator. In that case, the two-local nearest-neighbor Suzuki-Trotter gate takes the form
```math
\\hat{g}_{jk}  = 
e^{\\frac{\\tau}{4}(
    \\hat{D}_j + \\hat{\\mathcal{D}}_j + 
    \\hat{D}_k + \\hat{\\mathcal{D}}_k
)}
e^{\\tau a \\hat{X}_j \\hat{X}_k }
e^{\\frac{\\tau}{4}(
    \\hat{D}_j + \\hat{\\mathcal{D}}_j + 
    \\hat{D}_k + \\hat{\\mathcal{D}}_k
)}.
```
# Constructors

    $(FUNCTIONNAME)([a = 1,], X::LocalOp{2}, [oneterms::Union{LocalOp{1}, Vector{LocalOp{1}}, dissipator::Dissipator])

Construct an Ising-like term with optional on-site terms `oneterms` and dissipator. Note, the
argument `X` is checked to be of Ising form, i.e. `X.C == X.B`.
"""
struct Ising{HL} <: AbstractSumOfTerms
    a::Float64
    CB::Matrix{ComplexF64}
    oneterms::Vector{LocalOp{1}}
    dissipator::Union{Nothing,Dissipator}
    function Ising(CB::LocalOp{2}, oneterms, ::Nothing)
        oneterms = vcat(LocalOp{1}[], oneterms)
        a, mat = checkisising(CB)
        return new{Hamiltonian}(a, mat, oneterms, nothing)
    end
    function Ising(CB::LocalOp{2}, oneterms, dissipator)
        oneterms = vcat(LocalOp{1}[], oneterms)
        a, mat = checkisising(CB)
        return new{Liouvillian}(a, mat, oneterms, dissipator)
    end
end

isising(op::LocalOp{2}) = op.C == op.B
function checkisising(op::LocalOp{2})
    if !isising(op)
        throw(ArgumentError("got `op.C != op.B`; provided `op` not an Ising term"))
    else
        mat = op.C
    end
    return op.a, mat
end

Ising(CB) = Ising(CB, LocalOp{1}[])

# Extra dissipator
Ising(CB, D::Dissipator) = Ising(CB, [], D)

# Extra lops, dissipator optional
Ising(CB, lops) = Ising(CB, lops, nothing)

function Base.:+(::Ising, term::AbstractTerm)
    throw(
        ArgumentError(
            "terms of type Ising are self contained and cannot be added to other terms"
        ),
    )
end

function Base.:*(a::Number, op::Ising)
    newop = Ising(a * op.a * LocalOp(op.CB, op.CB), op.oneterms, op.dissipator)
    return scalelocal(newop, a)
end

oneterms(L::Ising) = L.oneterms

function scalelocal(L::Ising, a)
    D = L.dissipator

    if !(D isa Nothing)
        D *= a
    end

    CB = L.CB
    return Ising(L.a * LocalOp(CB, CB), L.oneterms .* a, D)
end

isaxial(::Ising) = true

# `Ising` stores its coupling as a single matrix with C == B; the equivalent term is a * C ⊗ C.
_equation(t::Ising{Hamiltonian}) = Hamiltonian((t.a * LocalOp(t.CB, t.CB),), t.oneterms)
function _equation(t::Ising{Liouvillian})
    return Liouvillian(Hamiltonian((t.a * LocalOp(t.CB, t.CB),), t.oneterms), t.dissipator)
end
