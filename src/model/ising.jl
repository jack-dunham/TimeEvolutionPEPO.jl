"""
$(TYPEDEF)

Represents an Ising-like term of the form
```math
    \\hat{H}_{jk} = a \\hat{X}_j \\hat{X}_k + \\hat{D}_j
```
together with an (optional) on-site dissipator ``\\hat{\\mathcal{D}}_j``.
This type allows for certain exact low-rank truncations to take place when using 
[`TEBD`](@ref). In that case, the two-local nearest-neighbor Suzuki-Trotter gate takes the form
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
        return new{Liouvillian}(a, mat, oneterms, nothing)
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
Ising(CB, lops, D=nothing) = Ising(CB, lops, D)

function Base.:+(::Ising, term::AbstractTerm)
    throw(
        ArgumentError(
            "terms of type Ising are self contained and cannot be added to other terms"
        ),
    )
end

Base.:*(a::Number, op::Ising) = scalelocal(@set(op.a .*= a), a)

oneterms(L::Ising) = L.oneterms

function scalelocal(L::Ising, a)
    @reset L.oneterms .*= a
    D = L.dissipator

    if !(D isa Nothing)
        @reset L.dissipator *= a
    end

    return L
end

isaxial(::Ising) = true
