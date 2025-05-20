struct Ising{HL} <: AbstractSumOfTerms
    a::Float64
    CB::Matrix{ComplexF64}
    oneterms::Vector{LocalOp{1}}
    dissipator::Union{Nothing,Dissipator}
    Ising(a, CB, oneterms, ::Nothing) = new{Hamiltonian}(a, CB, oneterms, nothing)
    Ising(a, CB, oneterms, dissipator) = new{Liouvillian}(a, CB, oneterms, dissipator)
end

Ising(CB) = Ising(CB, LocalOp{1}[])
Ising(CB, dissipator::Dissipator) = Ising(CB, LocalOp{1}[], dissipator)
function Ising(CB, oneterms::Vector{LocalOp{1}}, dissipator=nothing)
    return Ising(1, CB, oneterms, dissipator)
end

function Base.:+(::Ising, term::AbstractTerm)
    throw(
        ArgumentError(
            "Terms of type Ising are self contained and cannot be added to other terms"
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
