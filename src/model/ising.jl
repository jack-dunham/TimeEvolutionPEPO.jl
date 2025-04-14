struct Ising{HL} <: AbstractSumOfTerms
    a::Float64
    CB::Matrix{ComplexF64}
    oneterms::Vector{LocalOp{1}}
    dissipator::Union{Nothing,Dissipator}
    Ising(a, CB, oneterms) = new{Hamiltonian}(a, CB, oneterms, nothing)
    Ising(a, CB, oneterms, dissipator) = new{Liouvillian}(a, CB, oneterms, dissipator)
end

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
