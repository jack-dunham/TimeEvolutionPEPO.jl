abstract type AbstractLindbladEquation end

Base.zero(eqn::AbstractLindbladEquation) = zero(typeof(eqn))

Base.:+(B::AbstractTerm, eqn::AbstractLindbladEquation) = +(eqn, B)

function Base.:+(eqn::AbstractLindbladEquation, B::AbstractTerm)
    eqnout = zero(eqn)
    eqnout = axpy!(1, eqn, eqnout)
    eqnout = push!!(eqnout, B)
    return eqnout
end

function Base.:+(eqn1::AbstractLindbladEquation, eqn2::AbstractLindbladEquation)
    eqnout = zero(eqn1)
    eqnout = axpy!(1, eqn1, eqnout)
    eqnout = axpy!(1, eqn2, eqnout)
    return eqnout
end

struct ZeroLindbladEquation <: AbstractLindbladEquation end

Base.zero(::Type{<:AbstractLindbladEquation}) = ZeroLindbladEquation()

function LinearAlgebra.axpy!(
    ::Number, ::ZeroLindbladEquation, eqn2::AbstractLindbladEquation
)
    return eqn2
end

struct LindbladEquation <: AbstractLindbladEquation
    onelocal::Vector{HamiltonianTerm{1}}
    twolocal::Vector{HamiltonianTerm{2}}
    longrange::Vector{LongRangeTerm}
    linops::Vector{LindbladTerm}
    physicaldim::Int
end

LindbladEquation(p::Int) = LindbladEquation([], [], [], [], p::Int)

function Base.show(io::IO, ::MIME"text/plain", eqn::LindbladEquation)
    d = physicaldim(eqn)

    n1 = length(eqn.onelocal)
    n2 = length(eqn.twolocal)

    nl = length(eqn.longrange)

    nd = length(eqn.linops)

    println(io, "LindbladEquation with d = $d containing:")
    println(io, "  $n1 local terms")
    println(io, "  $n2 nearest-neighbour terms")
    println(io, "  $nl long-range terms")
    return print(io, "  $nd lindblad operators")
end

physicaldim(eqn::LindbladEquation) = eqn.physicaldim

function physicalspace(S::Type{<:IndexSpace}, eqn::LindbladEquation)
    d = eqn.physicaldim
    return ⊕(oneunit(S)^d...)
end

function push!!(::ZeroLindbladEquation, term::AbstractTerm)
    outeqn = LindbladEquation(physicaldim(term)[1])
    push!!(outeqn, term)
    return outeqn
end

function push!!(L::LindbladEquation, A::AbstractTerm)
    ld = physicaldim(L)
    ad = physicaldim(A)
    if ld == ad[1]
        push!(L, A)
        return L
    else
        throw(
            DimensionMismatch(
                "cannot add term of dimension $ad to lindblad equation of dimension $ld"
            ),
        )
    end
end

Base.push!(L::LindbladEquation, A::HamiltonianTerm{1}) = push!(L.onelocal, A)
Base.push!(L::LindbladEquation, A::HamiltonianTerm{2}) = push!(L.twolocal, A)
Base.push!(L::LindbladEquation, A::LongRangeTerm) = push!(L.longrange, A)
Base.push!(L::LindbladEquation, A::LindbladTerm) = push!(L.linops, A)

## MATHS

function Base.:+(A::AbstractTerm, B::AbstractTerm)
    eqn = push!!(zero(LindbladEquation), A)
    eqn = push!!(eqn, B)
    return eqn
end

Base.:*(eqn::LindbladEquation, a::Number) = *(a, eqn)
function Base.:*(a::Number, eqn::LindbladEquation)
    eqnout = zero(eqn)
    axpy!(a, eqn, eqnout)
    return eqnout
end

function LinearAlgebra.axpy!(
    a::Number, eqn1::LindbladEquation, eqn2::AbstractLindbladEquation
)
    for name in (:onelocal, :twolocal, :longrange, :linops)
        for term in getfield(eqn1, name)
            eqn2 = push!!(eqn2, a * term)
        end
    end
    return eqn2
end

function Base.:(==)(t1::T, t2::T) where {T<:LindbladEquation}
    return all(name -> getfield(t1, name) == getfield(t2, name), fieldnames(T))
end

##
