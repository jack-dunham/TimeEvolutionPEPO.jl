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

struct LindbladEquation{N} <: AbstractLindbladEquation
    onelocal::Vector{HamiltonianTerm{1}}
    twolocal::Vector{HamiltonianTerm{2}}
    longrange::Vector{LongRangeTerm}
    linops::Vector{LindbladTerm}
    physicaldim::Int
end

stepper_type(::Type{<:LindbladEquation{N}}, _) where {N} = lat -> AxialStepper(lat, N)

function LindbladEquation(onelocal, twolocal, longrange, linops, physicaldim)
    rv = LindbladEquation{(1, 2)}(onelocal, twolocal, longrange, linops, physicaldim)
    return rv
end

function haxis(L::LindbladEquation)
    return LindbladEquation{1}(L.onelocal, L.twolocal, L.longrange, L.linops, L.physicaldim)
end
function vaxis(L::LindbladEquation)
    return LindbladEquation{2}(L.onelocal, L.twolocal, L.longrange, L.linops, L.physicaldim)
end

function tebdtensor(model::LindbladEquation, p1::IndexSpace, p2::IndexSpace, dt; compress)

    # c2 = H -> (-im * (H ⊗ one(H) - one(H) ⊗ transpose(H)))
    # d2 = L -> L ⊗ transpose(L') - 1 / 2 * (L' * L ⊗ one(L) + one(L) * transpose(L' * L))
    #
    # # We need both local terms that have support on two sites, and therefore need to divide
    # # by two to avoid double counting in the brickwork pattern
    # c1 = H -> 1 / 2 * (c2(one(H) ⊗ H) + c2(H ⊗ one(H)))
    # d1 = L -> 1 / 2 * (d2(one(L) ⊗ L) + d2(L ⊗ one(L)))

    # ket1 * ket2 * bra1 * bra2
    spacecod = p1 * p2 * p1' * p2'
    spacedom = p1 * p2 * p1' * p2'

    data = zeros(ComplexF64, dim(spacecod), dim(spacedom))
    symdata = zeros(ComplexF64, dims(spacecod * spacedom))

    # We divide by half again for local terms as in 2D we effectively apply these twice
    data = 1 / 2 * sum(vectorise, model.onelocal; init=data)
    data = 1 / 2 * sum(vectorise, model.linops; init=data)

    data = sum(vectorise, model.twolocal; init=data)

    copyto!(symdata, exp(dt * data))

    tensor = TensorMap(symdata, spacecod, spacedom)

    # move site one to codomain and site two to domain
    tensor = permute(tensor, (1, 3, 5, 7), (2, 4, 6, 8))

    u, s, v = tsvd(tensor; trunc=compress)

    return u * sqrt(s), _transpose(sqrt(s) * v)
end

function (model::LindbladEquation)(physical, method::DMPO)
    rv =
        timestep -> begin
            mpo = MPO(model)

            data = dynamicalmpo(
                mpo,
                timestep,
                method.order;
                method.extend,
                method.compress,
                method.approx,
            )

            tensor = dmpotensor(data, physical)

            return tensor
        end

    return rv
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
