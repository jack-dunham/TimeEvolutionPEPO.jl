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

stepper_type(::Type{<:LindbladEquation}, ::Type) = LocalStepper
stepper_type(::Type{<:LindbladEquation}, ::Type{<:DMPO}) = AxialStepper

function tebdtensor(model::LindbladEquation, p1::IndexSpace, p2::IndexSpace, dt; compresstol=eps())


    # ket1 * ket2 * bra1 * bra2
    spacecod = p1 * p2 * p1' * p2'
    spacedom = p1 * p2 * p1' * p2'

    data = zeros(ComplexF64, dim(spacecod), dim(spacedom))
    symdata = zeros(ComplexF64, dims(spacecod * spacedom))

    if isempty(model.linops)
        vecf = thermalvectorise
        dt = dt / 2
    else
        vecf = vectorise
        data += 1 / 2 * sum(vecf, model.linops; init=zero(data))
    end

    # We divide by half again for local terms as in 2D we effectively apply these twice
    data += 1 / 2 * sum(vecf, model.onelocal; init=zero(data))

    data += sum(vecf, model.twolocal; init=zero(data))

    copyto!(symdata, exp(dt * data))

    tensor = TensorMap(symdata, spacecod, spacedom)

    # move site one to codomain and site two to domain
    tensor = permute(tensor, (1, 3, 5, 7), (2, 4, 6, 8))

    u, s, v, err = tsvd(tensor; trunc=truncbelow(compresstol))

    return u * sqrt(s), _transpose(sqrt(s) * v)
end

function broadcastmodel(model::OnAxis)
    axis = model.axis
    return broadcast
end

function (model::LindbladEquation)(physical, method::AbstractDMPO; kwargs...)
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

function broadcastmodel(
    ::Type{<:LindbladEquation}, model, physical, method::AbstractTEBD; axis, kwargs...
)
    model = unwrap(model)

    rv =
        timestep -> begin
            if axis == 1
                incr = (1, 0)
            elseif axis == 2
                incr = (0, 1)
            end

            p1p2 = broadcast(tuple, physical, circshift(physical, .-incr))

            uvs = broadcast(p1p2, model) do (p1, p2), modeli
                return tebdtensor(modeli, p1, p2, timestep; compresstol=method.compresstol)
            end

            lat = broadcast(tuple, first.(uvs), circshift(last.(uvs), incr))

            if method isa AxialTEBD


                lat = broadcast(lat) do (u,v)
                    @tensoropt out[ok ob ik ib; e w] :=
                        u[ok ob x y; e] * v[x y ik ib; w]
                    return out
                end
            end

            return lat
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
