abstract type AbstractModel <: AbstractLindbladEquation end

@kwdef struct Heisenberg <: AbstractModel
    Jx::Float64 = 0.0
    Jy::Float64 = 0.0
    Jz::Float64 = 0.0
    L::Vector{LindbladTerm}
end

ModelStyle(::Type{Heisenberg}) = IsAxial()

stepper_type(::Type{Heisenberg}, ::Type{<:AxialTEBD}) = AxialStepper
stepper_type(::Type, ::Type{<:GlobalTEBD}) = FullStepper

@kwdef struct J1J2Heisenberg <: AbstractModel
    Jx::Float64
    Jy::Float64
    Jz::Float64
    Kx::Float64
    Ky::Float64
    Kz::Float64
    L::Vector{LindbladTerm}
end

ModelStyle(::Type{J1J2Heisenberg}) = IsGlobal()

stepper_type(::Type{J1J2Heisenberg}, ::Type{<:TEPEPSO}) = FullStepper

function (M::Type{<:AbstractModel})(L...; kwargs...)
    itr = map(L) do op
        return op isa LindbladTerm ? op : lin(op)
    end

    return M(; L=collect(itr), kwargs...)
end

function paramkwargs(eqn::AbstractModel)
    rv = Pair{Symbol,Float64}[]
    for name in fieldnames(typeof(eqn))
        if name === :L
            continue
        end
        push!(rv,  (name => getfield(eqn, name)))
    end
    return rv
end

getparams(eqn) = map(last, paramkwargs(eqn))

linops(eqn::AbstractModel) = eqn.L
function broadcastmodel(
    ::Type{<:Heisenberg}, model, physical, method::AbstractTEBD; kwargs...
)
    model = unwrap(model)

    rv = timestep -> exact_tebd_time_stepper(model, physical, method, timestep; kwargs...)

    return rv
end

function broadcastmodel(
    ::Type{Combine}, model, physical, method::GlobalTEBD; kwargs...
)

    model = unwrap(model)

    rv = timestep -> exact_tebd_time_stepper(model, physical, method, timestep; kwargs...)

    return rv
end

function (model::Union{J1J2Heisenberg,Heisenberg})(physical, method::TEPEPSO; kwargs...)
    kw = paramkwargs(model)

    L = linops(model)

    rv =
        timestep -> begin
            fsmheisenberg(Matrix.(L), physical, timestep; extend=method.extend, kw...)
        end

    return rv
end

# function (model::Heisenberg)(physical, method::AxialTEBD; kwargs...)
#     return dt -> tebdtensor(model, physical, method, dt)
# end

function J1J2Heisenberg(L...; Jx=0, Jy=0, Jz=0, Kx=0, Ky=0, Kz=0)
    return J1J2Heisenberg(Jx, Jy, Jz, Kx, Ky, Kz, L...)
end

function J1J2Heisenberg(
    Jx, Jy, Jz, Kx, Ky, Kz, L::Vararg{Union{AbstractMatrix{<:Number},LindbladTerm}}
)
    itr = map(L) do op
        return op isa LindbladTerm ? op : lin(op)
    end

    return J1J2Heisenberg(Jx, Jy, Jz, Kx, Ky, Kz, collect(itr))
end

getparams(hm::J1J2Heisenberg) = (hm.Jx, hm.Jy, hm.Jz, hm.Kx, hm.Ky, hm.Kz)
twolocalequal(H1, H2) = getparams(H1) === getparams(H2)

function _exppp(t)
    C0 = zeros(ComplexF64, 2, 2)
    C1 = zeros(ComplexF64, 2, 2)

    C0[1, 1] = cosh(t)
    C0[2, 2] = sinh(t)

    C1[1, 2] = C1[2, 1] = sqrt(sinh(t) * cosh(t))

    return C0, C1
end

function expxx(t, L...)
    M = zeros(ComplexF64, 2, 2, 2, 2)

    C0, C1 = _exppp(t)

    # Fix physical indices
    @. M[1, 1, :, :] = M[2, 2, :, :] = C0
    @. M[1, 2, :, :] = M[2, 1, :, :] = C1

    return M
end

function expyy(t, L...)
    M = zeros(ComplexF64, 2, 2, 2, 2)

    C0, C1 = _exppp(t)

    # Fix physical indices
    @. M[1, 1, :, :] = M[2, 2, :, :] = C0
    M[1, 2, :, :] .= -im * C1
    M[2, 1, :, :] .= im * C1

    return M
end

function expzz(t)
    M = zeros(ComplexF64, 2, 2, 2, 2)

    C0, C1 = _exppp(t)

    # Fix physical indices
    M[1, 1, :, :] .= C0 + C1
    M[2, 2, :, :] .= C0 - C1

    return M
end

function expAA(t, A)
    d1, d2 = size(A)
    M = zeros(ComplexF64, d1, d2, 2, 2)

    C0, C1 = _exppp(t)

    for ind in CartesianIndices(A)
        @views M[ind, :, :] = C0 * one(A)[ind] + C1 * A[ind]
    end

    return M
end

function vectorise(t, AK, AB)
    MK = expAA(-t, AK)
    MB = expAA(t, AB)

    # MK = expzz(-t)
    # MB = expzz(t)

    M = zeros(ComplexF64, 2, 2, 2, 2, 2, 2, 2, 2)

    @tensor M[ok ob ik ib; ek eb wk wb] = MK[ok ik; ek wk] * MB[ob ib; eb wb]

    data = reshape(M, 2, 2, 2, 2, 4, 4)

    return data
end

function hopping(t::Number)
    D = 2 #bond dim
    M = zeros(ComplexF64, D, 2, 2)

    M[:, 1, 1] = [sqrt(Complex(cos(t))), sqrt(Complex(im * sin(t)))]
    M[:, 2, 2] = [sqrt(Complex(cos(t))), -sqrt(Complex(im * sin(t)))]
    @tensor Hzz[l, r, g, t] := M[l, s1, t] * M[r, s1, g]

    if !(Hzz ≈ permutedims(Hzz, (2, 1, 3, 4)))
        @warn "MPO not symmetric"
    end
    return Hzz
end

function vec_hopping(τ::Float64, V::Float64)
    x = τ * V
    @tensor Hzz[l, l', r, r', g, g', t, t'] :=
        hopping(x)[l, r, g, t] * hopping(-x)[l', r', g', t'] #exp(+) x exp(-) for time dynamics

    return reshape(Hzz, (4, 4, 4, 4))
end

function tebdtimestepper(
    style, model::AbstractUnitCell{G,L}, physical::AbstractUnitCell; kwargs...
) where {G,L<:J1J2Heisenberg}
    steps = exact_tebd_time_stepper(style, model, physical; kwargs...)
    return steps
end

function tebdtensor(model::Heisenberg, p1::IndexSpace, p2::IndexSpace, method::TEBD, dt; kwargs...)
    X, Y, Z = PAULI

    ops = Matrix{ComplexF64}[]

    Jx, Jy, Jz = getparams(model)

    expf = (xi, A) -> begin
        return cosh(xi) * kron(one(A), one(A)) + sinh(xi) * kron(A, A)
    end

    expfv = (xi, A) -> begin
        return kron(expf(-xi, A), expf(xi, transpose(A)))
    end

    if !iszero(Jx)
        xi = im * dt * Jx
        # push!(ops, vectorise(Jx / 4 * ham(X, X)))
        push!(ops, expfv(xi, X))
    end

    if !iszero(Jy)
        xi = im * dt * Jy
        # push!(ops, vectorise(Jy / 4 * ham(Y, Y)))
        push!(ops, expfv(xi, Y))
    end

    if !iszero(Jz)
        xi = im * dt * Jz
        # push!(ops, vectorise(Jz / 4 * ham(Z, Z)))
        push!(ops, expfv(xi, Z))
    end

    numops = length(ops)

    space = p1 * p2 * p1' * p2'

    # TODO: this is not generic atm.
    rank = dim(p1 * p1')

    lop = sum(model.L; init=zeros(ComplexF64, rank^2, rank^2)) do op
        return vectorise(op) / (4 * numops)
    end

    L = exp(dt * lop)

    rv = map(ops) do op
        tensor = TensorMap(L * op * L, space, space)

        H = permute(tensor, (1, 3, 5, 7), (2, 4, 6, 8))

        u, s, v, err = tsvd(H; trunc=truncdim(rank))

        @info "Compression error: $err for η=$(dim(codomain(s)))"

        U = u * sqrt(s)
        V = _transpose(sqrt(s) * v)

        return U, V
    end

    return rv
end

function tebdtensor(model::Heisenberg, p::IndexSpace, method::AbstractTEBD, dt; kwargs...)
    X, Y, Z = PAULI

    ops = Array{ComplexF64,6}[]

    if !iszero(model.Jx)
        data = vectorise(dt * im * model.Jx, X, transpose(X))
        push!(ops, data)
    end

    if !iszero(model.Jy)
        data = vectorise(dt * im * model.Jy, Y, transpose(Y))
        push!(ops, data)
    end

    if !iszero(model.Jz)
        data = vectorise(dt * im * model.Jz, Z, transpose(Z))
        push!(ops, data)
    end

    numops = length(ops)

    space = p * p' * p' * p

    lop = sum(model.L; init=zeros(ComplexF64, 4, 4)) do op
        return localvectorise(op) / (4 * numops)
    end

    lop = reshape(exp(dt * lop), dims(space))

    rv = map(ops) do op
        @tensor data[ok ob ik ib; e w] :=
            lop[ok ob; ook oob] * op[ook oob iik iib; e w] * lop[iik iib; ik ib]

        _, _, _, _, η1, η2 = size(data)

        D1 = mapreduce(ComplexSpace, ⊕, ones(Int, η1))
        D2 = mapreduce(ComplexSpace, ⊕, ones(Int, η2))

        return TensorMap(data, space, D1 * D2)
    end

    return rv
end

function exact_tebd_time_stepper(model, physical, method, tau; axis, inreverse=false)
    # if !all(y -> twolocalequal(y, model[1]), model)
    #     throw(
    #         ArgumentError(
    #             "model is not translationally invariant at the two-site operator level"
    #         ),
    #     )
    # end

    lattice_of_vectors = construct_exact_tebd_time_stepper(model, physical, method, tau; axis=axis)

    numopsvec = Int[]

    first_lattice = broadcast(lattice_of_vectors) do sitevector
        len = length(sitevector)
        push!(numopsvec, len)
        return getindex(sitevector, 1)
    end

    if allequal(numopsvec)
        numops = numopsvec[begin]
    else
        throw(ErrorException(""))
    end

    vector_of_lattices = finalize_exact_tebd_time_stepper(
        lattice_of_vectors, method, numops
    )

    return vector_of_lattices
end

function construct_exact_tebd_time_stepper(model, physical, method::TEBD, dt; axis)
    if axis == 1
        incr = (1, 0)
    elseif axis == 2
        incr = (0, 1)
    end

    phys1 = physical
    phys2 = circshift(physical, .-incr)

    uvs = broadcast(phys1, phys2, model) do p1, p2, modeli
        return tebdtensor(modeli, p1, p2, method, dt; compresstol=method.compresstol)
    end

    return uvs
end

function construct_exact_tebd_time_stepper(model, physical, method::AbstractTEBD, dt; axis)
    ts = broadcast(physical, model) do p, modeli
        return tebdtensor(modeli, p, method, dt; compresstol=method.compresstol)
    end
    return ts
end
function finalize_exact_tebd_time_stepper(
    lattice_of_vectors::Lattice{G,Vector{Tuple{U,V}}}, ::TEBD, numops=Int; axis
) where {G,U,V}

    ulats = [similar(lattice_of_vectors, U) for _ in 1:numops]
    vlats = [similar(lattice_of_vectors, V) for _ in 1:numops]

    for i in 1:numops
        lattice = broadcast(x -> getindex(x, i), lattice_of_vectors)

        ulat = ulats[i]
        vlat = vlats[i]

        for ind in eachindex(lattice)
            x = ind[1]
            y = ind[2]

            u1 = ulat[i]
            v2 = vlat[i]

            ulat[x, y] = u1
            if axis == 1
                vlat[x + 1, y] = v2
            elseif axis == 2
                vlat[x, y + 1] = v2
            end
        end
    end

    rv = map(ulats, vlats) do (u, v)
        return broadcast(tuple, u, v)
    end

    return rv
end

function finalize_exact_tebd_time_stepper(
    lattice_of_vectors::Lattice{G,Vector{T}}, ::AbstractTEBD, numops=Int
) where {G,T}
    first_lattice = getindex.(lattice_of_vectors, 1)

    vector_of_lattices = [first_lattice]

    for i in 2:numops
        next_lattice = broadcast(x -> getindex(x, i), lattice_of_vectors)
        push!(vector_of_lattices, next_lattice)
    end

    return vector_of_lattices
end
