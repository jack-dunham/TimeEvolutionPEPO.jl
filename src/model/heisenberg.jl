function broadcastmodel(::Type{Combine}, model, physical, method::GlobalTEBD; kwargs...)
    model = unwrap(model)

    rv = timestep -> exact_tebd_time_stepper(model, physical, method, timestep; kwargs...)

    return rv
end
function broadcastmodel(
    ::Type{Combine}, model, physical, method::GlobalClusterMPO; kwargs...
)
    model = unwrap(model)

    rv =
        timestep -> begin
            mpo = MPO(model)

            data = clustermpo(
                mpo,
                timestep,
                method.order;
                method.extend,
                method.compress,
                method.approx,
            )

            tensor = clustermpotensor(data, physical)

            return tensor
        end

    return rv
end

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

function vectorise(t, AK, AB; thermal=false)
    if thermal
        c1 = -t
        c2 = -t
    else
        c1 = -im * t
        c2 = im * t
    end

    MK = expAA(c1, AK)
    MB = expAA(c2, AB)

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

function tebdtensor(model::Ising, p1::IndexSpace, p2::IndexSpace, dt; kwargs...)
    X, Y, Z = PAULI

    ops = Matrix{ComplexF64}[]

    linops = model.dissipator

    if isnothing(linops)
        thermal = true
        c1 = -1 / 2
        c2 = -1 / 2
    else
        c1 = -im
        c2 = im
    end

    # 

    CB = model.CB
    if CB * CB ≈ one(CB)
        expf = (xi, A) -> begin
            return cosh(xi) * kron(one(A), one(A)) + sinh(xi) * kron(A, A)
        end
    else
        expf = (xi, A) -> begin
            return exp(xi * kron(A, A))
        end
    end

    expfv = (xi, A) -> begin
        return kron(expf(c1 * xi, A), expf(c2 * xi, transpose(A)))
    end

    xi = dt * model.a

    op = expfv(xi, CB)

    # ket ket bra bra
    space = p1 * p2 * p1' * p2'

    # TODO: this is not generic atm.
    rank = dim(p1 * p1')

    id = one(X)

    lop = kron(zero(id), zero(id))
    lop = kron(lop, lop)

    for term in oneterms(model)
        h = term.a
        A = term.D

        # Make two-local
        hA = 1 / 2 * (kron(id, A) + kron(A, id))

        # Then vectorise k1 k2 ; b1 b2
        lop += h * (c1 * kron(hA, one(hA)) + c2 * kron(one(hA), transpose(hA)))
    end

    if !isnothing(model.dissipator)
        lop += vectorise(model.dissipator, 2)
    end

    L = exp(dt / 2 * lop)

    # k1 k2 b1 b2 
    tensor = TensorMap(L * op * L, space, space)

    H = permute(tensor, (1, 3, 5, 7), (2, 4, 6, 8))

    u, s, v, err = tsvd(H; trunc=truncdim(rank))

    @info "Compressed to η=$(dim(codomain(s))) with error $err"

    U = u * sqrt(s)
    V = _transpose(sqrt(s) * v)

    return U, V
end

function tebdtensor(model::Ising, p::IndexSpace, method::AbstractTEBD, dt; kwargs...)
    X, Y, Z = PAULI

    D = model.dissipator

    if length(D) == 0
        thermal = true
        c1 = -1
        c2 = -1
        dt = dt / 2
    else
        thermal = false
        c1 = -im
        c2 = im
    end
    H = model.CB

    op = vectorise(dt * model.a, H, transpose(H); thermal=thermal)

    space = p * p' * p' * p

    id = one(X)

    # Local operators
    lop = kron(zero(id), zero(id))

    for term in model.oneterms
        A = term.D
        h = term.a

        lop += h / 2 * (c1 * kron(A, id) + c2 * kron(id, transpose(A)))
    end

    # Local operators
    lop += sum(D; init=zero(lop)) do linop
        return 0.5 * vectorise(linop, 1)
    end

    lop = reshape(exp(dt * lop), dims(space))

    @tensor op[ok ob ik ib; e w] :=
        lop[ok ob; ook oob] * op[ook oob iik iib; e w] * lop[iik iib; ik ib]

    _, _, _, _, η1, η2 = size(op)

    D1 = mapreduce(ComplexSpace, ⊕, ones(Int, η1))
    D2 = mapreduce(ComplexSpace, ⊕, ones(Int, η2))

    return TensorMap(data, space, D1 * D2')
end

function exact_tebd_time_stepper(model, physical, method, tau; axis, inreverse=false)
    # if !all(y -> twolocalequal(y, model[1]), model)
    #     throw(
    #         ArgumentError(
    #             "model is not translationally invariant at the two-site operator level"
    #         ),
    #     )
    # end

    lattice_of_vectors = construct_exact_tebd_time_stepper(
        model, physical, method, tau; axis=axis
    )

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
        lattice_of_vectors, method, numops; axis=axis
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
    lattice_of_vectors::UnitCell{G,Vector{Tuple{U,V}}}, ::TEBD, numops=Int; axis
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

            lat = lattice_of_vectors[x, y]

            u1, v2 = lat[i]

            ulat[x, y] = u1
            if axis == 1
                vlat[x + 1, y] = v2
            elseif axis == 2
                vlat[x, y + 1] = v2
            end
        end
    end

    rv = map(ulats, vlats) do u, v
        return broadcast(tuple, u, v)
    end

    return rv
end

function finalize_exact_tebd_time_stepper(
    lattice_of_vectors::UnitCell{G,Vector{T}}, ::AbstractTEBD, numops=Int; kwargs...
) where {G,T}
    first_lattice = getindex.(lattice_of_vectors, 1)

    vector_of_lattices = [first_lattice]

    for i in 2:numops
        next_lattice = broadcast(x -> getindex(x, i), lattice_of_vectors)
        push!(vector_of_lattices, next_lattice)
    end

    return vector_of_lattices
end
