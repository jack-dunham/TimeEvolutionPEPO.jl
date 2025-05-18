struct AxialHeisenberg <: AbstractModel
    hx::Float64
    hy::Float64
    hz::Float64
    Jx::Float64
    Jy::Float64
    Jz::Float64
    Kx::Float64
    Ky::Float64
    Kz::Float64
    L::Dissipator
end

struct AxialIsing <: AbstractModel
    J::Float64
    K::Float64
    L::Dissipator
end

Base.broadcastable(o::AxialHeisenberg) = Ref(o)

function AxialHeisenberg(
    L...; hx=0.0, hy=0.0, hz=0.0, Jx=0.0, Jy=0.0, Jz=0.0, Kx=0.0, Ky=0.0, Kz=0.0
)
    itr = map(L) do op
        return op isa LindbladTerm ? op : lin(op)
    end
    return AxialHeisenberg(hx, hy, hz, Jx, Jy, Jz, Kx, Ky, Kz, collect(itr))
end

getparams(axp::AxialHeisenberg) = (axp.Jx, axp.Jy, axp.Jz, axp.Kx, axp.Ky, axp.Kz)

function AxialIsing(L...; J=0.0, K=0.0)
    itr = map(L) do op
        return op isa LindbladTerm ? op : lin(op)
    end
    return AxialIsing(J, K, collect(itr))
end

getparams(axp::AxialIsing) = (axp.J, axp.K)

stepper_type(::AxialHeisenberg, _) = AxialStepper
stepper_type(::AbstractMatrix{<:AxialHeisenberg}, _) = AxialStepper

function (model::AxialHeisenberg)(physical, method::AbstractClusterMPO; kwargs...)
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

function MPO(model::AxialHeisenberg)
    linops = model.L

    (Jx, Jy, Jz, Kx, Ky, Kz) = getparams(model)

    Id = Matrix{ComplexF64}(I, 2, 2)

    chi = 0
    for (J, K) in zip((Jx, Jy, Jz), (Kx, Ky, Kz))
        if !iszero(K)
            chi += 4
        elseif !iszero(J)
            chi += 2
        end
    end

    T = BlockMatrix

    A = T(undef, chi, chi)
    B = T(undef, chi, 1)
    C = T(undef, 1, chi)
    D = T(undef, 1, 1)

    nil = kron(zero(Id), zero(Id))

    fill!(A, nil)
    fill!(B, nil)
    fill!(C, nil)
    fill!(D, nil)

    D[1, 1] = nil

    i = 1

    if isempty(linops)
        @info "Is thermal"
        c1 = c2 = -1 / 2
    else
        c1 = -im
        c2 = im
    end

    for (J, K, O) in zip((Jx, Jy, Jz), (Kx, Ky, Kz), PAULI)
        if iszero(J) && iszero(K)
            continue
        end

        if !iszero(J)
            B[i, 1] = J * c1 * kron(O, Id)
            B[i + 1, 1] = J * c2 * kron(Id, transpose(O))
        end

        # Always need these rules
        C[1, i] = kron(O, Id) # ket
        C[1, i + 1] = kron(Id, transpose(O)) # bra

        # C[i, 1] = kron(O, Id) # ket
        # C[i + 1, 1] = kron(Id, transpose(O)) # bra

        if iszero(K)
            str = 2
        else
            str = 4

            A[i, i + 2] = kron(Id, Id)
            A[i + 1, i + 3] = kron(Id, Id)

            B[i + 2, 1] = K * c1 * kron(O, Id)
            B[i + 3, 1] = K * c2 * kron(Id, transpose(O))
        end

        i += str
    end

    foreach(linops) do term
        # Absorb coefficient
        L = Matrix{ComplexF64}(term)

        anticomm = kron(L' * L, one(L)) + kron(one(L), transpose(L' * L))
        D[1, 1] += 1 / 2 * kron(L, transpose(L')) - 1 / 2 * anticomm
    end

    foreach((model.hx, model.hy, model.hz), PAULI) do coeff, op
        # Absorb coefficient
        H = Matrix{ComplexF64}(coeff * op)

        D[1, 1] += 1 / 2 * (c1 * kron(H, one(H)) + c2 * kron(one(H), transpose(H)))
    end

    return MPO(A, B, C, D)
end

function (model::AxialIsing)(physical, method::ClusterPEPO; kwargs...)
    rv =
        timestep -> begin
            tensor = fsmaxialising(Matrix.(model.L), physical, timestep; J=model.J, K=model.K)

            return tensor
        end

    return rv
end

function fsmaxialising(vecL, space, dt; extend=true, J, K)
    T = Dict{NTuple{4,Int},Matrix{ComplexF64}}

    rules = (A=T(), B=T(), C=T(), D=T())

    X, Y, Z = PAULI

    D = kron(zero(Z), zero(Z))

    for L in vecL
        D +=
            kron(L, transpose(L')) -
            1 / 2 * (kron(L' * L, one(L)) + kron(one(L), transpose(L' * L)))
    end

    if !isempty(vecL)
        c1 = -im
        c2 = im
    else
        @info "Is thermal"
        dt = dt / 2
        c1 = -1
        c2 = -1
    end

    rules.A[1, 1, 1, 1] = one(D)
    rules.D[1, 1, 1, 1] = D

    i = 2
    j = 2

    A = Z

    Id = one(A)

    # We need these rules if either term is present, we just use J = 1 if no nearest-
    # neighbour term present
    rules.C[i, 1, 1, 1] = kron(A, Id)
    rules.C[i + 1, 1, 1, 1] = kron(Id, transpose(A))

    rules.C[1, 1, 1, j] = kron(A, Id)
    rules.C[1, 1, 1, j + 1] = kron(Id, transpose(A))

    rules.B[1, j, 1, 1] = c1 * J * kron(A, Id)
    rules.B[1, j + 1, 1, 1] = c2 * J * kron(Id, transpose(A))

    rules.B[1, 1, i, 1] = c1 * J * kron(A, Id)
    rules.B[1, 1, i + 1, 1] = c2 * J * kron(Id, transpose(A))

    rules.A[i + 2, 1, i, 1] = kron(Id, Id)
    rules.A[i + 3, 1, i + 1, 1] = kron(Id, Id)

    rules.B[1, 1, i + 2, 1] = c1 * K * kron(A, Id)
    rules.B[1, 1, i + 3, 1] = c2 * K * kron(Id, transpose(A))

    # The above rules add an extra 4 levels
    i += 4
    j += 2

    @info "TNO bond dimension: $((i-1,j-1))"

    d = space
    di = dim(d)

    data = zeros(ComplexF64, di, di, di, di, i - 1, j - 1, i - 1, j - 1)

    fsmpepo!(data, rules, dt; extend=extend)

    Dx = reduce(oplus, fill(oneunit(d), i - 1))
    Dy = reduce(oplus, fill(oneunit(d), j - 1))

    return TensorMap(data, d * d' * d' * d, Dx * Dy * Dx' * Dy')
end
