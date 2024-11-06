struct AxialHeisenberg <: AbstractLindbladEquation
    Jx::Float64
    Jy::Float64
    Jz::Float64
    Kx::Float64
    Ky::Float64
    Kz::Float64
    L::Vector{LindbladTerm}
end

struct AxialIsing <: AbstractLindbladEquation
    J::Float64
    K::Float64
    L::Vector{LindbladTerm}
end

Base.broadcastable(o::AxialHeisenberg) = Ref(o)

function AxialHeisenberg(L...; Jx=0.0, Jy=0.0, Jz=0.0, Kx=0.0, Ky=0.0, Kz=0.0)
    itr = map(L) do op
        return op isa LindbladTerm ? op : lin(op)
    end
    return AxialHeisenberg(Jx, Jy, Jz, Kx, Ky, Kz, collect(itr))
end

getparams(axp::AxialHeisenberg) = (axp.Jx, axp.Jy, axp.Jz, axp.Kx, axp.Ky, axp.Kz)

function AxialIsing(L...; J=0.0, K = 0.0)
    itr = map(L) do op
        return op isa LindbladTerm ? op : lin(op)
    end
    return AxialIsing(J, K, collect(itr))
end

getparams(axp::AxialIsing) = (axp.J, axp.K)

stepper_type(::AxialHeisenberg, _) = AxialStepper
stepper_type(::AbstractMatrix{<:AxialHeisenberg}, _) = AxialStepper

function (model::AxialHeisenberg)(physical, method::AbstractDMPO; kwargs...)
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


function MPO(model::AxialHeisenberg)
    linops = model.L

    (Jx, Jy, Jz, Kx, Ky, Kz) = getparams(model)

    Id = Matrix{ComplexF64}(I, 2, 2)

    chi = 2 * count(!iszero, getparams(model))

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

    for (J, K, O) in zip((Jx, Jy, Jz), (Kx, Ky, Kz), PAULI)
        if iszero(J) && iszero(K)
            continue
        end

        if iszero(J)
            J = 1
        else
            B[i, 1] = -J * im * kron(O, Id)
            B[i + 1, 1] = J * im * kron(Id, transpose(O))
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

            B[i + 2, i] = -K * im * kron(O, Id)
            B[i + 3, i] = K * im * kron(Id, transpose(O))
        end

        i += str
    end

    D[1, 1] = 1 / 2 * sum(linops; init=D[1, 1]) do term
        # Absorb coefficient
        L = Matrix{ComplexF64}(term)

        anticomm = kron(L' * L, one(L)) + kron(one(L), transpose(L' * L))
        return kron(L, transpose(L')) - 1 / 2 * anticomm
    end

    return MPO(A, B, C, D)
end

function (model::AxialIsing)(physical, method::TEPEPSO; kwargs...)
    rv =
        timestep -> begin

            tensor = fsmaxialising(Matrix.(model.L), physical, timestep; J = model.J, K = model.K)

            return tensor
        end

    return rv
end

function fsmaxialising(
    vecL,
    space,
    dt;
    extend=true,
    J,
    K,
)
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

    rules.C[1, 1, 1, j] =  kron(A, Id)
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
