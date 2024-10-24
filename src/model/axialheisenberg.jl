struct AxialHeisenberg{Ax} <: AbstractLindbladEquation
    Jx::Float64
    Jy::Float64
    Jz::Float64
    Kx::Float64
    Ky::Float64
    Kz::Float64
    L::Vector{LindbladTerm}
end

AxialHeisenberg(L...; kwargs...) = AxialHeisenberg{(1, 2)}(L...; kwargs...)

Base.broadcastable(o::AxialHeisenberg) = Ref(o)

haxis(axp::AxialHeisenberg) = AxialHeisenberg{1}(getparams(axp)..., axp.L)
vaxis(axp::AxialHeisenberg) = AxialHeisenberg{2}(getparams(axp)..., axp.L)

function AxialHeisenberg{N}(L...; Jx=0.0, Jy=0.0, Jz=0.0, Kx=0.0, Ky=0.0, Kz=0.0) where {N}
    itr = map(L) do op
        return op isa LindbladTerm ? op : lin(op)
    end
    return AxialHeisenberg{N}(Jx, Jy, Jz, Kx, Ky, Kz, collect(itr))
end

getparams(axp::AxialHeisenberg) = (axp.Jx, axp.Jy, axp.Jz, axp.Kx, axp.Ky, axp.Kz)

stepper_type(::AxialHeisenberg, _) = AxialStepper
stepper_type(::AbstractMatrix{<:AxialHeisenberg}, _) = AxialStepper

function (model::AxialHeisenberg{N})(physical, method::DMPO) where {N}
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


            B[i + 2, i] = - K * im * kron(O, Id)
            B[i + 3, i] = K * im * kron(Id, transpose(O))
        end

        i += str
    end

    D[1, 1] = 1/2 * sum(linops; init=D[1, 1]) do term
        # Absorb coefficient
        L = Matrix{ComplexF64}(term)

        anticomm = kron(L' * L, one(L)) + kron(one(L), transpose(L' * L))
        return kron(L, transpose(L')) - 1 / 2 * anticomm
    end

    return MPO(A, B, C, D)
end
