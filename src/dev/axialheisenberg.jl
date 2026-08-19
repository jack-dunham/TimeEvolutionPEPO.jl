# TODO: Deprec
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
