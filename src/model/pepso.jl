function fsmheisenberg(
    vecL,
    space,
    dt;
    extend=true,
    Jx=0.0,
    Jy=0.0,
    Jz=0.0,
    Kx=0.0,
    Ky=0.0,
    Kz=0.0,
    hx=0.0,
    hy=0.0,
    hz=0.0,
)
    T = Dict{NTuple{4,Int},Matrix{ComplexF64}}

    rules = (A=T(), B=T(), C=T(), D=T())

    X, Y, Z = PAULI

    D = -im * hx * (kron(X, one(X)) - kron(one(X), transpose(X)))
    D += -im * hy * (kron(Y, one(Y)) - kron(one(Y), transpose(Y)))
    D += -im * hz * (kron(Z, one(Z)) - kron(one(Z), transpose(Z)))

    for L in vecL
        D +=
            kron(L, transpose(L')) -
            1 / 2 * (kron(L' * L, one(L)) + kron(one(L), transpose(L' * L)))
    end

    Js = (Jx, Jy, Jz)
    Ks = (Kx, Ky, Kz)

    rules.A[1, 1, 1, 1] = one(D)
    rules.D[1, 1, 1, 1] = D

    i = 2
    j = 2

    for (J, K, A) in zip(Js, Ks, PAULI)
        Id = one(A)

        if iszero(J) && iszero(K)
            continue
        end

        rules.C[i, 1, 1, 1] = kron(A, Id)
        rules.C[i + 1, 1, 1, 1] = kron(Id, transpose(A))

        rules.C[1, 1, 1, j] = kron(A, Id)
        rules.C[1, 1, 1, j + 1] = kron(Id, transpose(A))

        if !iszero(J)

            # Nearest neighbour specific
            rules.B[1, j, 1, 1] = -im * J * kron(A, Id)
            rules.B[1, j + 1, 1, 1] = im * J * kron(Id, transpose(A))

            rules.B[1, 1, i, 1] = -im * J * kron(A, Id)
            rules.B[1, 1, i + 1, 1] = im * J * kron(Id, transpose(A))

            # If we don't have any next-nearest term, then we are done; move on
        end

        if iszero(K)
            i += 2
            j += 2
            continue
        end

        # West to north
        rules.A[1, 1, i, j + 2] = kron(Id, Id)
        rules.A[1, 1, i + 1, j + 3] = kron(Id, Id)

        # South to west
        rules.A[1, j, i + 2, 1] = kron(Id, Id)
        rules.A[1, j + 1, i + 3, 1] = kron(Id, Id)

        # Finish from north
        rules.B[1, j + 2, 1, 1] = -im * K * kron(A, Id)
        rules.B[1, j + 3, 1, 1] = im * K * kron(Id, transpose(A))

        # Finish from west
        rules.B[i + 2, 1, 1, 1] = -im * K * kron(A, Id)
        rules.B[i + 3, 1, 1, 1] = im * K * kron(Id, transpose(A))

        # The above rules add an extra 4 levels
        i += 4
        j += 4
    end

    @info "TNO bond dimension: $((i-1,j-1))"

    d = space
    di = dim(d)

    data = zeros(ComplexF64, di, di, di, di, i - 1, j - 1, i - 1, j - 1)

    fsmpepo!(data, rules, dt; extend=extend)

    Dx = reduce(oplus, fill(oneunit(d), i - 1))
    Dy = reduce(oplus, fill(oneunit(d), j - 1))

    return TensorMap(data, d * d' * d' * d, Dx * Dy * Dx' * Dy')
end

function _fsmheisenberg(
    vecL,
    space,
    dt;
    extend=true,
    Jx=0.0,
    Jy=0.0,
    Jz=0.0,
    Kx=0.0,
    Ky=0.0,
    Kz=0.0,
    hx=0.0,
    hy=0.0,
    hz=0.0,
)
    T = Dict{NTuple{4,Int},Matrix{ComplexF64}}

    rules = (A=T(), B=T(), C=T(), D=T())

    X, Y, Z = PAULI

    D = -im * hx * (kron(X, one(X)) - kron(one(X), transpose(X)))
    D += -im * hy * (kron(Y, one(Y)) - kron(one(Y), transpose(Y)))
    D += -im * hz * (kron(Z, one(Z)) - kron(one(Z), transpose(Z)))

    for L in vecL
        D +=
            kron(L, transpose(L')) -
            1 / 2 * (kron(L' * L, one(L)) + kron(one(L), transpose(L' * L)))
    end

    Js = (Jx, Jy, Jz)
    Ks = (Kx, Ky, Kz)

    rules.A[1, 1, 1, 1] = one(D)
    rules.D[1, 1, 1, 1] = D

    i = 2
    j = 2

    for (J, K, A) in zip(Js, Ks, PAULI)
        Id = one(A)

        if !iszero(J)
            # We need these rules if either term is present, we just use J = 1 if no nearest-
            # neighbour term present
            rules.C[i, 1, 1, 1] = -im * J * kron(A, Id)
            rules.C[i + 1, 1, 1, 1] = im * J * kron(Id, transpose(A))

            rules.C[1, 1, 1, j] = -im * J * kron(A, Id)
            rules.C[1, 1, 1, j + 1] = im * J * kron(Id, transpose(A))

            rules.B[1, j, 1, 1] = kron(A, Id)
            rules.B[1, j + 1, 1, 1] = kron(Id, transpose(A))

            rules.B[1, 1, i, 1] = kron(A, Id)
            rules.B[1, 1, i + 1, 1] = kron(Id, transpose(A))

            # If we don't have any next-nearest term, then we are done; move on

            i += 2
            j += 2
        end

        if iszero(K)
            continue
        end

        rules.C[i, 1, 1, 1] = -im * K * kron(A, Id)
        rules.C[i + 1, 1, 1, 1] = im * K * kron(Id, transpose(A))

        # go up
        rules.A[1, 1, i, j] = kron(Id, Id)
        rules.A[1, 1, i + 1, j + 1] = kron(Id, Id)

        # go down
        rules.A[1, j + 2, i, 1] = kron(Id, Id)
        rules.A[1, j + 3, i + 1, 1] = kron(Id, Id)

        rules.B[1, j, 1, 1] = kron(A, Id)
        rules.B[1, j + 1, 1, 1] = kron(Id, transpose(A))

        rules.B[1, 1, 1, j + 2] = kron(A, Id)
        rules.B[1, 1, 1, j + 3] = kron(Id, transpose(A))

        # The above rules add an extra 4 levels
        i += 2
        j += 4
    end

    @info "TNO bond dimension: $((i-1,j-1))"

    d = space
    di = dim(d)

    data = zeros(ComplexF64, di, di, di, di, i - 1, j - 1, i - 1, j - 1)

    fsmpepo!(data, rules, dt; extend=extend)

    Dx = reduce(oplus, fill(oneunit(d), i - 1))
    Dy = reduce(oplus, fill(oneunit(d), j - 1))

    return TensorMap(data, d * d' * d' * d, Dx * Dy * Dx' * Dy')
end
function fsmpepo!(tensor::AbstractArray{<:Number,8}, rules, t::Real; extend=true)
    physdim = map(ind -> size(tensor, ind), (1, 2, 3, 4))

    incr! = (ind, rhs) -> begin
        bondind = CartesianIndex(ind)

        rhsdata = reshape(rhs, physdim)

        for physind in CartesianIndices(rhsdata)
            tensor[physind, bondind] += rhsdata[physind]
        end

        return tensor
    end

    rule_types = (:A, :B, :C, :D)

    if !extend
        for type in rule_types
            if type in (:A, :C)
                pref = 1
            elseif type in (:B, :D)
                pref = t
            end
            for (ind, rule) in getproperty(rules, type)
                incr!(ind, pref * rule)
            end
        end
    else
        for (type1, type2) in Iterators.product(rule_types, rule_types)
            dict1 = getproperty(rules, type1)
            dict2 = getproperty(rules, type2)

            for (kv1, kv2) in Iterators.product(dict1, dict2)
                ind1, rule1 = kv1
                ind2, rule2 = kv2

                any(@. !isone(ind1) && !isone(ind2)) && continue

                ind = ind1 .* ind2

                if type1 in (:A, :C)
                    if type2 in (:A, :C)
                        pref = ind1 == (1, 1, 1, 1) && ind2 == (1, 1, 1, 1) ? 1 : 1 / 2
                    elseif type2 in (:B, :D)
                        pref = t / 2
                    end
                elseif type1 in (:B, :D)
                    if type2 in (:A, :C)
                        pref = t / 2
                    elseif type2 in (:B, :D)
                        pref = t^2 / factorial(2)
                    end
                end

                incr!(ind, pref * rule1 * rule2)
            end
        end
    end

    return tensor
end
