const BlockMatrix = Matrix{Matrix{ComplexF64}}

struct MPO
    A::BlockMatrix
    B::BlockMatrix
    C::BlockMatrix
    D::BlockMatrix
    data::Matrix{BlockMatrix}
end

Base.getindex(mpo::MPO, i1, i2) = mpo.data[i1, i2]

function physicalspace(S::Type{<:IndexSpace}, mpo::MPO)
    d = Int(sqrt(size(mpo.D[1, 1], 1)))
    return ⊕(oneunit(S)^d...)
end

function Base.show(io::IO, ::MIME"text/plain", mpo::MPO)
    chi = size(mpo.A, 1)
    return print(io, "MatrixProductOperator with χ = $chi")
end

function MPO(A::T, B::T, C::T, D::T) where {T}
    data = Matrix{T}(undef, 3, 3)

    # row, col
    data[1, 1] = map(one, D)
    data[1, 2] = C
    data[1, 3] = D

    data[2, 1] = map(zero, B)
    data[2, 2] = A
    data[2, 3] = B

    data[3, 1] = map(zero, D)
    data[3, 2] = map(zero, C)
    data[3, 3] = map(one, D)

    return MPO(A, B, C, D, data)
end


function MPO(eqn::LindbladEquation)
    Id = Matrix{ComplexF64}(I, 2, 2)

    chi = 2 * (length(eqn.twolocal) + length(eqn.longrange))
    T = BlockMatrix
    A = T(undef, chi, chi)
    B = T(undef, chi, 1)
    C = T(undef, 1, chi)
    D = T(undef, 1, 1)

    nil = kron(zero(Id), zero(Id))

    fill!(A, nil)
    D[1, 1] = nil

    i = 1

    for term in eqn.twolocal
        X, Y = term.term
        a = term.coeff

        B[i, 1] = -a * im * kron(X, Id)
        B[i + 1, 1] = a * im * kron(Id, transpose(X))

        C[1, i] = kron(Y, Id)
        C[1, i + 1] = kron(Id, transpose(Y))

        i += 2
    end

    for term in eqn.longrange
        X, Y = term.term
        a = term.coeff

        B[i, 1] = -a * im * kron(X, Id)
        B[i + 1, 1] = -a * im * kron(Id, transpose(X))

        C[1, i] = kron(Y, Id)
        C[1, i + 1] = kron(Id, transpose(Y))

        A[i, i] = term.decay * Id
        A[i + 1, i + 1] = term.decay * Id

        i += 2
    end

    # LOCAL TERMS

    D[1, 1] = 1 / 2 * sum(eqn.onelocal; init=D[1, 1]) do term
        H = Matrix{ComplexF64}(term)
        return -im * (kron(H, one(H)) - kron(one(H), transpose(H)))
    end

    D[1, 1] = 1 / 2 * sum(eqn.linops; init=D[1, 1]) do term
        L = Matrix{ComplexF64}(term)
        anticomm = kron(L' * L, one(L)) + kron(one(L), transpose(L' * L))
        return kron(L, transpose(L')) - 1 / 2 * anticomm
    end

    return MPO(A, B, C, D)
end

function dynamicalmpo(eqn::LindbladEquation, dt::Number, order::Integer; kwargs...)
    return dynamicalmpo(MPO(eqn), dt, order; kwargs...)
end

function dynamicalmpo(
    H::MPO,
    dt::Number,
    order::Integer;
    extend=true,
    compress=true,
    approx=true,
)
    N = order

    levels = Vector{Int}[]

    # Generate the levels
    for tup in Iterators.product(fill(1:3, N)...)
        v = collect(reverse(tup))
        push!(levels, v)
    end

    prodlevels = Iterators.product(levels, levels)

    # Raise H to the power of N
    data = map(i -> lazypower(H, i[1], i[2]), prodlevels)

    # Extension step
    if extend
        for (posa, a) in enumerate(levels), (posb, b) in enumerate(levels)
            # There is a typo in the paper; paper reads do if 1 ∈ b; should read
            # do if 1 ∉ b. As such, the following conditional skips if 1 ∈ b.
            if 1 ∈ b
                continue
            end

            if (2 ∉ a && 3 ∈ a)
                continue
            end

            aes = [insertat(a, pos, 1) for pos in 1:(N + 1)]
            bes = [insertat(b, pos, 3) for pos in 1:(N + 1)]

            for ae in aes, be in bes
                n1 = count(isequal(1), ae)
                n3 = count(isequal(3), be)

                coeff = factorial(N) / (factorial(N + 1) * n1 * n3)
                # @debug "Extending with ae=$ae, be=$be into a=$a, b=$b and n1*n3=$(n1 * n3)"

                data[posa, posb] += lazypower(H, ae, be) * dt * coeff
            end
        end
    end

    # Construct the time evolution MPO
    for a in 1:N
        generator = fill(1, N)
        for i in 1:a
            generator[end + 1 - i] = 3
        end

        @assert length(generator) == N

        P = permutations(generator)

        for lev in P
            if lev ∈ levels
                # get the index corresponding to that particular level
                b = findfirst(isequal(lev), levels)

                # add column b to the first column
                data[:, 1] += (dt^a * factorial(N - a) / factorial(N)) * data[:, b]

                # remove row and column b
                data = data[1:end .!= b, 1:end .!= b]
                deleteat!(levels, b)
            end
        end
    end

    if compress
        data = exactcompression!(levels, data)
    end

    if approx
        data = approxcompression!(levels, data, dt)
    end

    return data
end

function exactcompression!(levels, data)
    for (c, lev) in enumerate(levels)
        k1 = map(x -> isequal(x, 1) ? 2 : 1, lev)
        k3 = map(x -> isequal(x, 3) ? 2 : 1, lev)

        sc = findfirst(isequal(lev[sortperm(k1)]), levels)
        sr = findfirst(isequal(lev[sortperm(k3)]), levels)

        n1 = count(isequal(1), lev)
        n3 = count(isequal(3), lev)

        if n3 ≤ n1 && sc != c
            data[sc, :] += data[c, :]

            @debug "Removing $lev"

            data = data[1:end .!= c, 1:end .!= c]
            deleteat!(levels, c)

        elseif n3 > n1 && sr != c
            data[:, sr] += data[:, c]

            @debug "Removing $lev"

            data = data[1:end .!= c, 1:end .!= c]
            deleteat!(levels, c)
        end
    end
    return data
end

function approxcompression!(levels::Vector{Vector{Int}}, data, dt)
    for (ac, alev) in enumerate(levels)
        N = length(alev)

        n3 = count(isequal(3), alev)
        (1 in alev || n3 == 0) && continue

        blev = [isequal(i, 3) ? 1 : i for i in alev]

        bc = findfirst(isequal(blev), levels)

        data[:, bc] += data[:, ac] * dt^n3 * factorial(N - n3) / factorial(N)

        data = data[1:end .!= ac, 1:end .!= ac]
        deleteat!(levels, ac)
    end
    return data
end

function curly(mpo, args...)
    P = collect(permutations(args))
    S = Tuple(Set(P))

    func = term -> kron(1, term...)

    mats = map(term -> map(symb -> getfield(mpo, symb), term), S)

    return sum(func, mats)
end

function lazypower(H::MPO, row, col)
    terms = (H[r, c] for (r, c) in zip(row, col))
    return kron(1, terms...)
end

function insertat(level::Vector, pos::Integer, val)
    rv = copy(level)

    insert!(rv, pos, val)

    return rv
end
