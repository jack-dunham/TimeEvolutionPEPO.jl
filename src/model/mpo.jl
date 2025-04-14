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
    return print(io, "Matrix product operator with χ = $chi")
end

function MPO(A::T, B::T, C::T, D::T) where {T}
    data = Matrix{T}(undef, 3, 3)

    # C should have 1 row; B should have 1 column

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

function MPO(eqn::Union{Hamiltonian, Liouvillian})

    if eqn isa Hamiltonian
        c1 = c2 = -1/2
    elseif eqn isa Liouvillian
        c1 = -im
        c2 = im
    end

    Id = Matrix{ComplexF64}(I, 2, 2)

    chi = 2 * length(eqn)

    T = BlockMatrix
    A = T(undef, chi, chi)
    B = T(undef, chi, 1)
    C = T(undef, 1, chi)
    D = T(undef, 1, 1)

    nil = kron(zero(Id), zero(Id))

    fill!(A, nil)
    D[1, 1] = nil

    i = 1

    for h in terms(eqn)

        a1 = c1 * h.a
        a2 = c2 * h.a

        if h isa ExpOp
            a1 *= h.λ
            a2 *= h.λ

            A[i, i] = h.λ * kron(Id, Id)
            A[i + 1, i + 1] = h.λ * kron(Id, Id)

        elseif h isa LocalOp{2}
            @debug "Adding 2-local term" h.a
            nothing
        else
            throw(ArgumentError("incompatible term of type $(typeof(h)) for method DMPO"))
        end

        @debug a1 a2


        C[1, i] = a1 * kron(h.C, Id)
        C[1, i + 1] = a2 * kron(Id, transpose(h.C))

        B[i, 1] = kron(h.B, Id) # ket
        B[i + 1, 1] = kron(Id, transpose(h.B)) # bra

        i += 2
    end

    # LOCAL TERMS (divide each by 1/2 on assumption of two-dimensional simulation)

    D[1,1] += sum(oneterms(eqn); init=zero(D[1,1])) do term
        # Absorb coefficient
        @debug "Adding local term" term.a
        H = Matrix{ComplexF64}(term)
        return c1 * kron(H, one(H)) + c2 * kron(one(H), transpose(H))
    end

    if eqn isa Liouvillian
        foreach(dissipator(eqn)) do L
            anticomm = kron(L' * L, one(L)) + kron(one(L), transpose(L' * L))
            D[1, 1] += 1 / 2 * (kron(L, transpose(L')) - 1 / 2 * anticomm)
        end
    end

    return MPO(A, B, C, D)
end

function dynamicalmpo(eqn::Model, dt::Number, order::Integer; kwargs...)
    return dynamicalmpo(MPO(eqn), dt, order; kwargs...)
end

function dynamicalmpo(
    H::MPO, dt::Number, order::Integer; extend=true, compress=true, approx=true
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

    rv = kron(1, terms...)

    return rv
end

function testlazypower(H, N)
    X, Y, Z = PAULI

    C1 = H[1, 2][1, 1]
    C2 = H[1, 2][1, 1]

    @assert lazypower(H, [1, 1], [1, 2]) == H[1, 2]
    @assert lazypower(H, [1, 1], [3, 1]) == H[1, 3]
end

function insertat(level::Vector, pos::Integer, val)
    rv = copy(level)

    insert!(rv, pos, val)

    return rv
end

# Unfinsihed...for testing
function secondordermpo(H)
    (_, chi) = size(H.C)

    eta = 4 + 4 * chi + chi^2

    rv = similar(H.A, (eta, eta))

    D = H.D
    Id = one(D)

    i = 1
    rv[1, i] = one(D)
    i += 1

    for i in 1:chi
        rv[1, i] = C[1, i]
    end

    ##
    rv[1, 1] = Id

    rv[1, 2] = C[1, 1]
    rv[1, 3] = C[1, 2]

    rv[1, 4] = D[1, 1]

    rv[1, 5] = C[1, 1]
    rv[1, 6] = C[1, 2]

    rv[1, 7] = C[1, 1] * C[1, 1]
    rv[1, 8] = C[1, 1] * C[1, 2]
    rv[1, 9] = C[1, 2] * C[1, 1]
    rv[1, 10] = C[1, 2] * C[1, 2]

    #(2,3)
    rv[1, 11] = C[1, 1] * D[1, 1]
    rv[1, 12] = C[1, 2] * D[1, 1]

    rv[1, 13] = D[1, 1]

    rv[1, 14] = D[1, 1] * C[1, 1]
    rv[1, 15] = D[1, 1] * C[1, 2]

    rv[1, 16] = D[1, 1] * D[1, 1]
    ##

    rv[2, 2] = A[1, 1]
    rv[2, 3] = A[1, 2]

    rv[2, 4] = B[1, 1]

    rv[2, 7] = C[1, 1] * A[1, 1]
    rv[2, 8] = C[1, 1] * A[1, 2]
    rv[2, 9] = C[1, 2] * A[1, 1]
    rv[2, 10] = C[1, 2] * A[1, 2]

    rv[2, 11] = C[1, 1] * B[1, 1]
    rv[2, 12] = C[1, 2] * B[1, 1]

    rv[2, 14] = D[1, 1] * A[1, 1]
    rv[2, 15] = D[1, 1] * A[1, 2]

    rv[2, 16] = D[1, 1] * B[1, 1]

    # 
    rv[3, 2] = A[2, 1]
    rv[3, 3] = A[2, 2]

    rv[3, 4] = B[2, 1]

    rv[3, 7] = C[1, 1] * A[2, 1]
    rv[3, 8] = C[1, 1] * A[2, 2]
    rv[3, 9] = C[1, 2] * A[2, 1]
    rv[3, 10] = C[1, 2] * A[2, 2]

    rv[3, 11] = C[1, 1] * B[2, 1]
    rv[3, 12] = C[1, 2] * B[2, 1]

    rv[3, 14] = D[1, 1] * A[2, 1]
    rv[3, 15] = D[1, 1] * A[2, 2]

    rv[3, 16] = D[1, 1] * B[2, 1]
    #
    rv[4, 11] = C[1, 1]
    rv[4, 12] = C[1, 2]

    rv[4, 16] = D[1, 1]
    #
    rv[5, 1] = Id

    rv[5, 2] = C[1, 1]
    rv[5, 3] = C[1, 2]

    rv[5, 4] = D[1, 1]

    rv[5, 5] = C[1, 1]
    rv[5, 6] = C[1, 2]

    rv[5, 7] = C[1, 1] * C[1, 1]
    rv[5, 8] = C[1, 1] * C[1, 2]
    rv[5, 9] = C[1, 2] * C[1, 1]
    rv[5, 10] = C[1, 2] * C[1, 2]

    #(2,3)
    rv[5, 11] = C[1, 1] * D[1, 1]
    rv[5, 12] = C[1, 2] * D[1, 1]

    rv[5, 13] = D[1, 1]

    rv[5, 14] = D[1, 1] * C[1, 1]
    rv[5, 15] = D[1, 1] * C[1, 2]

    return rv[5, 16] = D[1, 1] * D[1, 1]
end

function symmetryicmpo(h::TensorMap{T,S,2,2}) where {T,S}
    U, s, V = tsvd(h, (1, 3), (2, 4))

    C = U * sqrt(s)  # 0 -> eta
    B = sqrt(s) * V  # eta -> 0

    η = dim(domain(s))

    dst = zeros(T, 2, 2, η + 2, η + 2)

    dst[:, :, 1, 2:(end - 1)] .= convert(Array, C)
    dst[:, :, 2:(end - 1), 1] .= permutedims(convert(Array, B), (2, 3, 1))

    dst[1, 1, 1, 1] = 1
    dst[2, 2, 1, 1] = 1

    dst[1, 1, η, η] = 1
    dst[2, 2, η, η] = 1

    sp = oneunit(S) ⊕ domain(C, 1) ⊕ oneunit(S)

    tdst = TensorMap(dst, codomain(h, 1) * domain(h, 1)', sp' * sp)

    return dst == convert(Array, tdst)
end

function bhm(; natom=1, nfluc=1, U, V, t)
    # Define the charge sectors.
    charges = (-nfluc):1:nfluc

    # Construct the U(1) symmetric vector space.
    s = U1Space(q => 1 for q in charges)

    # Get the charge sectors; TensorKit.jl defines a particular order of the basis states
    # so we must respect this. This vector tells us that order.
    sectors = blocksectors(s)

    # The dimension of the local vector space.
    d = dim(s)

    # Initialise some empty matrices.
    ladderop = zeros(ComplexF64, d, d)
    numberop = zeros(ComplexF64, d, d)

    for q in charges[1:end]
        # Get the position in the basis of the states with charge b and k
        b = findfirst(isequal(U1Irrep(q)), sectors)
        k = findfirst(isequal(U1Irrep(q - 1)), sectors)

        # We can't annihilate below the cut-off, so that matrix element remains set to 0.
        if q > -nfluc
            ladderop[k, b] = sqrt(q + natom)
        end

        numberop[b, b] = q + natom
    end

    # Just for convenience.
    a = ladderop
    n = numberop

    id = one(n)

    # Two local terms...
    eqn = V * ham(n, n) + -t * ham(a', a) + -t * ham(a, a') + U * ham(n * (n - id))

    # Define the Hamiltonian as a tensor map. See above for reasoning on the 1/4.
    # h = TensorMap(tl + 1 / 4 * ol, s * s, s * s)

    # As our ladder and number operators have an unusual basis ordering, we also return 
    # these for convenience.

    return eqn, s, a, n
end
