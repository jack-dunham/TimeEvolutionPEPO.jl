# TODO: This need deprec, but may still be useful now we have symmetric loopback
function fsmsymmetric(
    op,
    localops,
    linops,
    space,
    dt,
    w,
    mks,
    lks;
    J=1.0,
    localscaling=1.0,
    extend=true,
    thermal=false,
)
    Id = one(op)

    T = Dict{NTuple{4,Int},Matrix{ComplexF64}}

    rules = (A=T(), B=T(), C=T(), D=T())

    X, Y, Z = PAULI

    D = kron(zero(Id), zero(Id))

    for A in localops
        if thermal
            D += -0.5 * (kron(A, Id) + kron(Id, transpose(A)))
        else
            D += -im * (kron(A, Id) - kron(Id, transpose(A)))
        end
    end

    for L in linops
        acomm = (kron(L' * L, one(L)) + kron(one(L), transpose(L' * L)))
        D += kron(L, transpose(L')) - 1 / 2 * acomm
    end

    D = D / localscaling

    rules.A[1, 1, 1, 1] = one(D)
    rules.D[1, 1, 1, 1] = D

    if thermal
        j1 = j2 = -J / 2
    else
        j1 = -im * J
        j2 = im * J
    end

    C1 = kron(op, Id)
    B1 = kron(op, Id)

    C2 = kron(Id, transpose(op))
    B2 = kron(Id, transpose(op))

    # Get strides

    hstr = 1
    vstr = 1

    for (C, B, j) in zip((C1, C2), (B1, B2), (j1, j2))
        Id = kron(one(op), one(op))
        gapprox = zip(mks, lks)

        for (n1, (m1, l1)) in enumerate(gapprox)
            # mklk = mk * lk 

            c = hstr + 2 * n1 - 1
            u = hstr + 2 * n1

            rules.C[c, 1, 1, 1] = 0.5 * j * w * sqrt(m1 * l1) * C
            rules.C[1, c, 1, 1] = 0.5 * j * w * sqrt(m1 * l1) * C

            rules.C[1, 1, u, 1] = 0.5 * j * w * sqrt(m1 * l1) * C
            rules.C[1, 1, 1, u] = 0.5 * j * w * sqrt(m1 * l1) * C

            rules.B[1, 1, c, 1] = sqrt(m1 * l1) * B
            rules.B[1, 1, 1, c] = sqrt(m1 * l1) * B

            rules.B[u, 1, 1, 1] = sqrt(m1 * l1) * B
            rules.B[1, u, 1, 1] = sqrt(m1 * l1) * B

            # Continue going east
            rules.A[c, 1, c, 1] = l1 * Id
            rules.A[1, c, 1, c] = l1 * Id
            # Continue going west (unconventional)
            rules.A[u, 1, u, 1] = l1 * Id
            rules.A[1, u, 1, u] = l1 * Id

            for (n2, (m2, l2)) in enumerate(gapprox)
                c1 = c
                u1 = u

                c2 = hstr + 2 * n2 - 1
                u2 = hstr + 2 * n2

                rules.A[1, 1, c1, c2] = sqrt(m1 * m2 * l1 * l2) * Id
                rules.A[u1, u2, 1, 1] = sqrt(m1 * m2 * l1 * l2) * Id

                rules.A[u1, 1, 1, c2] = sqrt(m1 * m2 * l1 * l2) * Id
                rules.A[1, u1, c2, 1] = sqrt(m1 * m2 * l1 * l2) * Id
            end
        end

        hstr += 2 * length(gapprox)
        vstr += 2 * length(gapprox)
    end

    @info "TNO bond dimension: ($hstr,$vstr)"

    d = space
    di = dim(d)

    Dx = reduce(oplus, fill(oneunit(d), hstr))
    Dy = reduce(oplus, fill(oneunit(d), vstr))

    data = zeros(ComplexF64, di, di, di, di, hstr, vstr, hstr, vstr)

    fsmpepo!(data, rules, dt; extend=extend)

    pepo = TensorMap(data, d * d' * d' * d, Dx * Dy * Dx' * Dy')

    # println(pepo ≈  permutedom(pepo, (2,1,4,3)))

    return pepo
end
function fsmaxial(
    vecL, space, dt; extend=true, Jx=0.0, Jy=0.0, Jz=0.0, ax=0.0, ay=0.0, az=0.0
)
    T = Dict{NTuple{4,Int},Matrix{ComplexF64}}

    rules = (A=T(), B=T(), C=T(), D=T())

    X, Y, Z = PAULI

    if isempty(length(vecL))
        L = kron(zero(X), zero(X))
        thermal = true
        dt = dt / 2
    end

    D = mapreduce(+, vecL) do L
        return kron(L, transpose(L')) -
               1 / 2 * (kron(L' * L, one(L)) + kron(one(L), transpose(L' * L)))
    end

    Js = (Jx, Jy, Jz)
    as = (ax, ay, az)

    rules.A[1, 1, 1, 1] = one(D)
    rules.D[1, 1, 1, 1] = D

    i = 2
    j = 2

    for (J, a, A) in zip(Js, as, PAULI)
        Id = one(A)

        if !iszero(J)
            # We need these rules if either term is present, we just use J = 1 if no nearest-
            # neighbour term present

            if thermal
                c1 = sqrt(-J)
                c2 = sqrt(-J)
            end
            c1 = sqrt(-im * J)
            c2 = sqrt(im * J)
        else
            rules.C[i, 1, 1, 1] = c1 * kron(A, Id)
            rules.C[i + 1, 1, 1, 1] = c2 * kron(Id, transpose(A))

            rules.C[1, 1, 1, j] = c1 * kron(A, Id)
            rules.C[1, 1, 1, j + 1] = c2 * kron(Id, transpose(A))

            rules.A[i, 1, i, 1] = a * kron(Id, Id)
            rules.A[i + 1, i + 1, 1, 1] = a * kron(Id, Id)

            rules.A[1, j, 1, j] = a * kron(Id, Id)
            rules.A[1, j + 1, 1, j + 1] = a * kron(Id, Id)

            rules.B[1, j, 1, 1] = c1 * kron(A, Id)
            rules.B[1, j + 1, 1, 1] = c2 * kron(Id, transpose(A))

            rules.B[1, 1, i, 1] = c1 * kron(A, Id)
            rules.B[1, 1, i + 1, 1] = c2 * kron(Id, transpose(A))

            # If we don't have any next-nearest term, then we are done; move on

            i += 2
            j += 2
        end
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

function twodapprox(alpha, n, ngmax=2; V=1.0, N=50, T=ComplexF64, gtol=1e-10, force=false)
    wks, lam = approx_as_sum_of_exp(T, alpha / 2, n; N=N, V=V)

    pks = -log.(lam)

    mks = Vector{ComplexF64}[]
    lks = Vector{ComplexF64}[]

    for (wk, pk) in zip(wks, pks)
        f = r -> exp(-pk * r^2)
        errf = (m, l) -> maximum(abs.((x -> sum(@. m * l^x)).(1:N) - f.(1:N)))

        ng0 = force ? ngmax : 1

        mkis, lkis = generalised_approx_as_sum_of_exp(f, ng0; N=N, T=T)

        errfinal = errf(mkis, lkis)

        ngfinal = 1

        for ng in (ng0 + 1):ngmax
            errfinal < gtol && break

            mkisnew, lkisnew = generalised_approx_as_sum_of_exp(f, ng; N=N, T=T)

            errnew = errf(mkisnew, lkisnew)

            if errnew < errfinal
                errfinal = errnew
                mkis = mkisnew
                lkis = lkisnew
                ngfinal = ng
            end
        end

        println("Error for $(real(pk)) with $ngfinal terms: ", errfinal)

        push!(mks, mkis)
        push!(lks, lkis)
    end

    return wks, mks, lks
end

function approx_as_sum_of_exp(alpha, n; kwargs...)
    return approx_as_sum_of_exp(Float64, alpha, n; kwargs...)
end
function approx_as_sum_of_exp(T, α, n; N=50, offset=0, V=1.0)
    f = N -> (V * (N + offset))^float(-α)
    F = zeros(T, N - n + 1, n)

    for r in axes(F, 1)
        for c in axes(F, 2)
            val = f(r + c - 1)
            F[r, c] = val
        end
    end

    QR = qr(F)
    U = QR.Q
    # V = QR.R

    U1 = U[1:(N - n), 1:n]
    U2 = U[2:end, 1:n]

    Λ = eigen(pinv(U1) * U2).values

    A = zeros(T, N, n)
    for c in axes(A, 1)
        A[c, :] .= Λ .^ (c)
    end

    invsol = [f(k) for k in 1:N]

    sol = pinv(A) * invsol

    # return k -> (A * sol)[k]
    return sol, Λ
end

function generalised_approx_as_sum_of_exp(f, n; N=10, T=ComplexF64)
    F = zeros(T, N - n + 1, n)

    for c in axes(F, 2)
        for r in axes(F, 1)
            p = r + c - 1

            val = f(p)
            F[r, c] = val
        end
    end

    QR = qr(F)
    U = QR.Q
    # V = QR.R

    U1 = U[1:(N - n), 1:n]
    U2 = U[2:end, 1:n]

    Λ = eigen(pinv(U1) * U2).values

    A = zeros(T, N, n)
    for c in axes(A, 1)
        A[c, :] .= Λ .^ (c)
    end

    invsol = [f(k) for k in 1:N]

    sol = pinv(A) * invsol

    # return k -> (A * sol)[k]
    return sol, Λ
end


function mfj1j2(Jz,K)
    sx = 0
    sy = 0

    dt = 0.025
    t = 0

    X = Float64[0.0]
    Y = Float64[0.0]

    for _ in 1:300

        t += dt

        dsx = -2 * Jz *  exp(-1/2 * t) * (1 + K) * sy - (sx + 1)
        dsy = +2 * Jz *  exp(-1/2 * t) * (1 + K) * sx - 1/2 * sy

        sx += dsx * dt
        sy += dsy * dt

        push!(X, sx)
        push!(Y, sy)
    end

    return X,Y
end

"""
    $(TYPEDSIGNATURES)

J1-J2 Heisenberg model: nearest-neighbour couplings `Jx`, `Jy`, `Jz`, diagonal next-nearest
couplings `Kx`, `Ky`, `Kz`, on-site fields `hx`, `hy`, `hz`, and on-site Lindblad `jumps`. Returns a
`Hamiltonian` when `jumps` is empty and a `Liouvillian` otherwise.

Each component is one [`J1J2SymmetricOp`](@ref), which carries both ranges on a single pair of
finite-state-machine levels per axis.
"""
function j1j2model(
        jumps = Matrix{ComplexF64}[];
        Jx = 0, Jy = 0, Jz = 0, Kx = 0, Ky = 0, Kz = 0, hx = 0, hy = 0, hz = 0
    )
    model = zero(Hamiltonian)

    for (J, K, A) in zip((Jx, Jy, Jz), (Kx, Ky, Kz), PAULI)
        (iszero(J) && iszero(K)) && continue
        model += J1J2SymmetricOp(float(J), float(K), Matrix{ComplexF64}(A))
    end
    for (h, A) in zip((hx, hy, hz), PAULI)
        iszero(h) && continue
        model += h * LocalOp(Matrix{ComplexF64}(A))
    end

    return isempty(jumps) ? model : model + Dissipator(jumps)
end

"""
    $(TYPEDSIGNATURES)

Propagator of a [`j1j2model`](@ref) over one time step, as an `InfinitePEPSO`. The keywords are
those of `j1j2model`.
"""
function fsmheisenberg(
        jumps, P::IndexSpace, dt::Number; unitcell = (1, 1), extend = true, kwargs...
    )
    return fsmpepso(j1j2model(jumps; kwargs...), P, dt; unitcell, extend)
end
