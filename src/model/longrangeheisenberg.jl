struct LongRangeHeisenberg <: AbstractLindbladEquation
    Jx::Float64
    Jy::Float64
    Jz::Float64
    ax::Float64
    ay::Float64
    az::Float64
    L::Vector{LindbladTerm}
    nk::Int
    N::Int
end

struct AxialLongRangeHeisenberg <: AbstractModel
    Jx::Float64
    Jy::Float64
    Jz::Float64
    ax::Float64
    ay::Float64
    az::Float64
    L::Vector{LindbladTerm}
end

UpdateStyle(::Type{LongRangeHeisenberg}) = IsGlobal()
UpdateStyle(::Type{AxialLongRangeHeisenberg}) = IsGlobal()

getparams(lr::LongRangeHeisenberg) = (lr.Jx, lr.Jy, lr.Jz, lr.ax, lr.ay, lr.az)

stepper_type(::AbstractMatrix{<:LongRangeHeisenberg}, _) = FullStepper

function LongRangeHeisenberg(
    L...; nk=6, N=200, Jx=0.0, Jy=0.0, Jz=0.0, ax=Inf, ay=Inf, az=Inf
)
    linops = map(L) do op
        return op isa LindbladTerm ? op : lin(op)
    end

    return LongRangeHeisenberg(Jx, Jy, Jz, ax, ay, az, collect(linops), nk, N)
end
function AxialLongRangeHeisenberg(L...; Jx=0.0, Jy=0.0, Jz=0.0, ax=0, ay=0, az=0)
    linops = map(L) do op
        return op isa LindbladTerm ? op : lin(op)
    end

    return AxialLongRangeHeisenberg(Jx, Jy, Jz, ax, ay, az, collect(linops))
end

function (model::LongRangeHeisenberg)(physical, method::TEPEPSO)
    (Jx, Jy, Jz, ax, ay, az) = getparams(model)

    rv =
        timestep -> begin
            fsmlongrange(
                Matrix.(model.L),
                physical,
                timestep;
                extend=method.extend,
                Jx=Jx,
                Jy=Jy,
                Jz=Jz,
                ax=ax,
                ay=ay,
                az=az,
                n=model.nk,
                N=model.N,
            )
        end

    return rv
end

function (model::AxialLongRangeHeisenberg)(physical, method::TEPEPSO; kwargs...)
    (Jx, Jy, Jz, ax, ay, az) = getparams(model)

    rv =
        timestep -> begin
            fsmaxial(
                Matrix.(model.L),
                physical,
                timestep;
                extend=method.extend,
                Jx=Jx,
                Jy=Jy,
                Jz=Jz,
                ax=ax,
                ay=ay,
                az=az,
            )
        end

    return rv
end

function fsmlongrange(
    linops,
    space,
    dt;
    extend=true,
    n=5,
    N=50,
    Jx=0.0,
    Jy=0.0,
    Jz=1.0,
    ax=6.0,
    ay=6.0,
    az=6.0,
)
    T = Dict{NTuple{4,Int},Matrix{ComplexF64}}

    rules = (A=T(), B=T(), C=T(), D=T())

    D = mapreduce(+, linops) do L
        acomm = kron(L' * L, one(L)) + kron(one(L), transpose(L' * L))
        return kron(L, transpose(L')) - 1 / 2 * acomm
    end

    Js = (Jx, Jy, Jz)
    as = (ax, ay, az)

    rules.A[1, 1, 1, 1] = one(D)
    rules.D[1, 1, 1, 1] = D

    Js = (Jx, Jy, Jz)

    terms = []
    term_wks = []
    term_mks = []
    term_lks = []

    for (op, J, a) in zip(PAULI, Js, as)
        Id = one(op)

        iszero(J) && continue

        wks, mks, lks = twodapprox(a, n; N=N)

        push!(terms, (-im * J * kron(op, Id), kron(op, Id)))
        push!(term_wks, wks)
        push!(term_mks, mks)
        push!(term_lks, lks)

        push!(terms, (im * J * kron(Id, transpose(op)), kron(Id, transpose(op))))
        push!(term_wks, wks)
        push!(term_mks, mks)
        push!(term_lks, lks)
    end

    # Get strides

    hstr = 1
    vstr = 1

    for (mu, term) in enumerate(terms)
        C, B = term

        Id = one(C)

        itr = zip(term_wks[mu], term_mks[mu], term_lks[mu])

        for (wk, mks, lks) in itr
            gapprox = zip(mks, lks)

            for (nh, (mk, lk)) in enumerate(gapprox)
                # mklk = mk * lk 

                mklk = mk * lk

                c = hstr + 2 * nh - 1
                u = hstr + 2 * nh

                # Start going east
                rules.C[c, 1, 1, 1] = wk * mklk * C

                # Finish going east
                rules.B[1, 1, c, 1] = B
                # Finish going west (unconventional)
                rules.B[u, 1, 1, 1] = B

                # Continue going east
                rules.A[c, 1, c, 1] = lk * Id
                # Continue going west (unconventional)
                rules.A[u, 1, u, 1] = lk * Id

                # Direction change rules
                for nv in 1:length(gapprox)
                    v = vstr + nv

                    # Change from going north to going east
                    rules.A[c, v, 1, 1] = mklk * Id
                    # Change from going north to going west (unconventional)
                    rules.A[1, v, u, 1] = mklk * Id
                end
            end
            for (nv, (mk, lk)) in enumerate(gapprox)
                v = vstr + nv
                # Start going north
                rules.C[1, 1, 1, v] = wk * mk * lk * C

                # Finish going north
                rules.B[1, v, 1, 1] = B

                # Continue going north
                rules.A[1, v, 1, v] = lk * Id
            end

            hstr += 2 * length(gapprox)
            vstr += length(gapprox)
        end
    end

    @info "TNO bond dimension: ($hstr,$vstr)"

    d = space
    di = dim(d)

    Dx = reduce(oplus, fill(oneunit(d), hstr))
    Dy = reduce(oplus, fill(oneunit(d), vstr))

    data = zeros(ComplexF64, di, di', di', di, hstr, vstr, hstr, vstr)

    fsmpepo!(data, rules, dt; extend=extend)

    return TensorMap(data, d * d' * d' * d, Dx * Dy * Dx' * Dy')
end

struct ExponentialBasisFit{T}
    w::Vector{T}
    mk::Vector{Vector{T}}
    lk::Vector{Vector{T}}
end

function (fit::ExponentialBasisFit)(x,y)
    wks = fit.w
    mks = fit.mk
    lks = fit.lk

    if iszero(x) && iszero(y)
        return NaN
    end

    cumsum = 0

    for (wk, mkis, lkis) in zip(wks, mks, lks)

        itr = zip(mkis, lkis)

        for (mki, lki) in itr
            if iszero(x)
                cumsum += wk * mki * lki^y
            elseif iszero(y)
                cumsum += wk * mki * lki^x
            else
                for (mkj, lkj) in itr
                    cumsum += wk * mki * lki^x * mkj * lkj^y
                end
            end
        end
    end

    return cumsum
end

function fsmlongrangeterm(
    op,
    localops,
    linops,
    space, 
    dt; 
    # Interaction strength parameters
    alpha=Inf,
    kmax = 1,
    ngmax = 1,
    N=200,
    T= ComplexF64,
    gtol=1e-8,
    # Model parameters
    hx = 0.0,
    hy = 0.0,
    hz = 0.0,
    J = 0.0,
    # FSM parameters
    extend = true,
    thermal = false,
)
    if alpha == Inf
        wk = zeros(T, kmax)
        wk[1] = eps()^(-2)
        mks = fill(T[eps()], kmax)
        lks = fill(T[eps()], kmax)
    else
        wk, mks, lks = TimeEvolutionPEPO.twodapprox(alpha, kmax, ngmax; V=1.0, N=N, T=T, gtol=gtol)
    end

    println(wk)

    X, Y, Z = PAULI

    rv = []

    for (w, mk, lk) in zip(wk, mks, lks)
        term = fsmgeneral(
            op,
            localops,
            linops,
            space,
            dt,
            w,
            mk,
            lk;
            J=J,
            localscaling = length(wk),
            extend = extend,
            thermal = thermal,
        )
        push!(rv, term)
    end

    return rv, ExponentialBasisFit(wk, mks, lks)
end

function fsm_long_range_term(
    op, space, dt, wk, mks, lks; numterm, extend=true, J=1.0, g, gp, delta, omega
)
    T = Dict{NTuple{4,Int},Matrix{ComplexF64}}

    rules = (A=T(), B=T(), C=T(), D=T())

    L = LADDER[1]

    X, Y, Z = PAULI

    D =
        g * (
            kron(L, transpose(L')) -
            1 / 2 * (kron(L' * L, one(L)) + kron(one(L), transpose(L' * L)))
        )
    D +=
        gp * (
            kron(Z, transpose(Z')) -
            1 / 2 * (kron(Z' * Z, one(Z)) + kron(one(Z), transpose(Z' * Z)))
        )

    NZ = 0.5 * (one(Z) + Z)
    # NZ = Z

    D += -im * 1 / 2 * omega * (kron(X, one(X)) - kron(one(X), transpose(X)))
    # D += -im * 1 / 2 * (-delta) * (kron(Z, one(Z)) - kron(one(Z), transpose(Z)))
    D += -im * 1 / 2 * (-delta) * (kron(NZ, one(NZ)) - kron(one(NZ), transpose(NZ)))

    D = D / numterm

    rules.A[1, 1, 1, 1] = one(D)
    rules.D[1, 1, 1, 1] = D

    Id = one(op)

    C1 = -im * J * kron(op, Id)
    B1 = kron(op, Id)

    C2 = im * J * kron(Id, transpose(op))
    B2 = kron(Id, transpose(op))

    # Get strides

    hstr = 1
    vstr = 1

    for (C, B) in zip((C1, C2), (B1, B2))
        Id = kron(one(op), one(op))
        gapprox = zip(mks, lks)

        for (nh, (mk, lk)) in enumerate(gapprox)
            # mklk = mk * lk 

            c = hstr + 2 * nh - 1
            u = hstr + 2 * nh

            # Start going east
            rules.C[c, 1, 1, 1] = wk * mk * lk * C

            # Finish going east
            rules.B[1, 1, c, 1] = B
            # Finish going west (unconventional)
            rules.B[u, 1, 1, 1] = B

            # Continue going east
            rules.A[c, 1, c, 1] = lk * Id
            # Continue going west (unconventional)
            rules.A[u, 1, u, 1] = lk * Id

            # Direction change rules
            for nv in 1:length(gapprox)
                v = vstr + nv

                # Change from going north to going east
                rules.A[c, v, 1, 1] = mk * lk * Id
                # Change from going north to going west (unconventional)
                rules.A[1, v, u, 1] = mk * lk * Id
            end
        end
        for (nv, (mk, lk)) in enumerate(gapprox)
            v = vstr + nv
            # Start going north
            rules.C[1, 1, 1, v] = wk * mk * lk * C

            # Finish going north
            rules.B[1, v, 1, 1] = B

            # Continue going north
            rules.A[1, v, 1, v] = lk * Id
        end

        hstr += 2 * length(gapprox)
        vstr += length(gapprox)
    end

    @info "TNO bond dimension: ($hstr,$vstr)"

    d = space
    di = dim(d)

    Dx = reduce(oplus, fill(oneunit(d), hstr))
    Dy = reduce(oplus, fill(oneunit(d), vstr))

    data = zeros(ComplexF64, di, di, di, di, hstr, vstr, hstr, vstr)

    fsmpepo!(data, rules, dt; extend=extend)

    pepo = TensorMap(data, d * d' * d' * d, Dx * Dy * Dx' * Dy')

    return pepo
end

function fsmaxy(space, dt; J=0.5, alpha=3, kmax=15, ngmax=3, tol=eps(), gtol=1e-10, T=Float64, hx=0.0, hy=0.0, hz=0.0)
    wk_, mks_, lks_ = TimeEvolutionPEPO.twodapprox(alpha, kmax, ngmax;gtol=gtol, N=200, T=T)

    numremoved = 0

    wk = empty(wk_)
    mks = empty(mks_)
    lks = empty(lks_)

    for i in eachindex(wk_)

        if abs(wk_[i]) < tol
            numremoved += 1
            continue
        end

        push!(wk, wk_[i])
        push!(mks, mks_[i])
        push!(lks, lks_[i])
    end

    println(wk)

    @info "Removed $numremoved terms from expansion"

    X, Y, _ = PAULI

    rv = []

    linops = [LADDER[1]]

    scale = length(wk) * 2

    for (op, j) in zip((X, Y), (J, -J))
        for (w, mk, lk) in zip(wk, mks, lks)
            term = fsmgeneral(op, [],linops, space, dt, w, mk, lk; localscaling=scale, J=j)
            push!(rv, term)
        end
    end

    return rv
end

function fsmlongrangeterm2(space, dt; 
                           J=1.0, 
                           alpha=3, kmax=10, ngmax=2, force=false, tol=eps(), T=Float64, hx=0.0, hy=0.0, hz=0.0, N=200)
    wk_, mks_, lks_ = TimeEvolutionPEPO.twodapprox(alpha, kmax, ngmax; N=N, T=T, force)

    numremoved = 0

    wk = empty(wk_)
    mks = empty(mks_)
    lks = empty(lks_)

    for i in eachindex(wk_)

        if abs(wk_[i]) < tol
            numremoved += 1
            continue
        end

        push!(wk, wk_[i])
        push!(mks, mks_[i])
        push!(lks, lks_[i])
    end

    println(wk)

    @info "Removed $numremoved terms from expansion"

    linops = [LADDER[1]]

    op = PAULI[3]

    rv = []

    scale = 0

    for mk in mks
        scale += length(mk)^2
    end

    println(scale)

    for (w, mk, lk) in zip(wk, mks, lks)
        println(w)
        println(w * mk .* lk)
        for ((mkx, mky), (lkx, lky)) in zip(Iterators.product(mk, mk), Iterators.product(lk, lk))
            # println("(1,0): ", w * mkx * lkx)
            # println(w * mkx * lkx * mky * lky)
            # println(w * mkx * lkx^2 * mky * lky)
            term = fsmbigeneral(op, linops, space, dt, w, mkx, lkx, mky,lky; numterm=scale, ng = length(mk), J=J, hx=hx, hy=hy, hz=hz)
            push!(rv, term)
        end
    end

    return rv
end
function fsmbigeneral(
    op,
    linops,
    space,
    dt,
    wk,
    mkx,
    lkx,
    mky,
    lky;
    hx=0.0,
    hy=0.0,
    hz=0.0,
    J=1.0,
    numterm,
    ng,
    extend=true,
    thermal=false,
)
    T = Dict{NTuple{4,Int},Matrix{ComplexF64}}

    rules = (A=T(), B=T(), C=T(), D=T())

    X, Y, Z = PAULI

    D = -im * hx * (kron(X, one(X)) - kron(one(X), transpose(X)))
    D += -im * hy * (kron(Y, one(Y)) - kron(one(Y), transpose(Y)))
    D += -im * hz * (kron(Z, one(Z)) - kron(one(Z), transpose(Z)))

    for L in linops
        acomm = (kron(L' * L, one(L)) + kron(one(L), transpose(L' * L)))
        D += kron(L, transpose(L')) - 1 / 2 * acomm
    end

    D = D / numterm

    rules.A[1, 1, 1, 1] = one(D)
    rules.D[1, 1, 1, 1] = D

    Id = one(op)

    if thermal
        c1 = c2 = -J / 2
    else
        c1 = -im * J
        c2 = im * J
    end

    C1 = c1 * kron(op, Id)
    B1 = kron(op, Id)

    C2 = c2 * kron(Id, transpose(op))
    B2 = kron(Id, transpose(op))

    # Get strides

    hstr = 1
    vstr = 1

    for (C, B) in zip((C1, C2), (B1, B2))
        Id = kron(one(op), one(op))

        nh = 1
        nv = 1

        c = hstr + 2 * nh - 1
        u = hstr + 2 * nh

        v = vstr + nv

        ## X-Rules

        # Start going east
        rules.C[c, 1, 1, 1] =  wk * mkx * lkx * C / ng

        # Finish going east
        rules.B[1, 1, c, 1] = B
        # Finish going west (unconventional)
        rules.B[u, 1, 1, 1] = B

        # Continue going east
        rules.A[c, 1, c, 1] = lkx * Id
        # Continue going west (unconventional)
        rules.A[u, 1, u, 1] = lkx * Id

        ## Y -> X rules

        # Change from going north to going east
        rules.A[c, v, 1, 1] = ng * mkx * lkx * Id
        # Change from going north to going west (unconventional)
        rules.A[1, v, u, 1] = ng * mkx * lkx * Id

        ## Y-rules

        # Start going north
        rules.C[1, 1, 1, v] = wk * mky * lky * C / ng

        # Finish going north
        rules.B[1, v, 1, 1] = B

        # Continue going north
        rules.A[1, v, 1, v] = lky * Id

        hstr += 2
        vstr += 1
    end

    @info "TNO bond dimension: ($hstr,$vstr)"

    d = space
    di = dim(d)

    Dx = reduce(oplus, fill(oneunit(d), hstr))
    Dy = reduce(oplus, fill(oneunit(d), vstr))

    data = zeros(ComplexF64, di, di, di, di, hstr, vstr, hstr, vstr)

    fsmpepo!(data, rules, dt; extend=extend)

    return TensorMap(data, d * d' * d' * d, Dx * Dy * Dx' * Dy')
end

function fsmgeneral(
    op,
    localops,
    linops,
    space,
    dt,
    wk,
    mks,
    lks;
    J=1.0,
    localscaling = 1.0,
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
        c1 = c2 = -J / 2
    else
        c1 = -im * J
        c2 = im * J
    end

    C1 = c1 * kron(op, Id)
    B1 = kron(op, Id)

    C2 = c2 * kron(Id, transpose(op))
    B2 = kron(Id, transpose(op))

    # Get strides

    hstr = 1
    vstr = 1

    for (C, B) in zip((C1, C2), (B1, B2))
        Id = kron(one(op), one(op))
        gapprox = zip(mks, lks)

        for (nh, (mk, lk)) in enumerate(gapprox)
            # mklk = mk * lk 

            c = hstr + 2 * nh - 1
            u = hstr + 2 * nh

            # Start going east
            rules.C[c, 1, 1, 1] = wk * mk * lk * C

            # Finish going east
            rules.B[1, 1, c, 1] = B
            # Finish going west (unconventional)
            rules.B[u, 1, 1, 1] = B

            # Continue going east
            rules.A[c, 1, c, 1] = lk * Id
            # Continue going west (unconventional)
            rules.A[u, 1, u, 1] = lk * Id

            # Direction change rules
            for nv in 1:length(gapprox)
                v = vstr + nv

                # Change from going north to going east
                rules.A[c, v, 1, 1] = mk * lk * Id
                # Change from going north to going west (unconventional)
                rules.A[1, v, u, 1] = mk * lk * Id
            end
        end
        for (nv, (mk, lk)) in enumerate(gapprox)
            v = vstr + nv
            # Start going north
            rules.C[1, 1, 1, v] = wk * mk * lk * C

            # Finish going north
            rules.B[1, v, 1, 1] = B

            # Continue going north
            rules.A[1, v, 1, v] = lk * Id
        end

        hstr += 2 * length(gapprox)
        vstr += length(gapprox)
    end

    @info "TNO bond dimension: ($hstr,$vstr)"

    d = space
    di = dim(d)

    Dx = reduce(oplus, fill(oneunit(d), hstr))
    Dy = reduce(oplus, fill(oneunit(d), vstr))

    data = zeros(ComplexF64, di, di, di, di, hstr, vstr, hstr, vstr)

    fsmpepo!(data, rules, dt; extend=extend)

    pepo = TensorMap(data, d * d' * d' * d, Dx * Dy * Dx' * Dy')
    return pepo
end

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
    localscaling = 1.0,
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

    for (C, B, j) in zip((C1, C2), (B1, B2), (j1,j2))
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
function twodapprox(alpha, n, ngmax=2; V=1.0, N=50, T=ComplexF64, gtol = 1e-10, force=false)
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
