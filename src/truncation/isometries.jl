function isometries(pepo::AbstractPEPO, environment, bond::Bond, method::Union{FET,NTU})
    rankreduce = method.rankreduce
    gauge = method.gauge
    xi = method.ξ

    iso = isometries(pepo, environment, bond; rankreduce=rankreduce, gauge=gauge, xi=xi)

    return iso
end

function isometries(pepo::AbstractPEPO, ::SimpleUpdateState, bond::Bond, ::SU)
    return isometries(pepo, bond)
end

function isometries(pepo, environment, bond; D=bondspace(pepo, bond), rankreduce, gauge, xi)
    if rankreduce
        left0, right0 = rankreduce!(pepo, bond)
        tmetric = metric(pepo, environment, bond)
    else
        tmetric = metric(pepo, environment, bond)
        left0, right0 = isometryinit(tmetric)
    end
    iso = metricoptimise(
        left0, right0, tmetric, truncspace(D); rankreduce=rankreduce, gauge=gauge, xi=xi
    )

    # iso = TruncationIsometries(pepo, u, s, v)
    #
    # wtg!(iso, tmetric, Val(gauge))
    # if xi < Inf
    #     copy!(iso.s, rescale(iso.s, xi))
    #     wtg!(iso, tmetric, Val(gauge))
    # end

    return iso
end

function isometries(pepo, bond; D=bondspace(pepo, bond))
    left, right = rankreduce!(pepo, bond)

    _, sun, _, _ = tsvd(left * right)
    u, s, v, err = tsvd(left * right; trunc=truncspace(D))

    iso = TruncationIsometries(pepo, u, s, v)

    # println("sun:", sun)
    println("s:", convert(Array, s))

    return iso
end

function rankreduce!(pepo::LocalUpdatablePEPO, bond::Bond)
    left, right = evolvedpair(pepo, bond)
    P1, R, L, P2 = rankreduce(left, right)
    setevolvedpair!(pepo, P1, P2, bond)
    return R, L
end

rankreduce!(pepo::GlobalUpdatablePEPO, bond::Bond) = rankreduce!(evolvedtensors(pepo), bond)

function rankreduce!(lattice::AbstractLattice, bond::Bond)
    left, right = lattice[bond]

    P1, R, L, P2 = rankreduce(left, right)

    lattice[bond] = P1, P2

    return R, L
end

function rankreduce(left_site::AbstractTensorMap, right_site::AbstractTensorMap)
    P1, R = leftrankreduce(left_site)
    L, P2 = rightrankreduce(right_site)

    return P1, R, L, P2
end

function leftrankreduce(p::AbsTen{2,4})
    Q, R = try
        leftorth(p, (1, 2, 4, 5, 6), (3,); alg=Polar())
    catch e
        if e isa AssertionError && e.msg == "V ≅ domain(t)"
            @error """
            potential bad choice of bond dimension sectors; try different sectors or
            setting `rankreduce=false` if you insist on these sectors
            """
        end
        rethrow()
    end
    P = permute(Q, ((1, 2), (6, 3, 4, 5)))
    return P, R
end
function rightrankreduce(p::AbsTen{2,4})
    L, Q = try
        rightorth(p, (5,), (1, 2, 3, 4, 6); alg=Polar())
    catch e
        if e isa AssertionError && e.msg == "V ≅ domain(t)"
            @error """
            potential bad choice of bond dimension sectors; try different sectors or
            setting `rankreduce=false` if you insist on these sectors
            """
        end
        rethrow()
    end
    P = permute(Q, ((2, 3), (4, 5, 1, 6)))
    return L, P
end

function constructG(L, g, R)
    @tensoropt G[1 2; 3 4] := L[3; 3] * R[4; 4] * g[1 2; 3 4] * (L')[1; 1] * (R')[2; 2]
end

function isometryinit(metric::AbstractLattice)
    leftright = isometryinit.(metric)

    left = getindex.(leftright, 1)
    right = getindex.(leftright, 2)

    return left, right
end

# Isometry update requires one to contruct initial guesses for L & R.
function isometryinit(metric::AbsTen{2,2})
    η_left = domain(metric)[1]      # Upper left index
    η_right = codomain(metric)[1]   # Upper right index

    @assert η_left == η_right

    # left = id(η_left)
    # right = id(η_right)
    # normalize!(left)
    # normalize!(right)

    initmat = id(η_left)

    u, s, v = tsvd!(initmat; alg=TensorKit.SVD())
    left = u * sqrt(s)
    right = sqrt(s) * v

    # left = TensorMap(randn, eltype(metric), η_left, η_left)
    # right = TensorMap(randn, eltype(metric), η_right, η_right)

    normalize!(left)
    normalize!(right)

    return left, right
end

function new_costfunc(A, B, Y, LR)
    AB = permute(A * B, (1, 2), ())
    # normalize!(AB)
    # LR = normalize(LR)
    return norm(1 - scalar(AB' * Y * LR) - scalar(LR' * Y * AB) + scalar(AB' * Y * AB))
end
function fidelity(A, B, Y, LR)
    AB = permute(A * B, (1, 2), ())
    # normalize!(AB)
    # LR = normalize(LR)
    return scalar(AB' * Y * LR) * scalar(LR' * Y * AB) / scalar(AB' * Y * AB)
end

# Optimise the isometries `left` and `right` with respect to `metric`.
function metricoptimise(
    left, right, metric::AbsTen{2,2}, trunc; rankreduce, gauge, xi, kwargs...
)
    if rankreduce
        left0 = left
        right0 = right
    else
        left0 = one(left)
        right0 = one(right)
    end

    #left0 -> u
    #right0 -> v
    LR0 = permute(left0 * right0, (1, 2), ())
    # normalize!(LR0)

    # Put in the appropriate matrix form for the Ritz procedure. 
    Y = permute(metric, (2, 4), (3, 1))

    # n = Y.data[1, 1]
    # rmul!(Y, 1 / n)
    # Check if the metric is hermitian, If not, use SU

    F0 = scalar(LR0' * Y * LR0)

    hermiticity = norm(Y - Y')

    # htol = 1e-10
    # htol = 0.0 # force SU
    htol = Inf # force update

    hermiticity > htol &&
        @warn "Metric not hermitian! non-hermiticity:  $hermiticity, tolerance $htol"

    @debug "Metric normalisation (F0): $F0"

    if hermiticity > htol
        @info "Metric not hermitian: using SU"
        u, s, v, err = tsvd(left * right; trunc=trunc)
        @info "Truncation error: $err"
        return TruncationIsometries(metric, u, s, v)
    end

    # Obtain the normalisation constant.

    #Symmetrise and normalise the truncation metric.
    Y = (0.5 * (Y + Y')) / F0

    # This is type unstable and making things slow
    # Y, err = posapprox(Y)
    # @info "Positive approximation error: $err"

    # Construct the cost function to measure convergence.
    # cf = (Ma, Mb) -> metric_costfunction(Ma, Mb, Y, left, right, 1)
    cf = (x, y) -> new_costfunc(x, y, Y, LR0)

    F0 = scalar(LR0' * Y * LR0)
    @debug "Minimal infidelity: $(cf(left , right)). Normalisation check: $F0"

    # Proceed with metric optimisation.
    # u, s, v = optiso(cf, left, right, Y, trunc, left0, right0; kwargs...)
    # u, s, v = optisoalt(cf, left, right, Y, trunc, left0, right0; kwargs...)
    # u, s, v = optiso_og(cf, left, right, Y, trunc, left0, right0; kwargs...)
    # u, s, v = optisonew(cf, left, right, Y, trunc, left0, right0; kwargs...)
    u, s, v = optisonew2(cf, left, right, Y, trunc, left0, right0; kwargs...)

    # @info "" norm(u1 - convert(Array, u2))
    # # @info "" norm(v1 - v2)
    #
    # u = u2
    # v = v2
    # s = s2

    # s = s / tr(s)
    @info "" fidelity = fidelity(u, v, Y, LR0)

    iso = TruncationIsometries(metric, u, s, v)

    permute!(metric, Y, ((4, 1), (3, 2)))

    wtg!(iso, metric, Val(gauge))
    if xi < Inf
        copy!(iso.s, rescale(iso.s, xi))
        wtg!(iso, metric, Val(gauge))
    end

    return iso
end

function _metric_costfunction(Ma, Mb, g, Ra, Rb)
    MM = normalize(Ma * Mb)
    RR = normalize(Ra * Rb)
    left = (MM - RR)'
    right = left'
    @tensor err = left[b; a] * g[a b; c d] * right[c; d]
end

function metric_costfunction(Ma, Mb, g, Ra, Rb, F0)
    return real((_metric_costfunction(Ma, Mb, g, Ra, Rb) / F0))
end

function optiso(
    cf, left, right, metric, trunc, left0=left, right0=right; EAT=false, maxiter=100
)
    # Convenience
    Y = metric
    L = left
    R = right

    # The following tensor will be useful...
    LR0 = permute(left0 * right0, (1, 2), ())
    # normalize!(LR0)

    Y_LR = Y * LR0

    u_su, s_su, v_su = try
        tsvd(left0 * right0; trunc=trunc)
    catch e
        @warn "gesdd failed; trying gesvd:"
        tsvd(left0 * right0; trunc=trunc, alg=TensorKit.SVD())
    end

    @debug "Simple update error: $(cf(u_su * sqrt(s_su), sqrt(s_su) * v_su))"

    # Perform EAT if we want to.
    if EAT
        Ma0, Mb0 = eat(L, R, metric, trunc)
    else
        Ma0, Mb0 = L, R
    end

    # Initialise isometries for optimisation.
    u0, s0, Mb0 = try
        tsvd(Ma0 * Mb0; trunc=trunc)
    catch e
        if e isa LAPACKException
            @warn "gesdd failed; trying gesvd:"
            tsvd(Ma0 * Mb0; trunc=trunc, alg=TensorKit.SVD())
        else
            rethrow()
        end
    end

    ϵ0 = cf(u0 * sqrt(s0), sqrt(s0) * Mb0)
    @debug "Initial (post optional EAT) error: $ϵ0"
    # Calculate the initial tolerance

    # Alias for iteration.
    u, s, Mb = u0, s0, Mb0

    # Get an appropriate identity matrix for the spaces.
    id_a = id(codomain(u))
    id_b = transpose(id(domain(Mb)))

    counter = 0
    δϵ = 1.0
    ϵ = ϵ0
    print("Optimising isometries: [")
    while !(counter > maxiter || δϵ < 1e-6)
        # while !(counter > maxiter || δϵ < 10 * eps())
        s_old = s

        ϵ_old = ϵ

        # Optimise for Ma with Mb fixed.
        Ma = u * s
        Ma_max = optsingleiso(id_a ⊗ transpose(Mb), Y, Y_LR)
        Ma, s, v = tsvd(Ma_max * Mb; trunc=trunc, alg=TensorKit.SVD())

        # normalize!(s)

        # Optimise for Mb with Ma fixed.
        Mb = s * v
        Mb_max = optsingleiso(Ma ⊗ id_b, Y, Y_LR)
        u, s, Mb = tsvd(Ma * Mb_max; trunc=trunc, alg=TensorKit.SVD())

        # normalize!(s)

        # calculate convergence measure
        ϵ = norm(s - s_old)
        # Calculate truncation error (this is slow)
        # ϵ = cf(u * sqrt(s), sqrt(s) * Mb)

        # Calculate convergence measure.
        δϵ = abs(ϵ - ϵ_old)

        print("|")

        counter = counter + 1
    end
    print("]\n")

    ϵ = cf(u * sqrt(s), sqrt(s) * Mb)

    @debug "Post-optimisation error: $ϵ"
    # Return new isometries for transformation to the WTG
    if ϵ0 < ϵ
        @info "Isometries could not be improved from an infidelity of $ϵ0 after $counter iterations"
        return u0, s0, Mb0
    else
        @info "Isometries optimised to an infidelity of: $ϵ from $ϵ0 after $counter iterations."
        return u, s, Mb
    end
end

# Evenbly
function optisonew(
    cf,
    left,
    right,
    metric,
    trunc,
    left0=left,
    right0=right;
    maxiter=100,
    verbose=true,
    eigentol=sqrt(eps(real(float(oneunit(scalartype(metric)))))),
    isotol=1e-8,
)
    D = trunc.space
    η = codomain(metric)[1]

    T = storagetype(metric)
    S = spacetype(metric)

    env = metric

    # v u ; v u
    env = 0.5 * (env + env')
    @tensor penvN = env[i, i, j, j]
    env /= penvN

    println(space(env))

    # v u ; v u
    env = permute(env, (2, 1), (4, 3))

    ϵ = Inf
    counter = 0

    ut = isometry(T, η, D) #+ im * isometry(T, η, D)
    vt = isometry(T, η', D') #+ im * isometry(T, η', D')

    # utp = TensorMap(randn, eltype(T), η, D) #+ im * isometry(T, η, D)
    # vtp = TensorMap(randn, eltype(T), η', D') #+ im * isometry(T, η', D')
    #
    # ut = ut + 1e-6 * utp
    # vt = vt + 1e-6 * vtp

    utold = isometry(T, η, D)
    vtold = isometry(T, η', D')

    B1 = similar(metric, η' * D, η' * D)
    B2 = similar(metric, D' * η, D' * η)

    P1 = similar(metric, η' * D, one(S))
    P2 = similar(metric, D' * η, one(S))

    s1old = id(D')
    s2old = id(D)

    sw = x -> cj(transpose(x'))

    verbose ? print("performing FET: [") : ()

    for k in 1:maxiter
        copy!(utold, ut)
        copy!(vtold, vt)

        @tensoropt B1[i j; k l] = env[i α; k β] * sw(ut)[α; j] * cj(ut)[β; l]
        @tensoropt P1[i; j] = env[i α; β β] * sw(ut)[α; j]

        u, s1, v = tsvd(pinv(0.5 * (B1 + B1'); rtol=eigentol) * P1, (1,), (2,))

        copy!(vt, u * v)

        @tensor B2[i j; k l] = (env)[α j; β l] * cj(sw(vt))[α; i] * vt[β; k]
        @tensor P2[i; j] = (env)[α j; β β] * cj(sw(vt))[α; i]

        R = pinv(0.5 * (B2 + B2'); rtol=eigentol) * P2

        # @info "" cj(transpose(permute(R, (1,), (2,))))

        RP = cj(transpose(permute(R, (1,), (2,))))

        u, s2, v = tsvd(RP; alg=TensorKit.SVD())

        copy!(ut, u * v)

        ϵ1 = norm(normalize(s1) - normalize(s1old))
        ϵ2 = norm(normalize(s2) - normalize(s2old))

        copy!(s1old, s1)
        copy!(s2old, s2)

        ϵ = max(ϵ1, ϵ2)

        counter += 1
        verbose ? print("|") : ()
        ϵ < isotol ? break : nothing
    end
    @tensoropt B1[i j; k l] = env[i α; k β] * sw(ut)[α; j] * cj(ut)[β; l]
    @tensoropt P1[i; j] = env[i α; β β] * sw(ut)[α; j]

    u, s, v = tsvd(pinv(0.5 * (B1 + B1'); rtol=eigentol) * P1, (1,), (2,))

    normalize!(s)

    r = u * sqrt(s) * v
    l = ut * cj(transpose(v)) * sw(sqrt(s)) * sw(v)

    ######## end iteration and output convergence data #########################
    verbose ? print("] iterations: ", counter, " ϵ: ", ϵ, "\n") : ()
    enva = convert(Array, env)
    la = convert(Array, l)
    ra = convert(Array, r)
    sa = diag(convert(Array, s))

    # isometries_to_WTG!(enva, la, ra, sa)
    to_WTG!(env, l, r, s)

    # l = TensorMap(la, η, D)
    # r = TensorMap(ra, η', D')

    return l, id(D), transpose(cj(r))
end

#Evenbly adapted
function optisonew2(
    cf,
    left,
    right,
    metric,
    trunc,
    left0=left,
    right0=right;
    maxiter=100,
    verbose=true,
    eigentol=sqrt(eps(real(float(oneunit(scalartype(metric)))))),
    isotol=1e-8,
)
    D = trunc.space
    η = codomain(metric)[1]

    T = storagetype(metric)
    S = spacetype(metric)

    env = metric

    # v u ; v u
    env = 0.5 * (env + env')
    @tensor penvN = env[i, i, j, j]
    env /= penvN

    # v u ; v u

    ϵ = Inf
    counter = 0

    ut = isometry(T, η, D) + 1e-3 * TensorMap(randn, eltype(T), η, D)
    vt = convert(TensorMap, ut')

    B1 = similar(metric, η * D', η * D')
    B2 = similar(metric, D * η', D * η')

    P1 = similar(metric, η * D', one(S))
    P2 = similar(metric, D * η', one(S))

    R1 = similar(P1)
    R2 = similar(P2)

    s1old = id(D)
    s2old = id(D)

    verbose ? print("performing FET: [") : ()

    for k in 1:maxiter
        @tensor B1[i j; k l] = transpose(vt')[j; α] * env[i α; k β] * transpose(vt)[β; l]
        @tensor P1[i; j] = transpose(vt')[j; α] * env[i α; β β]

        axpby!(0.5, B1', 0.5, B1)

        mul!(R1, pinv(B1; rtol=eigentol), P1)

        u, s1, v = tsvd(R1, (1,), (2,))

        copy!(ut, u * v)

        @tensor B2[i j; k l] = (ut')[i; α] * env[α j; β l] * ut[β; k]
        @tensor P2[i; j] = (ut')[i; α] * env[α j; β β]

        axpby!(0.5, B2', 0.5, B2)

        mul!(R2, pinv(B2; rtol=eigentol), P2)

        u, s2, v = tsvd(R2, (1,), (2,))

        copy!(vt, u * v)

        ϵ1 = norm(normalize(s1) - normalize(s1old))
        ϵ2 = norm(normalize(s2) - normalize(s2old))

        copy!(s1old, s1)
        copy!(s2old, s2)

        ϵ = max(ϵ1, ϵ2)

        counter += 1
        verbose ? print("|") : ()
        ϵ < isotol ? break : nothing
    end
    @tensor B1[i j; k l] = transpose(vt')[j; α] * env[i α; k β] * transpose(vt)[β; l]
    @tensor P1[i; j] = transpose(vt')[j; α] * env[i α; β β]

    axpby!(0.5, B1', 0.5, B1)

    mul!(R1, pinv(B1; rtol=eigentol), P1)

    u, s, v = tsvd(R1, (1,), (2,))

    normalize!(s)

    l = u * sqrt(s) * v
    # It's more stable to calculate v then take v'; v and vt' are isometries so v * v' = I
    r = vt' * v' * sqrt(s) * v

    # rp = v' * inv(sqrt(s)) * v * vt

    ######## end iteration and output convergence data #########################
    verbose ? print("] iterations: ", counter, " ϵ: ", ϵ, "\n") : ()
    enva = convert(Array, env)
    la = convert(Array, l)
    ra = convert(Array, r)
    sa = diag(convert(Array, s))

    envb = permutedims(enva, [2, 1, 4, 3])

    # isometries_to_WTG!(envb, la, ra, sa)
    rp = convert(TensorMap, r')
    # to_WTG2!(env, l, rp, s)

    env = permute(env, ((4, 1), (3, 2)))
    wtg!(l, s, rp, env)

    # l = TensorMap(la, η, D)
    # r = TensorMap(conj(permutedims(ra)), D, η)

    return l * sqrt(s), one(s), sqrt(s) * rp 
end

function optiso_og(
    cf,
    left,
    right,
    metric,
    trunc,
    left0=left,
    right0=right;
    maxiter=100,
    verbose=true,
    eigentol=sqrt(eps(real(float(oneunit(scalartype(metric)))))),
    isotol=1e-8,
)
    Dsp = trunc.space
    D = dim(Dsp)
    χsp = codomain(metric)[1]

    T = scalartype(metric)

    env = convert(Array, metric)

    χenv = size(env, 1)

    env .= 0.5 * (env .+ permutedims(conj(env), [3, 4, 1, 2]))
    @tensor penvN = env[i, i, j, j]
    env ./= penvN

    env = permutedims(env, [2, 1, 4, 3])
    #
    # Deig = eigen!(Hermitian(reshape(env, χenv^2, χenv^2)))
    # χeig = sum(real.(Deig.values) .> 1e-3eigentol)
    # vvv = sortperm(real(Deig.values); rev=true)
    # env .= reshape(
    #     (
    #         Deig.vectors[:, vvv[1:χeig]] *
    #         Diagonal(Deig.values[vvv[1:χeig]]) *
    #         (Deig.vectors[:, vvv[1:χeig]]')
    #     ),
    #     (χenv, χenv, χenv, χenv),
    # )

    # alternating optimization

    ϵ = 0.0
    counter = 0

    ut = Matrix{T}(I, χenv, D) #+ im * Matrix{T}(I, χenv, D)
    vt = Matrix{T}(I, χenv, D) #+ im * Matrix{T}(I, χenv, D)
    utold = Matrix{T}(I, χenv, D)
    vtold = Matrix{T}(I, χenv, D)

    B1 = zeros(T, χenv, D, χenv, D)
    B2 = zeros(T, D, χenv, D, χenv)
    P1 = zeros(T, χenv, D)
    P2 = zeros(T, D, χenv)

    s1old = zeros(Float64, D)
    s2old = zeros(Float64, D)

    verbose ? print("performing FET: [") : ()
    for k in 1:maxiter
        utold .= ut
        vtold .= vt

        @tensor B1[i, j, k, l] = (env)[i, α, k, β] * ut[α, j] * conj(ut)[β, l]
        @tensor P1[i, j] = (env)[i, α, β, β] * ut[α, j]

        F1 = svd(
            reshape(
                pinv(
                    0.5 * (
                        reshape(B1, (χenv * D, χenv * D)) .+
                        reshape(B1, (χenv * D, χenv * D))'
                    );
                    rtol=eigentol,
                ) * reshape(P1, (χenv * D)),
                χenv,
                D,
            ),
        )
        #F1 = svd(reshape(svdt!(0.5*(reshape(p.B1,(χenv*D,χenv*D)).+reshape(p.B1,(χenv*D,χenv*D))'),eigentol )\vec(p.P1),χenv,D))
        # F1 = svd(
        #     reshape(
        #         qr(
        #             0.5 * (
        #                 reshape(B1, (χenv * D, χenv * D)) .+
        #                 reshape(B1, (χenv * D, χenv * D))'
        #             ),
        #             Val(true),
        #         ) \ vec(P1),
        #         χenv,
        #         D,
        #     ),
        # )
        vt .= F1.U * F1.Vt

        # @info "$k" vt

        @tensor B2[i, j, k, l] = (env)[α, j, β, l] * conj(vt)[α, i] * vt[β, k]
        @tensor P2[i, j] = (env)[α, j, β, β] * conj(vt)[α, i]

        # B2inv = pinv(
        #     0.5 * (reshape(B2, (χenv * D, χenv * D)) .+ reshape(B2, (χenv * D, χenv * D))');
        #     rtol=eigentol,
        # )
        R = reshape(
            pinv(
                0.5 *
                (reshape(B2, (χenv * D, χenv * D)) .+ reshape(B2, (χenv * D, χenv * D))');
                rtol=eigentol,
            ) * reshape(P2, (χenv * D)),
            D,
            χenv,
        )
        F2 = svd(R')

        ut .= F2.U * F2.Vt

        # @info "$k" ut

        ϵ1 = norm(normalize(F1.S) - normalize(s1old))
        ϵ2 = norm(normalize(F2.S) - normalize(s2old))

        copy!(s1old, F1.S)
        copy!(s2old, F2.S)

        ϵ = max(ϵ1, ϵ2)

        counter += 1

        verbose ? print("|") : ()

        ϵ < isotol ? break : nothing
    end

    @tensor B1[i, j, k, l] = (env)[i, α, k, β] * ut[α, j] * conj(ut)[β, l]
    @tensor P1[i, j] = (env)[i, α, β, β] * ut[α, j]

    F1 = svd(
        reshape(
            pinv(
                0.5 *
                (reshape(B1, (χenv * D, χenv * D)) .+ reshape(B1, (χenv * D, χenv * D))');
                rtol=eigentol,
            ) * reshape(P1, (χenv * D)),
            χenv,
            D,
        ),
    )

    #F1 = svd(reshape(svdt(0.5*(reshape(p.B1,(χenv*D,χenv*D)).+reshape(p.B1,(χenv*D,χenv*D))'),eigentol )\vec(p.P1),χenv,D))
    #F1 = svd(reshape(qr(0.5*(reshape(p.B1,(χenv*D,χenv*D)).+reshape(p.B1,(χenv*D,χenv*D))'),Val(true))\vec(p.P1),χenv,D))

    F1.S ./= F1.S[1]

    # normalize!(F1.S)

    #EXPERIMENTAL: insert fictitious damping here?
    #of "Schmidt" coefficients here (?)
    #scl=[exp(-1e-0*(i-1)) for i =1:length(F1.S)]
    #F1.S.*=scl

    r = F1.U * Diagonal(sqrt.(F1.S)) * F1.Vt
    l = ut * F1.Vt' * Diagonal(sqrt.(F1.S)) * F1.Vt
    σ = F1.S[1:D]

    ######## end iteration and output convergence data #########################
    verbose ? print("] iterations:", counter, " δ:", ϵ, "\n") : ()
    isometries_to_WTG!(env, l, r, σ)

    lt = TensorMap(l, χsp, Dsp)
    rt = TensorMap(permutedims(conj(r)), Dsp, χsp)

    return lt, id(Dsp), rt
end

function to_WTG!(
    env::AbstractTensorMap{S}, l, r, σ, test::Bool=true, ξ::Float64=Inf
) where {S}
    sw = x -> cj(transpose(x'))

    # v' u ; v' u
    @tensor enviso[i j; k l] :=
        env[a b; c d] * sw(cj(r))[a; i] * sw(l)[b; j] * r[c; k] * cj(l)[d; l]
    @tensor envisoN = enviso[i i; j j]
    enviso /= envisoN
    enviso = 0.5 * (enviso + enviso')

    envx = permute(cj(enviso), (2, 4), (1, 3))
    x0 = isometry(domain(envx), one(S))

    _, vecs, _ = eigsolve(x -> envx * x, x0, 1, :LM)

    DL, VL = eigen(vecs[1], (1,), (2,))

    envy = permute(enviso, (1, 3), (2, 4))
    y0 = isometry(domain(envy), one(S))

    _, vecs, _ = eigsolve(x -> envy * x, y0, 1, :LM)

    DR, VR = eigen(vecs[1], (1,), (2,))

    DR = sw(DR)
    VR = sw(VR)

    for (_, v) in blocks(DL)
        if minimum(abs.(diag(v))) < 1e-14
            @info "skipping gauge fixing (left)"
            return nothing
        end
    end
    for (_, v) in blocks(DR)
        if minimum(abs.(diag(v))) < 1e-14
            @info "skipping gauge fixing (right)"
            return nothing
        end
    end

    # Lmin = minimum(abs.(DL.values))
    # Rmin = minimum(abs.(DR.values))
    # if (Lmin < 1e-14 || Rmin < 1e-14)
    #     (@info "skipped gauge fixing"; return nothing)
    # else
    #     nothing
    # end

    UT, ST, VT = tsvd!(sqrt(DL) * VL' * VR * sqrt(DR))

    xnI = l * VL * inv(sqrt(DL)) * UT * sqrt(ST)
    ynI = r * sw(sqrt(ST) * VT * inv(sqrt(DR)) * VR')'

    copy!(l, xnI)
    copy!(r, ynI)

    println(ST)

    if false
        #check that conditions for WTG have been met by new l, r and σ
        @tensor enviso[i, j, k, l] =
            env[a, b, c, d] * cj(r)[a, i] * l[b, j] * r[c, k] * cj(l)[d, l]
        envtest = TensorKit.TO.tensortrace(
            enviso .* reshape(sqrt.(F.S), (1, :, 1, 1)) .*
            reshape(sqrt.(F.S), (1, 1, 1, :)) ./ reshape(sqrt.(F.S), (:, 1, 1, 1)) ./
            reshape(sqrt.(F.S), (1, 1, :, 1)),
            [1, 2, 3, 2],
        )
        if envtest ./ real(envtest[1, 1]) ≈ I
            nothing
        else
            @warn "left gauge condition was not met"
        end
        envtest = TensorKit.TO.tensortrace(
            enviso ./ reshape(sqrt.(F.S), (1, :, 1, 1)) ./
            reshape(sqrt.(F.S), (1, 1, 1, :)) .* reshape(sqrt.(F.S), (:, 1, 1, 1)) .*
            reshape(sqrt.(F.S), (1, 1, :, 1)),
            [1, 2, 1, 4],
        )
        if envtest ./ real(envtest[1, 1]) ≈ I
            nothing
        else
            @warn "right gauge condition was not met"
        end
    end
    return nothing
end

function to_WTG2!(
    env::AbstractTensorMap{S}, u, v, s, test::Bool=true, ξ::Float64=Inf
) where {S}
    sw = x -> cj(transpose(x'))

    println(space(env))

    @tensor B[i j; k l] :=
        (u')[i; a] * transpose(v')[j; b] * env[a b; c d] * u[c; k] * transpose(v)[d; l]
    @tensor BN = B[i i; j j]

    axpby!(0.5 / BN, B', 0.5 / BN, B)

    # Swap 2 <-> 3
    Y = permute(B, (1, 3), (2, 4))

    x0 = isometry(domain(Y), one(S))
    _, vecs, _ = eigsolve(x -> Y * x, x0, 1, :LM)

    R0 = permute(vecs[1], (1,), (2,))
    DR, VR = eigen(R0)

    y0 = convert(TensorMap, x0')
    _, vecs, _ = eigsolve(x -> x * Y, y0, 1, :LM)

    L0 = permute(vecs[1], (2,), (1,))
    DL, VL = eigen(L0)

    for (_, v) in blocks(DL)
        if minimum(abs.(diag(v))) < 1e-14
            @info "skipping gauge fixing (left)"
            return nothing
        end
    end
    for (_, v) in blocks(DR)
        if minimum(abs.(diag(v))) < 1e-14
            @info "skipping gauge fixing (right)"
            return nothing
        end
    end

    UT, ST, VT = tsvd!(sqrt(DL) * VL' * VR * sqrt(DR))

    x = UT * sqrt(DL) * VL'
    invxa = UT * inv(sqrt(DL)) * VL'
    y = VR * sqrt(DR) * VT'
    invya = VR * inv(sqrt(DR)) * VT'

    @debug begin
        xn = norm(x' * x)
        L0n = norm(L0)

        yn = norm(y * y')
        R0n = norm(R0)

        diffx = norm((x' * x) / xn - L0 / L0n)
        diffy = norm((y * y') / yn - R0 / R0n)

        "check ≈ 0.0: x' * x - L0 = $(diffx); y * y' - R0 = $(diffy)"
    end

    copy!(s, ST)

    copy!(u, u * invya)
    copy!(v, invxa * v)

    for (_, v) in blocks(s)
        max = zero(scalartype(s))
        vmax = maximum(abs.(v))
        if vmax > max
            max = vmax
        end
        rmul!(s, 1/max)
    end
    
    @info "" s

    if test
        #check that conditions for WTG have been met by new l, r and σ

        @tensor C[i j; k l] :=
            (u')[i; a] * transpose(v')[j; b] * env[a b; c d] * u[c; k] * transpose(v)[d; l]


        @tensor T[j l] := s[x; i] * s[k; x] * C[i j; k l]
        println(T)

        @tensor T[i k] := s[j; x] * s[x; l] * C[i j; k l]

        println(T)

    end
    return nothing
end
function isometries_to_WTG!(
    env::Union{Array{T,4},Array{Complex{T},4}},
    l::Union{Array{T,2},Array{Complex{T},2}},
    r::Union{Array{T,2},Array{Complex{T},2},Adjoint},
    σ::Array{T,1};
    test::Bool=true,
    ξ::Float64=Inf,
) where {T}
    D = length(σ)
    #check_degeneracy(p.σ)
    @tensor enviso[i, j, k, l] :=
        env[a, b, c, d] * conj(r)[a, i] * l[b, j] * r[c, k] * conj(l)[d, l]
    @tensor envisoN = enviso[i, i, j, j]
    enviso .= enviso ./ envisoN
    enviso .= 0.5 * (enviso .+ permutedims(conj(enviso), [3, 4, 1, 2]))
    #enviso.=0.5*real.(enviso.+permutedims(enviso,[3,4,1,2]));
    temp = eigen((reshape(permutedims(conj(enviso), [2, 4, 1, 3]), D^2, D^2)))
    xtemp = reshape(temp.vectors[:, findmax(real(temp.values))[2]], D, D)
    temp = eigen((reshape(permutedims(enviso, [1, 3, 2, 4]), D^2, D^2)))
    ytemp = reshape(temp.vectors[:, findmax(real(temp.values))[2]], D, D)
    #DL = eigen!(Hermitian(xtemp));DR = eigen!(Hermitian(ytemp));
    DL = eigen!(xtemp)
    DR = eigen!(ytemp)

    #TODO: set any tiny eigenvectors to zero
    Lmin = minimum(abs.(DL.values))
    Rmin = minimum(abs.(DR.values))
    if (Lmin < 1e-14 || Rmin < 1e-14)
        (@info "skipped gauge fixing"; return nothing)
    else
        nothing
    end
    xnI = DL.vectors * Diagonal((1 ./ sqrt.(abs.(DL.values))))
    ynI = Diagonal((1 ./ sqrt.(abs.(DR.values)))) * DR.vectors'
    F = svd!(
        Diagonal((sqrt.(abs.(DL.values)))) *
        DL.vectors' *
        DR.vectors *
        Diagonal(sqrt.(abs.(DR.values))),
    )

    F.S ./= F.S[1] # added
    @info "" F.S

    if ξ != Inf
        scl = [exp(-(i - 1) / ξ) for i in 1:D]
        F.S .*= scl
    end
    xnI .= xnI * F.U * Diagonal(sqrt.(F.S))
    ynI .= Diagonal(sqrt.(F.S)) * F.Vt * ynI
    schg = sign.(real(diag(xnI)))
    xnI .= xnI * Diagonal(schg)
    ynI .= Diagonal(schg) * ynI
    l .= l * xnI
    r .= r * ynI'
    σ .= F.S
    if test
        #check that conditions for WTG have been met by new l, r and σ
        @tensor enviso[i, j, k, l] =
            env[a, b, c, d] * conj(r)[a, i] * l[b, j] * r[c, k] * conj(l)[d, l]
        envtest = TensorKit.TO.tensortrace(
            enviso .* reshape(sqrt.(F.S), (1, :, 1, 1)) .*
            reshape(sqrt.(F.S), (1, 1, 1, :)) ./ reshape(sqrt.(F.S), (:, 1, 1, 1)) ./
            reshape(sqrt.(F.S), (1, 1, :, 1)),
            [1, 2, 3, 2],
        )
        if envtest ./ real(envtest[1, 1]) ≈ I
            nothing
        else
            @warn "left gauge condition was not met"
        end
        envtest = TensorKit.TO.tensortrace(
            enviso ./ reshape(sqrt.(F.S), (1, :, 1, 1)) ./
            reshape(sqrt.(F.S), (1, 1, 1, :)) .* reshape(sqrt.(F.S), (:, 1, 1, 1)) .*
            reshape(sqrt.(F.S), (1, 1, :, 1)),
            [1, 2, 1, 4],
        )
        if envtest ./ real(envtest[1, 1]) ≈ I
            nothing
        else
            @warn "right gauge condition was not met"
        end
    end
    return nothing
end

function optisoalt(cf, left, right, metric, trunc, left0=left, right0=right; maxiter=10)

    #=
                ┌───────────────┐             ┌─────┐
                │               │     v  j◄───┤1   3│◄───i u
                │◄───3 i j 1◄───┤             │     │
                │               │  =          │     │
                │───►2 i'j'4───►│             │     │
                │               │    v' j'───►│4   2├───►i' u'
                └───────────────┘             └─────┘
                                              ┌─────┐
                                     u' i'◄───┤2   3│◄───i u
                                              │     │
                                   =          │     │
                                              │     │
                                     v' j'◄───│4' '1├◄───j v
                                              └─────┘

    =#
    # Convenience
    Y = metric
    L = left
    R = right

    # The following tensor will be useful...
    LR0 = permute(left0 * right0, (1, 2), ())
    # normalize!(LR0)

    # A useful tensor
    P0 = Y * LR0

    u, s, v = tsvd(L * R; trunc=trunc)

    # Get an appropriate identity matrix for the spaces.
    L0 = left0
    R0 = right0

    counter = 0
    δϵ = 1.0
    ϵ = Inf
    print("Optimising isometries: [")
    while !(counter > maxiter || δϵ < 0)
        # while !(counter > maxiter || δϵ < 10 * eps())
        s_old = s

        ϵ_old = ϵ

        # Optimise for v with u fixed
        P = (u ⊗ transpose(R0))' * P0
        B = (u ⊗ transpose(R0))' * Y * (u ⊗ transpose(R0))
        R = (pinv(0.5 * (B + B')) * P)'
        println(space(R))

        ui, si, vi = tsvd(permute(R, (2,), (1,)))
        v = (ui * vi)'

        # v = ui * vi

        # Optimise for u with v fixed
        P = (L0 ⊗ transpose(v))' * P0
        B = (L0 ⊗ transpose(v))' * Y * (L0 ⊗ transpose(v))
        R = (pinv(0.5 * (B + B')) * P)'

        ui, si, vi = tsvd(permute(R, (2,), (1,)))
        u = (ui * vi)'

        # u = ui * vi

        s = s / tr(s)
        println(s)

        # calculate convergence measure
        ϵ = norm(s - s_old)
        # Calculate truncation error (this is slow)
        # ϵ = cf(u * sqrt(s), sqrt(s) * Mb)

        # Calculate convergence measure.
        δϵ = abs(ϵ - ϵ_old)

        print("|")

        counter = counter + 1
    end

    # P = (u ⊗ transpose(R0))' * P0
    # B = (u ⊗ transpose(R0))' * Y * (u ⊗ transpose(R0))
    # R = pinv(0.5 * (B + B')) * P
    #
    u, s, v = tsvd(u * v; trunc=trunc)
    #
    # v = ui * sqrt(si) * vi
    # u = vi' * sqrt(si) * vi * u

    ϵ = cf(u * sqrt(s), sqrt(s) * v)
    # ϵ = cf(u * sqrt(si), sqrt(si) * v)
    @debug "Post-optimisation error: $ϵ"
    # Return new isometries for transformation to the WTG
    if Inf < ϵ
        # @info "Isometries could not be improved from an infidelity of $ϵ0 after $counter iterations"
        # return u0, s0, Mb0
    else
        @info "Isometries optimised to an infidelity of: $ϵ from $Inf after $counter iterations."
        return u, s, v
    end
end

function optsingleiso(id_Mb, Y, Y_LR)

    # temp = id_Mb' * Y

    # slow bit
    g = id_Mb' * Y * id_Mb

    # See docs for `pinv`
    rtol = sqrt(eps(real(float(oneunit(eltype(g))))))
    #
    j = id_Mb' * Y_LR

    Ma_max = pinv(0.5 * (g + g'); rtol=rtol) * j

    # ϵ_j = scalar(Ma_max' * j)
    # ϵ = scalar(Ma_max' * g * Ma_max) - ϵ_j - ϵ_j' + ϵ0

    Ma_max = permute(Ma_max, (1,), (2,)) # , abs(ϵ)

    return Ma_max
end

function eat(L, R, g, trunc; ptol=1e-8)
    # L = TensorMap(L.data, codomain(L), fuse(*(domain(L)...)))
    # R = TensorMap(R.data, fuse(*(codomain(R)...)), domain(R))

    @tensoropt G[-1 -2; -3 -4] :=
        L[3; -3] * R[-4; 4] * g[1 2; 3 4] * (L')[-1; 1] * (R')[2; -2]

    Ga, _, Gb = tsvd(G, (3, 1), (4, 2); trunc=truncdim(1))

    Gap = permute(Ga, (2,), (1, 3))
    Ga = TensorMap(Gap.data, codomain(Gap), fuse(*(domain(Gap)...)))
    Gbp = permute(Gb, (2,), (1, 3))
    Gb = TensorMap(Gbp.data, codomain(Gbp), fuse(*(domain(Gbp)...)))

    λa, Ua = eigh(Ga)
    λb, Ub = eigh(Gb)

    Wa, λ, Wb = tsvd(sqrt(λa) * Ua * Ub * sqrt(λb); trunc=trunc)

    try
        Ma = L * Ua' * pinv(sqrt(λa); rtol=ptol) * Wa * sqrt(λ)
        Mb = sqrt(λ) * Wb * pinv(sqrt(λb); rtol=ptol) * Ub' * R
        @debug "EAT success"
        return Ma, Mb
    catch
        @debug "EAT failed"
        return L, R
    end
end
