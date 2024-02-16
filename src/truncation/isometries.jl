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
    iso = metricoptimise(left0, right0, tmetric, truncspace(D); rankreduce=rankreduce, gauge=gauge, xi=xi)

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

    u, s, v, _ = tsvd(left * right; trunc=truncspace(D))

    iso = TruncationIsometries(pepo, u, s, v)

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
    Q, R = leftorth(p, (1, 2, 4, 5, 6), (3,))
    P = permute(Q, ((1, 2), (6, 3, 4, 5)))
    return P, R
end
function rightrankreduce(p::AbsTen{2,4})
    L, Q = rightorth(p, (5,), (1, 2, 3, 4, 6))
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

    # left = id(η_left)
    # right = id(η_right)

    left = TensorMap(randn, eltype(metric), η_left, η_left)
    right = TensorMap(randn, eltype(metric), η_right, η_right)

    return left, right
end

function new_costfunc(A, B, Y, LR)
    AB = permute(A * B, (1, 2), ())
    # normalize!(AB)
    # LR = normalize(LR)
    return norm(1 - scalar(AB' * Y * LR) - scalar(LR' * Y * AB) + scalar(AB' * Y * AB))
end

# Optimise the isometries `left` and `right` with respect to `metric`.
function metricoptimise(
    left, right, metric::AbsTen{2,2}, trunc; rankreduce, gauge, xi, kwargs...
)

    # normalize!(left)
    # normalize!(right)
    # Convenience
    L = left
    R = right

    if rankreduce
        left0 = left
        right0 = right
    else
        left0 = one(left)
        right0 = one(right)
    end

    LR0 = permute(left0 * right0, (1, 2), ())
    # normalize!(LR0)

    # Put in the appropriate matrix form for the Ritz procedure. 
    Y = permute(metric, (2, 4), (3, 1))

    # n = Y.data[1, 1]
    # rmul!(Y, 1 / n)
    # Check if the metric is hermitian, If not, use SU

    F0 = scalar(LR0' * Y * LR0)

    hermiticity = norm(Y - Y')

    htol = 1e-10
    # htol = 0.0 # force SU
    # htol = Inf # force update

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
    @debug "Minimal tolerance: $(cf(left , right)). Normalisation check: $F0"

    # Proceed with metric optimisation.
    u, s, v = optiso(cf, left, right, Y, trunc, left0, right0; kwargs...)

    iso = TruncationIsometries(metric, u, s, v)

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

    u_su, s_su, v_su = tsvd(left0 * right0; trunc=trunc)
    @debug "Simple update error: $(cf(u_su * sqrt(s_su), sqrt(s_su) * v_su))"

    # Perform EAT if we want to.
    if EAT
        Ma0, Mb0 = eat(L, R, metric, trunc)
    else
        Ma0, Mb0 = L, R
    end

    # Initialise isometries for optimisation.
    u0, s0, Mb0 = tsvd(Ma0 * Mb0; trunc=trunc)

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
    while !(counter > maxiter || δϵ < 10 * eps())
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
        ϵ = cf(u * sqrt(s), sqrt(s) * Mb)

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
        @info "Isometries could not be improved from a tolerance of $ϵ0 after $counter iterations"
        return u0, s0, Mb0
    else
        @info "Isometries optimised to a tolerance of: $ϵ from $ϵ0 after $counter iterations."
        return u, s, Mb
    end
end

function optsingleiso(id_Mb, Y, Y_LR)
    # temp = id_Mb' * Y

    # slow bit
    g = id_Mb' * Y * id_Mb
    #
    j = id_Mb' * Y_LR

    Ma_max = pinv(g; rtol=1e-7) * j

    # ϵ_j = scalar(Ma_max' * j)
    # ϵ = scalar(Ma_max' * g * Ma_max) - ϵ_j - ϵ_j' + ϵ0

    return permute(Ma_max, (1,), (2,)) # , abs(ϵ)
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
