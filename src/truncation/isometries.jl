function (alg::IsometryOptimisation)(pepo::AbstractPEPO, environment::AbstractEnvironment)
    bonds = getbonds(pepo)
    return alg.(pepo, environment, bonds)
end
function (alg::IsometryOptimisation)(pepo, environment, bond)
    D = domain.(getstate(pepo), 1)[left(bond)]

    tmetric = metric(pepo, environment, bond)
    left0, right0 = isometryinit(tmetric)
    u, s_f64, v = metricoptimise(left0, right0, tmetric, truncspace(D))

    s = similar(s_f64, eltype(u))

    copy!(s, s_f64)

    return TruncationIsometries(u, s, v), tmetric
end

# Only works in sequence
function (alg::TensorOptimisation)(pepo::AbstractPEPO, environment, bond)
    D = domain.(getstate(pepo), 1)[left(bond)]

    p_left, p_right = pepo[bond]

    left0 = _leftrankreduce!(p_left)
    right0 = _rightrankreduce!(p_right)

    tmetric = metric(pepo, environment, bond)

    u, s_f64, v = metricoptimise(left0, right0, tmetric, truncspace(D))

    s = similar(s_f64, numbertype(pepo))

    copy!(s, s_f64)

    return TruncationIsometries(u, s, v), tmetric
end

function _leftrankreduce!(p::AbsTen{2,4})
    Q, R = leftorth(p, (1, 2, 4, 5, 6), (3,))
    permute!(p, Q, (1, 2), (6, 3, 4, 5))
    return R
end
function _rightrankreduce!(p::AbsTen{2,4})
    L, Q = rightorth(p, (5,), (1, 2, 3, 4, 6))
    permute!(p, Q, (2, 3), (4, 5, 1, 6))
    return L
end

function constructG(L, g, R)
    @tensoropt G[1 2; 3 4] := L[3; 3] * R[4; 4] * g[1 2; 3 4] * (L')[1; 1] * (R')[2; 2]
end

function isometryinit(metric::AbstractOnLattice)
    leftright = isometryinit.(metric)

    left = getindex.(leftright, 1)
    right = getindex.(leftright, 2)

    return left, right
end

# Isometry update requires one to contruct initial guesses for L & R.
function isometryinit(metric::AbsTen{2,2})
    η_left = domain(metric)[1]      # Upper left index
    η_right = codomain(metric)[1]   # Upper right index

    left = id(η_left)
    right = id(η_right)

    # left = TensorMap(rand, eltype(metric), η_left, η_left)
    # right = TensorMap(rand, eltype(metric), η_right, η_right)

    return left, right
end

function metricoptimise(left, right, metric::AbstractOnLattice, trunc; kwargs...)
    T = promote_type(eltype(left), eltype(right), numbertype(metric))

    nx, ny = size(metric)

    u = Matrix{T}(undef, nx, ny)
    v = Matrix{T}(undef, nx, ny)
    s = Matrix{T}(undef, nx, ny)

    # TODO: mulithreading
    for y in axes(metric, 2)
        for x in axes(metric, 1)
            u[x, y], s[x, y], v[mod(x + 1, 1:nx), y] = metricoptimise(
                left[x, y], right[x, y], metric[x, y], trunc; kwargs...
            )
        end
    end
    return u, s, v
end

function new_costfunc(A, B, Y, LR)
    AB = normalize(permute(A * B, (1, 2), ()))
    LR = normalize(LR)
    return norm(1 - scalar(AB' * Y * LR) - scalar(LR' * Y * AB) + scalar(AB' * Y * AB))
end

# Optimise the isometries `left` and `right` with respect to `metric`.
function metricoptimise(left, right, metric::AbsTen{2,2}, trunc; kwargs...)
    # normalize!(left)
    # normalize!(right)
    # Convenience
    L = left
    R = right

    LR = normalize(permute(L * R, (1, 2), ()))

    # Put in the appropriate matrix form for the Ritz procedure. 
    Y = permute(metric, (2, 4), (3, 1))

    # n = Y.data[1, 1]
    # rmul!(Y, 1 / n)
    # Check if the metric is hermitian, If not, use SU

    F0 = scalar(LR' * Y * LR)

    hermiticity = norm(Y - Y')

    htol = 1e-10

    hermiticity > htol &&
        @warn "Metric not hermitian! non-hermiticity:  $hermiticity, tolerance $htol"

    @debug "Metric normalisation (F0): $F0"

    if hermiticity > htol
        @info "Metric not hermitian: using SU"
        u, s, v, err = tsvd(left * right; trunc=trunc)
        @info "Truncation error: $err"
        return u, s, v
    end

    # Obtain the normalisation constant.

    #Symmetrise and normalise the truncation metric.
    Y = (0.5 * (Y + Y')) / F0

    # Construct the cost function to measure convergence.
    # cf = (Ma, Mb) -> metric_costfunction(Ma, Mb, Y, left, right, 1)
    cf = (x, y) -> new_costfunc(x, y, Y, LR)

    F0 = scalar(LR' * Y * LR)
    @debug "Minimal tolerance: $(cf(L , R)). Normalisation check: $F0"

    # Proceed with metric optimisation.
    u, s, v = optiso(cf, L, R, Y, trunc; kwargs...)

    return u, s, v
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

function optiso(cf, left, right, metric, trunc; EAT=false, maxiter=100)
    # Convenience
    Y = metric
    L = left
    R = right

    # The following tensor will be useful...
    LR = normalize(permute(L * R, (1, 2), ()))
    Y_LR = Y * LR

    u_su, s_su, v_su = tsvd(L * R; trunc=trunc)
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
    while !(counter > maxiter || δϵ < 1e-14)
        ϵ_old = ϵ

        # Optimise for Ma with Mb fixed.
        Ma = u * s
        Ma_max = optsingleiso(id_a ⊗ transpose(Mb), Y, Y_LR)
        Ma, s, v = tsvd(Ma_max * Mb; trunc=trunc)

        # Optimise for Mb with Ma fixed.
        Mb = s * v
        Mb_max = optsingleiso(Ma ⊗ id_b, Y, Y_LR)
        u, s, Mb = tsvd(Ma * Mb_max; trunc=trunc)

        # Calculate truncation error.
        ϵ = cf(u * sqrt(s), sqrt(s) * Mb)

        # Calculate convergence measure.
        δϵ = abs(ϵ - ϵ_old)

        print("|")

        counter = counter + 1
    end
    print("]\n")

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

    g = id_Mb' * Y * id_Mb
    j = id_Mb' * Y_LR

    Ma_max = pinv(g; rtol=1e-10) * j

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
