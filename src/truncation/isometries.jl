# function (alg::IsometryOptimisation)(pepo::AbstractPEPO, environment::AbstractEnvironment)
#     bonds = getbonds(pepo)
#     return alg.(pepo, environment, bonds)
# end
function (alg::IsometryOptimisation)(pepo, environment, bond)
    D = domain.(getstate(pepo), 1)[left(bond)]

    tmetric = metric(pepo, environment, bond)
    left0, right0 = isometryinit(tmetric)

    u, s_f64, v = metricoptimise(left0, right0, tmetric, truncspace(D); rankreduced=false)

    s = similar(s_f64, eltype(u))

    copy!(s, s_f64)

    return TruncationIsometries(u, s, v), tmetric
end

# Only works in sequence
function (alg::TensorOptimisation)(pepo::AbstractPEPO, environment, bond)
    D = domain.(getstate(pepo), 1)[left(bond)]

    left0, right0 = rankreduce!(pepo)

    tmetric = metric(pepo, environment, bond)

    u, s_f64, v = metricoptimise(left0, right0, tmetric, truncspace(D))

    s = similar(s_f64, scalartype(pepo))

    copy!(s, s_f64)

    return TruncationIsometries(u, s, v), tmetric
end

function isometries(pepo, environment; kwargs...)
    axisbonds = getbonds(pepo)[1]
    iso = isometries.(pepo, environment, axisbonds; kwargs...)
    return iso
end

function isometries(pepo, ::SimpleUpdateState, bond; truncalg::SU)
    D = domain.(getstate(pepo), 1)[bond.left]

    left, right = rankreduce!(pepo)

    u, s_f64, v = tsvd(left * right; trunc=truncspace(D))

    s = similar(s_f64, scalartype(pepo))

    copy!(s, s_f64)

    return TruncationIsometries(u, s, v)
end

function isometries(pepo, environment, bond; truncalg::Union{FET,NTU})
    xi = truncalg.ξ
    gauge = truncalg.gauge

    # opt = truncalg.opt
    # opt = TensorOptimisation()
    opt = IsometryOptimisation()

    iso, metric = opt(pepo, environment, bond)

    wtg!(iso, metric, Val(gauge))
    if xi < Inf
        copy!(iso.s, rescale(iso.s, xi))
        wtg!(iso, metric, Val(gauge))
    end

    return iso
end

# function isometries(pepo, environment, bond; truncalg=NTU()::NTU)
#     xi = truncalg.ξ
#     gauge = truncalg.gauge
#
#     D = domain.(getstate(pepo), 1)[bond.left]
#
#     left0, right0 = rankreduce!(pepo)
#     # opt = truncalg.opt
#     iso = cgoptimize(
#         left0, right0, pepo, bond, truncspace(D), Val(dim(codomain(left0))), Val(dim(D))
#     )
#
#     # wtg!(iso, metric, Val(gauge))
#     # if xi < Inf
#     #     copy!(iso.s, rescale(iso.s, xi))
#     #     wtg!(iso, metric, Val(gauge))
#     # end
#
#     return iso
# end
# function rankreduce!(pepo::AbstractPEPO, bond)
#     left_site, right_site = pepo[bond]
#     left = leftrankreduce!(left_site)
#     right = rightrankreduce!(right_site)
#     return left, right
# end

# For 1x1 unitcell only
# function _quicktruncate!(pepo::GlobalUpdatablePEPO, environment)
#     bond = getbonds(pepo)[1, 1]
#     site = getevolved(pepo)[1, 1]
#
#     left_site = copy(site)
#     right_site = copy(site)
#
#     # left0, right0 = _rankreduce!(left_site, right_site)
#
#     tmetric = _metric(site, site, environment.tensors, bond)
#     left0, right0 = isometryinit(tmetric)
#
#     d = domain.(getstate(pepo), 1)[bond.left]
#
#     u, s_f64, v = metricoptimise(left0, right0, tmetric, truncspace(d); rankreduced=false)
#
#     s = similar(s_f64, eltype(u))
#
#     copy!(s, s_f64)
#
#     @info "" normalize(s)
#
#     iso = TruncationIsometries(u, s, v)
#
#     truncate!(pepo, iso)
#
#     return pepo
# end

function quicktruncate!(pepo_tensor, truncators)
    u = leftrankreduce!(pepo) # P -> Q * U
    v = rightrankreduce!(pepo) # Q -> V * W

    q_left = similar(pepo)
    q_right = similar(pepo)

    tensor_truncate!(q_left, pepo, u)
    tensor_truncate!(q_right, v, pepo)

    return metric = _metric(q_left, q_right, truncators)
end

function quicktruncate!(pepo::GlobalUpdatablePEPO, env; gauge=true, mcf=false, xi=Inf)
    only_bond = getbonds(pepo)[1, 1]

    src_pepo = getevolved(pepo)
    dst_pepo = getstate(pepo)

    src_site = src_pepo[1, 1]
    dst_site = dst_pepo[1, 1]

    tmetric = metric(pepo, env.tensors, only_bond)

    #=

    if !(validate!(metric)) || tmetric = metric(pepo, env.tensors, only_bond)
    =#

    # Initial guess for isometries (identities)
    left0, right0 = isometryinit(tmetric)

    d = domain(dst_site, 1)

    u, s_f64, v, tmetric = metricoptimise(
        left0, right0, tmetric, truncspace(d); rankreduced=false
    )

    s = similar(s_f64, eltype(u))

    copy!(s, s_f64)

    @debug "Normalized weights: " normalize(s)

    iso = TruncationIsometries(u, s, v)
    wtg!(iso, tmetric, Val(gauge))
    if xi < Inf
        copy!(iso.s, rescale(iso.s, xi))
        wtg!(iso, tmetric, Val(gauge))
    end
    ##

    setweights!(pepo, normalize(iso.s), only_bond)

    temp = partialtruncate(src_site, iso)
    rv = partialtruncate(iso, temp)

    hermitianize!(rv)

    if mcf
        minimalcanonicalform!(rv)
    end

    copy!(dst_site, rv)

    return pepo
end

function _sequentialtruncate(pepo::GlobalUpdatablePEPO, env; xi=Inf, gauge=true)
    ev_pepo = getevolved(pepo)

    inter_state = deepcopy(ev_pepo)

    left0s = leftrankreduce!.(ev_pepo) # P -> Q * U
    right0s = rightrankreduce!.(ev_pepo) # Q -> V * W

    q_lefts = similar.(ev_pepo)
    q_rights = similar.(ev_pepo)

    tensor_truncate!.(q_lefts, ev_pepo, left0s)
    tensor_truncate!.(q_rights, right0s, ev_pepo)

    bonds = getbonds(pepo)

    metrics = _metric.(q_lefts, q_rights, Ref(env.tensors), bonds)

    for col in eachcol(bonds)
        for bond in col
            D = domain.(getstate(pepo), 1)[bond.left]

            u, s_f64, v = metricoptimise(
                left0s[bond.left], right0s[bond.right], metrics[bond.left], truncspace(D)
            )

            s = similar(s_f64, scalartype(pepo))
            copy!(s, s_f64)
            iso = TruncationIsometries(u, s, v)

            # wtg!(iso, tmetric, Val(gauge))
            if xi < Inf
                copy!(iso.s, rescale(iso.s, xi))
                # wtg!(iso, tmetric, Val(gauge))
            end
            ##

            setweights!(pepo, normalize(iso.s), bond)

            left_site, right_site = inter_state[bond]

            #  = A = ->  = A -
            temp = partialtruncate(left_site, iso)
            #  = B = ->  - B =
            inter_state[bond.right] = partialtruncate(iso, temp)
        end
    end

    copy!.(getstate(pepo), inter_state)

    return pepo
end

function sequentialtruncate(pepo::GlobalUpdatablePEPO, env; xi=Inf, gauge=true)
    bonds = getbonds(pepo)
    lx, _ = size(bonds)

    if lx == 1
        quicktruncate!(pepo, env)
    else
        inter_state = copy(getevolved(pepo))

        env = truncators(inter_state, env)

        for col in eachcol(bonds)
            for bond in col
                left_site, right_site = inter_state[bond]

                ### put this in a function
                left0, right0 = _rankreduce!(left_site, right_site)

                tmetric = _metric(left_site, right_site, env.tensors, bond)

                D = domain.(getstate(pepo), 1)[bond.left]

                u, s_f64, v, tmetric = metricoptimise(left0, right0, tmetric, truncspace(D))

                s = similar(s_f64, scalartype(pepo))

                copy!(s, s_f64)

                iso = TruncationIsometries(u, s, v)

                wtg!(iso, tmetric, Val(gauge))
                if xi < Inf
                    copy!(iso.s, rescale(iso.s, xi))
                    wtg!(iso, tmetric, Val(gauge))
                end
                ##

                setweights!(pepo, normalize(iso.s), bond)

                #  = A = ->  = A -
                inter_state[bond.left] = partialtruncate(left_site, iso)
                #  = B = ->  - B =
                inter_state[bond.right] = partialtruncate(iso, right_site)

                # Tensors have changed bond dimensions so need to construct new env
                env = truncators(inter_state, env)
            end
        end
        # Recalculate the enviroment (intermediate renormalization step)
        copy!.(getstate(pepo), inter_state)

        hermitianize!.(getstate(pepo))
    end
    return pepo
end

function rankreduce!(pepo::LocalUpdatablePEPO)
    return _rankreduce!(evolvedpair(pepo)[1], evolvedpair(pepo)[2])
end
rankreduce!(mat::AbstractLattice, bond) = _rankreduce!(mat[bond]...)

function _rankreduce!(left_site, right_site)
    left = leftrankreduce!(left_site)
    right = rightrankreduce!(right_site)
    return left, right
end

function leftrankreduce!(p::AbsTen{2,4})
    Q, R = leftorth(p, (1, 2, 4, 5, 6), (3,))
    permute!(p, Q, ((1, 2), (6, 3, 4, 5)))
    return R
end
function rightrankreduce!(p::AbsTen{2,4})
    L, Q = rightorth(p, (5,), (1, 2, 3, 4, 6))
    permute!(p, Q, ((2, 3), (4, 5, 1, 6)))
    return L
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

function metricoptimise(left, right, metric::AbstractLattice, trunc; kwargs...)
    T = promote_type(eltype(left), eltype(right), scalartype(metric))

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
    AB = permute(A * B, (1, 2), ())
    # normalize!(AB)
    # LR = normalize(LR)
    return norm(1 - scalar(AB' * Y * LR) - scalar(LR' * Y * AB) + scalar(AB' * Y * AB))
end

# Optimise the isometries `left` and `right` with respect to `metric`.
function metricoptimise(
    left, right, metric::AbsTen{2,2}, trunc; rankreduced=true, kwargs...
)

    # normalize!(left)
    # normalize!(right)
    # Convenience
    L = left
    R = right

    if rankreduced
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
        return u, s, v
    end

    # Obtain the normalisation constant.

    #Symmetrise and normalise the truncation metric.
    Y = (0.5 * (Y + Y')) / F0

    Y, err = posapprox(Y)

    # @info "Positive approximation error: $err"

    # Construct the cost function to measure convergence.
    # cf = (Ma, Mb) -> metric_costfunction(Ma, Mb, Y, left, right, 1)
    cf = (x, y) -> new_costfunc(x, y, Y, LR0)

    F0 = scalar(LR0' * Y * LR0)
    @debug "Minimal tolerance: $(cf(left , right)). Normalisation check: $F0"

    # Proceed with metric optimisation.
    u, s, v = optiso(cf, left, right, Y, trunc, left0, right0; kwargs...)

    return u, s, v, permute(Y, (4, 1), (3, 2))
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
