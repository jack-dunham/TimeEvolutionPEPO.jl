# function autodiffoptimise(left, right, metric::AbsTen{2,2}; trunc, rankreduce, xi, optalg)
#     gauge = optalg.gauge
#
#     if rankreduce
#         left0 = left
#         right0 = right
#     else
#         left0 = id(codomain(left))
#         right0 = id(domain(right))
#     end
#
#     LR0 = permute(left0 * right0, (1, 2), ())
#
#     Y = permute(metric, (2, 4), (3, 1))
#
#     F0 = scalar(LR0' * Y * LR0)
#
#     stablizemetric!(Y, F0)
#
#     infid = (u, v) -> 1.0 - _fidelity(u, v, Y, LR0; trunc=trunc)
#
#     # optimizer = LBFGS(4; maxiter=1000, gradtol=1e-4, verbosity=1)
#     optimizer = LBFGS(4; gradtol=1e-8, maxiter=200, verbosity=1)
#     # optimizer = ConjugateGradient(; maxiter=200, gradtol=1e-8, verbosity=1)
#
#     # Initialize
#     u, s2, v = tsvd!(left0 * right0; trunc=trunc)
#
#     ut = u * sqrt(s2)
#     vt = sqrt(s2) * v
#
#     s1old = id(domain(ut))
#     s2old = id(domain(ut))
#
#     numiter = 0
#
#     f1old = Inf
#
#     while numiter < 10
#         numiter += 1
#
#         x0 = ITC.randnt!(similar(ut))
#
#         Rm, _ = optimize(ut * s2, optimizer; inner=(_, x, y) -> real(dot(x, y))) do R0
#             rv = withgradient(u -> infid(u, vt), R0)
#             return rv.val, only(rv.grad)
#         end
#
#         u, s1, v = safetsvd!(Rm; trunc=trunc)
#
#         # mul!(ut, u, v)
#         mul!(ut, u, s1)
#
#         x0 = ITC.randnt!(similar(vt))
#
#         Rm, f1, _ = optimize(s1 * vt, optimizer; inner=(_, x, y) -> real(dot(x, y))) do R0
#             rv = withgradient(v -> infid(ut, v), R0)
#             return rv.val, only(rv.grad)
#         end
#
#         u, s2, v = safetsvd!(Rm; trunc=trunc)
#
#         # mul!(vt, u, v)
#         mul!(vt, s2, v)
#
#         ϵ1 = norm(normalize(s1) - normalize(s1old))
#         ϵ2 = norm(normalize(s2) - normalize(s2old))
#
#         copy!(s1old, s1)
#         copy!(s2old, s2)
#
#         ϵ = max(ϵ1, ϵ2)
#
#         ϵ = (f1 - f1old) / f1
#
#         f1old = f1
#
#         println(ϵ)
#
#         ϵ < 1e-3 && break
#     end
#
#     # x0 = ITC.randnt!(similar(ut))
#     # #
#     # Rm, _ = optimize(ut * s2, optimizer; inner=(_, x, y) -> real(dot(x, y))) do R0
#     #     rv = withgradient(u -> infid(u, vt), R0)
#     #     return rv.val, only(rv.grad)
#     # end
#     #
#     # u, s, v = safetsvd!(Rm)
#     #
#     # maxnormalize!(s)
#     #
#     # println(s)
#
#
#     # l = u * sqrt(s) * v
#     # r = vt' * v' * sqrt(s) * v
#     #
#     # copy!(vt, r')
#     # iso = TruncationIsometries(l, s, vt)
#
#     u, s, v = safetsvd!(ut * vt; trunc=trunc)
#     l = u * sqrt(s)
#     r = sqrt(s) * v
#
#     maxnormalize!(s)
#
#
#     println(s)
#
#
#     iso = TruncationIsometries(l, s, r)
#
#     permute!(metric, Y, ((4, 1), (3, 2)))
#
#     fixgauge!(iso, metric, gauge)
#     #
#     # if xi < Inf
#     #     rescale!(iso, xi)
#     #     fixgauge!(iso, metric, gauge)
#     # end
#
#     return iso, 0
# end
function autodiffoptimise(
    left, right, pair::Pair, environment, bond::Bond; trunc, rankreduce, xi, optalg
)
    gauge = optalg.gauge

    if rankreduce
        left0 = left
        right0 = right
    else
        left0 = id(codomain(left))
        right0 = id(domain(right))
    end

    infid = ntu_total_infidelity(pair, environment, bond; trunc=trunc)
    # optimizer = LBFGS(4; maxiter=1000, gradtol=1e-4, verbosity=1)
    optimizer = LBFGS(4; gradtol=1e-8, maxiter=200, verbosity=2)
    # optimizer = GradientDescent(; maxiter=200, gradtol=1e-8, verbosity=1)
    # optimizer = ConjugateGradient(; maxiter=200, gradtol=1e-4, verbosity=2)

    # Initialize
    u, s2, v = tsvd!(left0 * right0; trunc=trunc)

    ut = u * sqrt(s2)
    vt = sqrt(s2) * v

    x0 = ITC.randn!(similar(ut * vt))

    D = dim(codomain(s2))

    sv = diagm([exp10(-i) for i in 0:(D - 1)])

    s2 = TensorMap(sv, codomain(s2), domain(s2))

    x0 = convert(typeof(x0), u * s2 * v)

    # Compile
    @time withgradient(uv -> infid(uv), x0)

    maxvec = x0

    rv, _ = optimize(x0, optimizer; inner=(_, x, y) -> real(dot(x, y))) do R0
        rv = withgradient(uv -> infid(uv), R0)
        return rv.val, only(rv.grad)
    end
    copy!(maxvec, rv)

    u, s, v = safetsvd!(maxvec; trunc=trunc)
    l = u * sqrt(s)
    r = sqrt(s) * v

    maxnormalize!(s)

    iso = TruncationIsometries(l, s, r)

    # yiso = ntu_total_wtg(pair, environment, bond, iso)
    #
    # wtg2!(iso.u, iso.s, iso.v, yiso)

    return iso, 0
end

function autodiffoptimise(left, right, metric::AbsTen{2,2}; trunc, rankreduce, xi, optalg)
    gauge = optalg.gauge

    if rankreduce
        left0 = left
        right0 = right
    else
        left0 = id(codomain(left))
        right0 = id(domain(right))
    end

    LR0 = permute(left0 * right0, (1, 2), ())

    Y = permute(metric, (2, 4), (3, 1))

    F0 = scalar(LR0' * Y * LR0)

    stablizemetric!(Y, F0)

    infid = uv -> 1.0 - _fidelity(uv, Y, LR0; trunc=trunc)

    # optimizer = LBFGS(4; maxiter=1000, gradtol=1e-4, verbosity=1)
    optimizer = LBFGS(4; gradtol=1e-8, maxiter=200, verbosity=1)
    # optimizer = GradientDescent(; maxiter=200, gradtol=1e-8, verbosity=1)
    # optimizer = ConjugateGradient(; maxiter=200, gradtol=1e-8, verbosity=1)

    # Initialize
    u, s2, v = tsvd!(left0 * right0; trunc=trunc)

    ut = u * sqrt(s2)
    vt = sqrt(s2) * v

    s1old = id(domain(ut))
    s2old = id(domain(ut))

    numiter = 0

    f1old = Inf

    x0 = ITC.randn!(similar(ut * vt))

    D = dim(codomain(s2))

    sv = diagm([exp10(-i) for i in 0:(D - 1)])

    s2 = TensorMap(sv, codomain(s2), domain(s2))

    x0 = convert(typeof(x0), u * s2 * v)

    Rm, _ = optimize(x0, optimizer; inner=(_, x, y) -> real(dot(x, y))) do R0
        rv = withgradient(uv -> infid(uv), R0)
        return rv.val, only(rv.grad)
    end

    u, s, v = safetsvd!(Rm; trunc=trunc)
    l = u * sqrt(s)
    r = sqrt(s) * v

    maxnormalize!(s)

    iso = TruncationIsometries(l, s, r)

    permute!(metric, Y, ((4, 1), (3, 2)))

    fixgauge!(iso, metric, gauge)
    #
    # if xi < Inf
    #     rescale!(iso, xi)
    #     fixgauge!(iso, metric, gauge)
    # end

    return iso, 0
end
function my_inner(_, x, y)
    x1, x2 = x
    y1, y2 = y

    xx = dot(x[1], y[1])
    yy = dot(x[2], y[2])

    return real(xx + yy)
end

function my_retract(x, eta, alpha)
    x1, x2 = x

    y1 = similar(x1)
    y2 = similar(x2)

    y1 = x1 + alpha * eta[1]
    y2 = x2 + alpha * eta[2]

    return (y1, y2)
end

function my_scale!(x, beta)
    rmul!(x[1], beta)
    rmul!(x[2], beta)
    return x
end

function my_add!(x, y, beta)
    LinearAlgebra.axpy!(beta, x[1], y[1])
    LinearAlgebra.axpy!(beta, x[2], y[2])
    return x
end

function _fidelity(AB, Y, LR; trunc=trunc)
    u, s, v = tsvd(AB; trunc=trunc)

    AB = permute(u * s * v, (1, 2), ())

    f = tr(AB' * Y * LR) * tr(LR' * Y * AB) / tr(AB' * Y * AB)

    # @tensor f[x;y] := AB[x y]
    ignore_derivatives() do
        @debug "imaginary part of fidelity: $(imag(f))"
    end

    return real(f)
end

function isometries(pair, environment, bond::Bond, truncalg::Union{FET,NTU}; kwargs...)
    return _generic_environment_isometries(pair, environment, bond, truncalg; kwargs...)
end
function parallelisometries(
    buffer, environment, bond::Bond, truncalg::Union{FET,NTU}; kwargs...
)
    return parallelisometries(
        buffer[bond], environment, bond, truncalg; parallel=true, kwargs...
    )
end
function parallelisometries(
    pair::Pair, environment, bond::Bond, truncalg::Union{FET,NTU}; kwargs...
)
    return _generic_environment_isometries(
        pair, environment, bond, truncalg; parallel=true, kwargs...
    )
end
function _generic_environment_isometries(
    pair, environment, bond, truncalg; parallel=false, kwargs...
)
    rankreduce = truncalg.rankreduce
    optalg = truncalg.optalg
    xi = truncalg.zeta

    iso, s, infid = single_bond_metric_isometries!(
        pair,
        environment,
        bond;
        parallel=parallel,
        optalg=optalg,
        doreduction=rankreduce,
        zeta=zeta,
        kwargs...,
    )

    return iso, s, infid
end

metric(pair::Pair, environment::Function, ::Bond) = environment(pair)

function metric_isometries_from_simple_update!(
    initpair, initiso, environment, bond; D, optkwargs...
)
    (u0, v0) = initiso
    pair = initpair

    truncmetric = metric(pair, environment, bond)

    iso, infid = metricoptimise(
        u0, v0, truncmetric; trunc=truncspace(D), doreduction=true, optkwargs...
    )
    return (iso.u, iso.v), iso.s, infid
end

function single_bond_metric_isometries!(
    pair, environment, bond; D, doreduction, parallel, optkwargs...
)
    if doreduction
        if parallel
            pair, (u0, v0) = rankreduce(pair)
        else
            pair, (u0, v0) = rankreduce!(pair)
        end
    end

    truncmetric = metric(pair, environment, bond)

    if !doreduction
        u0, v0 = isometryinit(truncmetric, D)
    end

    iso, infid = metricoptimise(
        u0, v0, truncmetric; trunc=truncspace(D), doreduction=doreduction, optkwargs...
    )
    # iso, infid = autodiffoptimise(
    #     u0, v0, truncmetric; trunc=truncspace(D), rankreduce=doreduction, optkwargs...
    # )
    # iso, infid = autodiffoptimise(
    #     u0, v0, pair, environment, bond; trunc=truncspace(D), rankreduce=doreduction, optkwargs...
    # )

    if parallel && doreduction
        copy!(iso.u, pinv(u0) * iso.u)
        copy!(iso.v, iso.v * pinv(v0))
    end

    return (iso.u, iso.v), iso.s, infid
end

function allisometries(pepo, truncator, truncalg; kwargs...)
    buffer = currentbuffer(pepo)

    infidelity = similar(buffer, Float64)

    xtrunc, ytrunc = truncator

    xiso, xvals, _ = axialisometries(buffer, xtrunc, truncalg)

    permbuffer = network(ytrunc)

    yiso, yvals, _ = permutedims(transpose, axialisometries(permbuffer, ytrunc, truncalg))

    isometries = Isometries(xiso, yiso)
    weights = UnitCellBonds(xvals, yvals)

    return isometries, weights
end

function axialisometries(tensors, truncator, truncalg; axis=1, kwargs...)
    lattice = currentbuffer(pepo)

    infidelity = similar(lattice, Float64)

    bonds = uniquebonds(tensors; axis=axis)

    data = map(bonds) do bond
        (u, v), s, infid = parallelisometries(tensors, truncator, truncalg, bond; kwargs...)

        infidelity[first(bond)] = infid

        return (u=u, v=v, s=s)
    end

    uiso = map(data -> data.u, data)
    viso = map(data -> data.v, circshift(data, (1, 0)))

    weights = map(data -> data.s, data)

    return AxialIsometries(uiso, viso), weights, infidelity
end

# function absorball!(site, weights, ind; inverse=false)
#
#     d = Bond(ind, ind + CartesianIndex(0, 1))
#     u = prevbond(d)
#
#     l = Bond(ind, ind + CartesianIndex(1, 0))
#     r = prevbond(l)
#
#     if inverse
#         gw = x -> inv(sqrt(weights[x]))
#     else
#         gw = x -> sqrt(weights[x])
#     end
#
#     sd = gw(d)
#     su = gw(u)
#
#     sl = gw(l)
#     sr = gw(r)
#
#     @tensor begin
#         site[k b; ee ss ww nn] = deepcopy(site)[k b; e s w n] * sd[s; ss]
#             su[nn; n] * sl[e; ee] * sr[ww; w]
#     end
#
#     return site
# end

function localfidelity(u, v, P, Q, u0, v0; axis=1)
    f0 = overlap(u0, v0, P, Q, u0, v0; axis=axis)
    f =
        overlap(u, v, P, Q, u0, v0; axis=axis) * overlap(u0, v0, P, Q, u, v; axis=axis) /
        overlap(u, v, P, Q, u, v; axis=axis)
    return real(f / f0)
end

function overlap(ua, va, P, Q, ub, vb; axis)
    if axis == 1
        @tensoropt rv =
            P[k1 b1; ea s1 w1 n1] *
            (P')[eb s1 w1 n1; k1 b1] *
            ua[ea; a] *
            va[a; wa] *
            (ub')[b; eb] *
            (vb')[wb; b] *
            Q[k2 b2; e2 s2 wa n2] *
            (Q')[e2 s2 wb n2; k2 b2]

    elseif axis == 2
        @tensoropt rv =
            P[k1 b1; e1 sa w1 n1] *
            (P')[e1 sb w1 n1; k1 b1] *
            ua[sa; a] *
            va[a; na] *
            (ub')[b; sb] *
            (vb')[nb; b] *
            Q[k2 b2; e2 s2 w2 na] *
            (Q')[e2 s2 w2 nb; k2 b2]
    end

    return rv
end

function rankreduce!(lattice::AbstractUnitCell, bond::Bond; copy=true)
    input = lattice[bond]

    axis = bondaxis(bond)

    if copy
        _, matrices = rankreduce!(input; axis=axis)
    else
        output, matrices = rankreduce(input; axis=axis)
        lattice[bond] = output
    end

    return matrices
end

function rankreduce(tensors::Pair{T,T}; axis=1) where {T<:AbsTen{2,4}}
    uten, vten = tensors

    if axis == 1
        i1, i2 = 1, 3
    elseif axis == 2
        i1, i2 = 2, 4
    end

    P1, R = leftrankreduce(uten, i1)
    L, P2 = rightrankreduce(vten, i2)

    return P1 => P2, R => L
end

function rankreduce!(tensors::Pair{T,T}; axis=1) where {T<:AbsTen{2,4}}
    uten, vten = tensors

    if axis == 1
        i1, i2 = 1, 3
    elseif axis == 2
        i1, i2 = 2, 4
    end

    _, R = leftrankreduce!(uten, i1)
    L, _ = rightrankreduce!(vten, i2)

    return uten => vten, R => L
end

function updatetensors(pair, weights, truncalg::SimpleUpdate, bond)
    absorb_adjacent_weights!(pair, weights, bond; inverse=false)
    newpair, s = simple_update_tensors(pair; axis=bondaxis(bond), zeta=truncalg.zeta)
    absorb_adjacent_weights!(newpair, weights, bond; inverse=true)
    return newpair, s
end
function updatetensors(pair, weights, truncalg::SVDU, bond)
    newpair, s = simple_update_tensors(pair; axis=bondaxis(bond), zeta=truncalg.zeta)
    return newpair, s
end
function evolve_updatetensors(pepo, weights, stepper, truncalg, bond)

    pair = mapreduce(deepcopy, =>, pepo[bond])
    op_pair = stepper[bond]

    absorb_adjacent_weights!(pair, weights, bond; inverse=false)

    newpair, s = evolve_simple_update_tensors(pair, op_pair; axis=bondaxis(bond), zeta=truncalg.zeta)

    absorb_adjacent_weights!(newpair, weights, bond; inverse=true)

    return newpair, s
end

function evolve_simple_update_tensors(pair::Pair{T}, op_pair; axis, zeta=Inf) where {T<:AbsTen{2,4}}

    uten, vten = pair
    u1op, v2op = op_pair

    if axis == 1
        D = domain(uten)[3]

        Q1, R = leftorth(uten, (4, 5, 6), (1, 2, 3))
        L, Q2 = rightorth(vten, (1, 2, 5), (3, 4, 6))
    elseif axis == 2
        D = domain(uten)[4]

        Q1, R = leftorth(uten, (3, 5, 6), (1, 2, 4))
        L, Q2 = rightorth(vten, (1, 2, 6), (3, 4, 5))
    end

    @tensoropt utenp[left ok ob; D η] := u1op[ok ob ik ib; η] * R[left; ik ib D]
    @tensoropt vtenp[D η; right ok ob] := v2op[ok ob ik ib; η] * L[ik ib D; right]


    u, s, v, err1 = tsvd(utenp * vtenp; trunc=truncdim(dim(D)))
    # u, s, v, err1 = tsvd(utenp * vtenp; trunc=truncspace(D))

    # u, s, v, err2 = tsvd!(R * L; trunc=truncspace(D))

    # println(err2)

    maxnormalize!(s)

    (u, v), s = applyweight(u => v, s, xi; verbose=false)

    # Q1p = Q1 * u
    # Q2p = v * Q2

    if axis == 1
        @tensoropt unew[k b; e s w n] := Q1[s w n; left] * u[left k b; e]
        @tensoropt vnew[k b; e s w n] := v[w; right k b] * Q2[right; e s n;] 
    elseif axis == 2
        @tensoropt unew[k b; e s w n] := Q1[e w n; left] * u[left k b; s]
        @tensoropt vnew[k b; e s w n] := v[n; right k b] * Q2[right; e s w;] 
    end

    return unew => vnew, s
end
function simple_update_tensors(pair::Pair{T}; axis, zeta=Inf) where {T<:AbsTen{2,5}}
    uten, vten = pair

    if axis == 1
        D = domain(uten)[3]

        Q1, R = leftorth(uten, (1, 2, 5, 6, 7), (3, 4))
        L, Q2 = rightorth(vten, (5, 6), (1, 2, 3, 4, 7))
    elseif axis == 2
        D = domain(uten)[4]

        Q1, R = leftorth(uten, (1, 2, 3, 6, 7), (4, 5))
        L, Q2 = rightorth(vten, (6, 7), (1, 2, 3, 4, 5))
    end

    u, s, v, err1 = tsvd!(R * L; trunc=truncdim(dim(D)))
    # u, s, v, err2 = tsvd!(R * L; trunc=truncspace(D))


    # println(s)
    maxnormalize!(s)

    (u, v), s = applyweight(u => v, s, xi; verbose=false)

    Q1p = Q1 * u
    Q2p = v * Q2

    if axis == 1
        unew = permute(Q1p, (1, 2), (6, 3, 4, 5))
        vnew = permute(Q2p, (2, 3), (4, 5, 1, 6))

    elseif axis == 2
        unew = permute(Q1p, (1, 2), (3, 6, 4, 5))
        vnew = permute(Q2p, (2, 3), (4, 5, 6, 1))
    end

    return unew => vnew, s
end

function simple_update_tensors(pair::Pair{T}; axis, zeta=Inf) where {T<:AbsTen{2,4}}
    uten, vten = pair

    if axis == 1
        D = domain(uten)[3]

        Q1, R = leftorth(uten, (1, 2, 4, 5, 6), (3,))
        L, Q2 = rightorth(vten, (5,), (1, 2, 3, 4, 6))
    elseif axis == 2
        D = domain(uten)[4]

        Q1, R = leftorth(uten, (1, 2, 3, 5, 6), (4,))
        L, Q2 = rightorth(vten, (6,), (1, 2, 3, 4, 5))
    end

    u, s, v, err1 = tsvd!(R * L;)# trunc=truncdim(dim(D)))
    # u, s, v, err2 = tsvd!(R * L; trunc=truncspace(D))

    # println(err2)

    maxnormalize!(s)

    (u, v), s = applyweight(u => v, s, Inf; verbose=false)

    # Q1p = Q1 * u
    # Q2p = v * Q2

    if axis == 1
        unew = permute(Q1, (1, 2), (6, 3, 4, 5))
        vnew = permute(Q2, (2, 3), (4, 5, 1, 6))

    elseif axis == 2
        unew = permute(Q1, (1, 2), (3, 6, 4, 5))
        vnew = permute(Q2, (2, 3), (4, 5, 6, 1))
    end

    return unew => vnew, u => v
end

function _leftrankreduce(p::AbsTen{2,4}, ind)
    codind = TupleTools.deleteat((1, 2, 3, 4, 5, 6), ind + 2)

    Q, R = try
        leftorth(p, codind, (ind + 2,); alg=Polar())
    catch e
        if e isa AssertionError && e.msg == "V ≅ domain(t)"
            @error """
            potential bad choice of bond dimension sectors; try different sectors, or
            setting `rankreduce=false` if you insist on these sectors
            """
        end
        rethrow()
    end

    perm = TupleTools.insertafter((3, 4, 5), ind - 1, (6,))
    return Q, R, perm
end
function leftrankreduce(P::AbsTen{2,4}, ind)
    Q, R, perm = _leftrankreduce(P, ind)
    P = permute(Q, ((1, 2), perm))
    return P, R
end

function leftrankreduce!(P::AbsTen{2,4}, ind)
    Q, R, perm = _leftrankreduce(P, ind)
    permute!(P, Q, ((1, 2), perm))
    return P, R
end

function _rightrankreduce(p::AbsTen{2,4}, ind)
    domind = TupleTools.deleteat((1, 2, 3, 4, 5, 6), ind + 2)

    L, Q = try
        rightorth(p, (ind + 2,), domind; alg=Polar())
    catch e
        if e isa AssertionError && e.msg == "V ≅ domain(t)"
            @error """
            potential bad choice of bond dimension sectors; try different sectors, or
            setting `rankreduce=false` if you insist on these sectors
            """
        end
        rethrow()
    end

    perm = TupleTools.insertafter((4, 5, 6), ind - 1, (1,))

    return L, Q, perm
end
function rightrankreduce(P::AbsTen{2,4}, ind)
    L, Q, perm = _rightrankreduce(P, ind)
    P = permute(Q, ((2, 3), perm))
    return L, P
end
function rightrankreduce!(P::AbsTen{2,4}, ind)
    L, Q, perm = _rightrankreduce(P, ind)
    permute!(P, Q, ((2, 3), perm))
    return L, P
end

function constructG(L, g, R)
    @tensoropt G[1 2; 3 4] := L[3; 3] * R[4; 4] * g[1 2; 3 4] * (L')[1; 1] * (R')[2; 2]
end

function isometryinit(metric::AbstractUnitCell)
    leftright = isometryinit.(metric)

    left = getindex.(leftright, 1)
    right = getindex.(leftright, 2)

    return left, right
end

# Isometry update requires one to contruct initial guesses for L & R.
function isometryinit(metric::AbsTen{2,2}, D)
    T = storagetype(metric)

    η = codomain(metric)[1]   # Upper right index

    left = id(T, η) #+ 1e-3 * TensorMap(randn, eltype(metric), η, η)
    right = convert(TensorMap, left')

    # normalize!(left)
    # normalize!(right)

    return left, right
end

# function new_costfunc(A, B, Y, LR)
#     AB = permute(A * B, (1, 2), ())
#     # normalize!(AB)
#     # LR = normalize(LR)
#     return norm(1 - scalar(AB' * Y * LR) - scalar(LR' * Y * AB) + scalar(AB' * Y * AB))
# end

function fidelity(A, B, Y, LR)
    AB = permute(A * B, (1, 2), ())

    f = scalar(AB' * Y * LR) * scalar(LR' * Y * AB) / scalar(AB' * Y * AB)

    ignore_derivatives() do
        @debug "imaginary part of fidelity: $(imag(f))"
    end

    return real(f)
end

function cycle_entropy(iso, metric)
    u = iso.u
    s = iso.s
    v = iso.v

    L = (iso.u * sqrt(s)) ⊗ transpose(sqrt(s) * iso.v)
    R = (iso.u * inv(sqrt(s))) ⊗ transpose(inv(sqrt(s)) * iso.v)

    Y = L' * permute(metric, (2, 4), (3, 1)) * R

    D, _ = eigen!(Y)

    vals = diag(convert(Array, D))

    vals /= sum(vals)

    ent = -sum(@. vals * log2(vals))

    @info "Cycle entropy: $(real(ent))"

    return ent
end

# Optimise the isometries `left` and `right` with respect to `metric`.
function metricoptimise(left, right, metric::AbsTen{2,2}; 
                        trunc,
                        doreduction,
                        xi,
                        optalg,
                        posapprox = true,
                        )
    gauge = optalg.gauge

    if doreduction
        left0 = left
        right0 = right
    else
        T = scalartype(metric)

        left0 = id(T, codomain(left))
        right0 = id(T, domain(right))
    end

    #left0 -> u
    #right0 -> v
    LR0 = permute(left0 * right0, (1, 2), ())
    # normalize!(LR0)

    # Put in the appropriate matrix form for the Ritz procedure. 
    Y = permute(metric, (2, 4), (3, 1))

    F0 = scalar(LR0' * Y * LR0)

    hermiticity = norm(Y - Y')

    # htol = 1e-10
    # htol = 0.0 # force SimpleUpdate
    htol = Inf # force update

    @debug "Metric normalisation (F0): $F0"

    if hermiticity > htol
        @warn "Metric not hermitian! inhermiticity: $hermiticity, tolerance: $htol"
        @info "Metric not hermitian: using SimpleUpdate"
        u, s, v, err = tsvd(left * right; trunc=trunc)
        @info "Truncation error: $err"
        return TruncationIsometries(metric, u, s, v)
    end


    stablizemetric!(Y, F0)
    posapprox!(Y)
    # posapprox && posapprox!(Y)

    infidelity = (x, y) -> 1 - fidelity(x, y, Y, LR0)

    F0 = scalar(LR0' * Y * LR0)
    @debug "Minimal infidelity: $(infidelity(left , right)). Normalisation check: $F0"

    # Proceed with metric optimisation.
    iso = isometryoptimise(infidelity, left, right, Y, left0, right0, optalg; trunc=trunc)

    permute!(metric, Y, ((4, 1), (3, 2)))

    # cycle_entropy(iso, metric)

    fixgauge!(iso, metric, gauge)
    # fixgauge!(iso, metric, WTG(skiptol=1e-14))

    if xi < Inf
        rescale!(iso, xi)
        fixgauge!(iso, metric, gauge)
    end

    infid = infidelity(iso.u, iso.v)

    @info "Optimised to an infidelity: $(infid)"

    # cycle_entropy(iso, metric)

    return iso, infid
end

function isometryoptimise(cf, u, v, metric, u0, v0, method::DZM; trunc)
    iso = dziarmaga(
        cf,
        u,
        v,
        metric,
        u0,
        v0;
        trunc=trunc,
        maxiter=method.maxiter,
        verbose=method.verbose,
        isotol=method.isotol,
        randomness=method.randomness,
        EAT=method.EAT,
    )
    return iso
end

function dzmtsvd!(cf, iso, u0, v0, min=Inf; trunc, check=false)
    u, s, v = safetsvd!(u0 * v0; trunc=trunc)
    maxnormalize!(s)

    if check
        min = dzmcheck!(cf, iso, u, s, v, min)
    end

    return u, s, v, min
end

function safetsvd!(m; trunc=notrunc())
    local u
    local s
    local v
    local err
    try
        u, s, v, err = tsvd!(m; trunc=trunc)
    catch e
        if e isa LAPACKException
            @warn "LAPACK routine `gesdd` failed; trying `gesvd`..."
            u, s, v, err = tsvd!(m; trunc=trunc, alg=TensorKit.SVD())
        else
            rethrow()
        end
    end

    return u, s, v, err
end

# safetsvd!(m; trunc=notrunc()) = tsvd!(m; trunc=trunc)

function dzmcheck!(cf, iso, u, s, v, min)
    uu = u * sqrt(s)
    vv = sqrt(s) * v

    infidelity = cf(uu, vv)

    if infidelity < min
        min = infidelity
        copy!(iso.u, uu)
        copy!(iso.s, s)
        copy!(iso.v, vv)
    end

    @debug "Obtained infidelity: $infidelity"
    return min
end

function dziarmaga(
    cf,
    ui,
    vi,
    metric,
    u0,
    v0;
    trunc,
    EAT=true,
    maxiter=100,
    isotol=1e-8,
    check=false,
    randomness=0.0,
    verbose=true,
)
    @info "Optimising isometries (Dziarmaga)..."

    # Convenience
    Y = metric

    # trunc = truncdim(dim(trunc.space))

    # The following tensor will be useful...
    Y0 = Y * permute(u0 * v0, (1, 2), ())

    # Initialize the initial best guess to that of simple-update.
    u, s, v = safetsvd!(ui * vi; trunc=trunc)

    maxnormalize!(s)

    iso = TruncationIsometries(metric, u * sqrt(s), s, sqrt(s) * v)

    minif = cf(iso.u, iso.v)
    u, s, v, minif = dzmtsvd!(cf, iso, ui, vi, minif; trunc=trunc, check=check)

    if dim(codomain(u)) == dim(domain(u)) && dim(codomain(v)) == dim(domain(v))
        @info "No need to optimise. Skipping..."
        return iso
    end
    # u, s, v = tsvd!(ui * vi; trunc=trunc)

    @warn "From simple update:" minif

    # Perform EAT if we want to.
    if EAT
        ui, vi = eat(ui, vi, metric, trunc)
        u, s, v, minif = dzmtsvd!(cf, iso, ui, vi, minif; trunc=trunc, check=check)
    end

    # Get an appropriate identity matrix for the spaces.
    idu = one(u0)
    idv = one(v0)

    counter = 0
    ϵ = Inf

    vt = similar(v)
    ut = similar(u)

    # u = (1 - randomness) * u * sqrt(s) + randomness * ITC.randnt!(similar(u))
    # v = randomness * ITC.randnt!(similar(v)) + (1 - randomness) * sqrt(s) * v

    # u = (1 - randomness) * u + randomness * ITC.randn!(similar(u))
    # v = randomness * ITC.randn!(similar(v)) + (1 - randomness) * v

    while counter < maxiter && ϵ ≥ isotol
        s_old = s

        # Optimise for ut with vt fixed.
        mul!(vt, s, v)
        maxu = optsingleiso(idu ⊗ transpose(vt), Y, Y0)
        u, s, v, minif = dzmtsvd!(cf, iso, maxu, vt, minif; trunc=trunc, check=check)

        # Optimise for vt with ut fixed.
        mul!(ut, u, s)
        maxv = optsingleiso(ut ⊗ transpose(idv), Y, Y0)
        u, s, v, minif = dzmtsvd!(cf, iso, ut, maxv, minif; trunc=trunc, check=check)

        # Calculate convergence measure as the change in singular values
        ϵ = norm(s - s_old)

        verbose && @info "\tIteration: $counter, ϵ: $ϵ"

        counter += 1
    end
    # If we have been checking the fidelity of the isometries, then our best result is
    # already stored in `iso`, otherwise we need update `iso`
    if !check
        maxnormalize!(s)

        mul!(iso.u, u, sqrt(s))
        copy!(iso.s, s)
        mul!(iso.v, sqrt(s), v)
    end

    @info "...done; iterations: $counter, ϵ: $ϵ"

    println(s)

    return iso
end

function stablizemetric!(metric, norm=@tensor N = metric[i i; j j])
    axpby!(0.5 / norm, metric', 0.5 / norm, metric)
    return metric
end

# function maxnormalize!(s)
#     max = 0.0
#
#     for (_, v) in blocks(s)
#         vmax = maximum(abs.(v))
#         if vmax > max
#             max = vmax
#         end
#     end
#
#     rmul!(s, 1 / max)
#     return s
# end
maxnormalize!(s) = rmul!(s, 1 / tr(s))

function isometryoptimise(cf, u, v, metric, u0, v0, method::EVB; trunc)
    iso = evenbly2(
        u,
        v,
        metric,
        u0,
        v0;
        trunc=trunc,
        maxiter=method.maxiter,
        verbose=method.verbose,
        pinvtol=method.pinvtol,
        isotol=method.isotol,
        randomness=method.randomness,
    )

    # u,s,v = evenblytrivial(
    #     metric;
    #     trunc,
    #     maxiter=100,
    #     verbose=true,
    #     eigentol=sqrt(eps(real(float(oneunit(scalartype(metric)))))),
    #     isotol=1e-8,
    # )
    # iso = TruncationIsometries(metric, u,s, v)

    return iso
end

function isometryoptimise(cf, u, v, metric, u0, v0, method::EVBTrivial; trunc)
    u,s,v = evenblytrivial(
        metric;
        trunc,
        maxiter=100,
        verbose=true,
        eigentol=sqrt(eps(real(float(oneunit(scalartype(metric)))))),
        isotol=1e-8,
    )
    iso = TruncationIsometries(metric, u,s, v)

    return iso
end

function evenbly2(
    u,
    v,
    metric,
    u0=id(u),
    v0=id(v);
    trunc,
    maxiter=100,
    verbose=true,
    pinvtol=sqrt(eps(real(float(oneunit(scalartype(metric)))))),
    isotol=1e-8,
    randomness=0.0,
)
    @info "Optimising isometries (Evenbly)..."

    metric = permute(metric, (2, 1), (4, 3))

    # metric = 0.5 * (metric + metric')
    # @tensor pmetricN = metric[i, i, j, j]
    # metric /= pmetricN
    #
    posapprox!(metric; tol=1e-3pinvtol)

    S = spacetype(metric)

    u, s, v = tsvd(u0 * v0; trunc=trunc)

    iso = TruncationIsometries(metric, u * sqrt(s), s, sqrt(s) * v)

    ut = (1 - randomness) * u * sqrt(s) + randomness * rand!(similar(u))
    vt = randomness * rand!(similar(v)) + (1 - randomness) * sqrt(s) * v

    normalize!(ut)
    normalize!(vt)

    D = domain(ut)
    η = codomain(ut)

    idmat = Matrix{ComplexF64}(I, dim(η), dim(D))

    ut = TensorMap(idmat, η', D')
    ut = ut + randomness * randn!(similar(ut))
    vt = TensorMap(idmat, η', D')
    vt = vt + randomness * randn!(similar(vt))

    B1 = similar(metric, η * D', η * D')
    B2 = similar(metric, D * η', D * η')

    P1 = similar(metric, η * D', one(S))
    P2 = similar(metric, D * η', one(S))

    RM1 = similar(P1)
    RM2 = similar(P2)

    R1 = similar(P1, D', η')
    # R2 = similar(P2, D, η)
    R2 = similar(P2, η', D')

    s1old = zero(id(D))
    s2old = zero(id(D))

    ϵ = Inf
    numiter = 0

    maxmethod = :pinv
    # maxmethod = :linsolve

    @info "bond env" space(metric)

    totaltime = 0
    # Dη <- U <- D
    #  D <- V <- Dη
    while numiter < maxiter
        totaltime += @elapsed begin
            @tensoropt B1[i j; k l] := ut[α; j] * metric[i α; k β] * (ut')[l; β]
            @tensoropt P1[i j] := ut[α; j] * metric[i α; β1 β2] * cj(u0)[β2; β] * cj(v0)[β; β1] 

            axpby!(0.5, B1', 0.5, B1)

            # mul!(RM1, safepinv(B1; rtol=pinvtol), P1)
            RM1 = safepinv(B1; rtol=pinvtol) * P1

            R1 = permute(RM1, ((1,), (2,)))
            u1, s1, v1 = safetsvd!(R1)

            copy!(vt, u1 * v1) 

            @tensoropt B2[i j; k l] := (vt')[i; α] * metric[α j; β l] * vt[β; k]
            @tensoropt P2[i j] := (vt')[i; α] * metric[α j; β1 β2] * cj(u0)[β2; β] * cj(v0)[β; β1] 

            # B1 is hermitian
            axpby!(0.5, B2', 0.5, B2)

            RM2 = safepinv(B2; rtol=pinvtol) * P2

            R2 = permute(RM2, ((1,), (2,)))

            u2, s2, v2 = safetsvd!(R2')
            copy!(ut, u2 * v2)
        end

        ϵ1 = norm(normalize(s1) - normalize(s1old))
        ϵ2 = norm(normalize(s2) - normalize(s2old))

        copy!(s1old, s1)
        copy!(s2old, s2)

        ϵ = max(ϵ1, ϵ2)

        numiter += 1

        verbose && @info "\tIteration: $numiter, ϵ: $ϵ"

        ϵ < isotol && break
    end
    @tensoropt B1[i j; k l] := ut[α; j] * metric[i α; k β] * (ut')[l; β]
    @tensoropt P1[i j] := ut[α; j] * metric[i α; β1 β2] * cj(u0)[β2; β] * cj(v0)[β; β1] 

    axpby!(0.5, B1', 0.5, B1)

    # mul!(RM1, safepinv(B1; rtol=pinvtol), P1)
    RM1 = safepinv(B1; rtol=pinvtol) * P1

    R1 = permute(RM1, ((1,), (2,)))
    u, s, v = safetsvd!(R1)

    # u = transpose(u)
    # s = transpose(s)
    # v = transpose(v)


    maxnormalize!(s)
    # As v is an isometry, we can avoid inverting `sqrt(s)` be instead calculating adjoint
    # u * sqrt(s) * v * v' * sqrt(s) * v * v' * u'
    l = ut * v' * sqrt(s) * v
    r = u * sqrt(s) * v

    # l = u * sqrt(s) * v
    # r = vt' * v' * sqrt(s) * v
    # sigma = diag(convert(Array, s))
    # lm,rm,sigma = isometries_to_WTG!(
    #     permutedims(convert(Array, metric), [2,1,4,3]),
    #     convert(Array, l), 
    #     convert(Array, r), 
    #     sigma,
    # )
    #
    # # println(sigma)
    #
    # lt = TensorMap(lm, η, D)
    # rt = TensorMap(permutedims(conj(rm)), D, η)
    # s = TensorMap(diagm(sigma), D,D)

    lt = _transpose(cj(l'))
    rt = _transpose(cj(r))

    copy!(iso.u, lt)
    copy!(iso.s, s)
    copy!(iso.v, rt)

    itertime = totaltime / numiter

    @info "...done; iterations: $numiter, ϵ: $ϵ, average iteration time: $(itertime)"

    return iso
end
function evenbly(
    u,
    v,
    metric,
    u0=id(u),
    v0=id(v);
    trunc,
    maxiter=100,
    verbose=true,
    pinvtol=sqrt(eps(real(float(oneunit(scalartype(metric)))))),
    isotol=1e-8,
    randomness=0.0,
)
    @info "Optimising isometries (Evenbly)..."

    # posapprox!(metric)

    S = spacetype(metric)

    u, s, v = tsvd(u0 * v0; trunc=trunc)

    iso = TruncationIsometries(metric, u * sqrt(s), s, sqrt(s) * v)

    ut = (1 - randomness) * u * sqrt(s) + randomness * rand!(similar(u))
    vt = randomness * rand!(similar(v)) + (1 - randomness) * sqrt(s) * v

    normalize!(ut)
    normalize!(vt)

    # utmat = Matrix{ComplexF64}(I, dim(codomain(ut)), dim(domain(ut)))
    # vtmat = Matrix{ComplexF64}(I, dim(codomain(vt)), dim(domain(vt)))
    #
    # ut = TensorMap(utmat, space(ut))
    # vt = TensorMap(vtmat, space(vt))

    D = domain(ut)
    η = codomain(ut)

    B1 = similar(metric, η * D', η * D')
    B2 = similar(metric, D * η', D * η')

    P1 = similar(metric, η * D', one(S))
    P2 = similar(metric, D * η', one(S))

    RM1 = similar(P1)
    RM2 = similar(P2)

    R1 = similar(P1, η, D)
    R2 = similar(P2, D, η)

    s1old = id(D)
    s2old = id(D)

    ϵ = Inf
    numiter = 0

    maxmethod = :pinv
    # maxmethod = :linsolve

    @info "bond env" space(metric)

    totaltime = 0
    # Dη <- U <- D
    #  D <- V <- Dη
    while numiter < maxiter
        totaltime += @elapsed begin
            @tensoropt B1[i j; k l] =
                transpose(vt')[j; α] * metric[i α; k β] * transpose(vt)[β; l]
            @tensoropt P1[i j] =
                transpose(vt')[j; α] * metric[i α; β1 β2] * u0[β1; β] * v0[β; β2]

            # B1 is hermitian
            axpby!(0.5, B1', 0.5, B1)

            mul!(RM1, safepinv(B1; rtol=pinvtol), P1)

            permute!(R1, RM1, ((1,), (2,)))

            u1, s1, v1 = safetsvd!(R1)
            copy!(ut, u1 * v1)

            @tensoropt B2[i j; k l] = (ut')[i; α] * metric[α j; β l] * ut[β; k]
            @tensoropt P2[i j] = (ut')[i; α] * metric[α j; β1 β2] * u0[β1; β] * v0[β; β2]

            axpby!(0.5, B2', 0.5, B2)

            mul!(RM2, safepinv(B2; rtol=pinvtol), P2)

            permute!(R2, RM2, ((1,), (2,)))
            u2, s2, v2 = safetsvd!(R2)

            copy!(vt, u2 * v2)
        end

        ϵ1 = norm(normalize(s1) - normalize(s1old))
        ϵ2 = norm(normalize(s2) - normalize(s2old))

        copy!(s1old, s1)
        copy!(s2old, s2)

        ϵ = max(ϵ1, ϵ2)

        numiter += 1

        verbose && @info "\tIteration: $numiter, ϵ: $ϵ"

        ϵ < isotol && break
    end

    @tensor begin
        B1[i j; k l] = transpose(vt')[j; α] * metric[i α; k β] * transpose(vt)[β; l]
        P1[i; j] = transpose(vt')[j; α] * metric[i α; β1 β2] * u0[β1; β] * v0[β; β2]
    end

    axpby!(0.5, B1', 0.5, B1)

    permute!(R1, safepinv(B1; rtol=pinvtol) * P1, ((1,), (2,)))
    # permute!(R1, pinv(B1; rtol=pinvtol) * P1, ((1,), (2,)))

    u, s, v = safetsvd!(R1)

    maxnormalize!(s)

    # As v is an isometry, we can avoid inverting `sqrt(s)` be instead calculating adjoint
    # u * sqrt(s) * v * v' * sqrt(s) * v * v' * u'
    l = u * sqrt(s) * v
    r = vt' * v' * sqrt(s) * v

    # l = u * sqrt(s) * v
    # r = vt' * v' * sqrt(s) * v

    println(s)

    copy!(iso.u, l)
    copy!(iso.s, s)
    copy!(iso.v, r')

    itertime = totaltime / numiter

    @info "...done; iterations: $numiter, ϵ: $ϵ, average iteration time: $(itertime)"

    return iso
end

function evenblytrivial(
    metric;
    trunc,
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
    Deig = eigen!(Hermitian(reshape(env, χenv^2, χenv^2)))
    χeig = sum(real.(Deig.values) .> 1e-3eigentol)
    vvv = sortperm(real(Deig.values); rev=true)
    env .= reshape(
        (
            Deig.vectors[:, vvv[1:χeig]] *
            Diagonal(Deig.values[vvv[1:χeig]]) *
            (Deig.vectors[:, vvv[1:χeig]]')
        ),
        (χenv, χenv, χenv, χenv),
    )

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

    # ut - s - vt'
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

    l = ut * F1.Vt' * Diagonal(sqrt.(F1.S)) * F1.Vt
    σ = F1.S[1:D]
    r = F1.U * Diagonal(sqrt.(F1.S)) * F1.Vt

    println("sigma: ", σ)

    ######## end iteration and output convergence data #########################
    verbose ? print("] iterations:", counter, " δ:", ϵ, "\n") : ()
    isometries_to_WTG!(env, l, r, σ)

    lt = TensorMap(l, χsp, Dsp)
    rt = TensorMap(permutedims(conj(r)), Dsp, χsp)

    s = TensorMap(diagm(σ), Dsp,Dsp)

    println(s)

    # return lt, id(Dsp), rt
    return lt, s, rt
end

function to_WTG!(
    env::AbstractTensorMap{S}, l, r, σ, test::Bool=true, zeta::Float64=Inf
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
            @info "Skipping gauge fixing (left)"
            return nothing
        end
    end
    for (_, v) in blocks(DR)
        if minimum(abs.(diag(v))) < 1e-14
            @info "Skipping gauge fixing (right)"
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
    env::AbstractTensorMap{S}, u, v, s, test::Bool=true, zeta::Float64=Inf
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
        rmul!(s, 1 / max)
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
    zeta::Float64=Inf,
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
        @info "skipped gauge fixing"
        return l,r,σ
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

    if zeta != Inf
        scl = [exp(-(i - 1) / zeta) for i in 1:D]
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
    return l, r, σ
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

    # TODO: safepinv is type unstable currently
    Ma_max = safepinv(0.5 * (g + g'); rtol=rtol) * j
    # Ma_max = pinv(0.5 * (g + g'); rtol=rtol) * j

    # ϵ_j = scalar(Ma_max' * j)
    # ϵ = scalar(Ma_max' * g * Ma_max) - ϵ_j - ϵ_j' + ϵ0

    Ma_max = permute(Ma_max, (1,), (2,)) # , abs(ϵ)

    return Ma_max
end

function eat(L, R, g, trunc; ptol=1e-8)
    @tensoropt G[-3 -4; -1 -2] :=
        L[3; -3] * R[-4; 4] * g[1 2; 3 4] * (L')[-1; 1] * (R')[2; -2]

    Ga, _, Gb = tsvd!(G; trunc=truncdim(1))

    Gap = permute(Ga, (2,), (1, 3))
    Ga = TensorMap(Gap.data, codomain(Gap), fuse(*(domain(Gap)...)))
    Gbp = permute(Gb, (2,), (1, 3))
    Gb = TensorMap(Gbp.data, codomain(Gbp), fuse(*(domain(Gbp)...)))

    λa, Ua = eigh!(Ga)
    λb, Ub = eigh!(Gb)

    λa = copy!(similar(λa, ComplexF64), λa)
    λb = copy!(similar(λb, ComplexF64), λb)

    Wa, λ, Wb = tsvd!(sqrt(λa) * Ua * Ub * sqrt(λb); trunc=trunc)

    try
        Ma = L * Ua' * safepinv(sqrt(λa); rtol=ptol) * Wa * sqrt(λ)
        Mb = sqrt(λ) * Wb * safepinv(sqrt(λb); rtol=ptol) * Ub' * R
        @debug "EAT success"
        return Ma, Mb
    catch
        @debug "EAT failed"
        return L, R
    end
end
