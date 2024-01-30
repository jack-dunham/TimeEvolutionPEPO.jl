function _directcf(u, v, l, r, M, bond)
    x = bond.left[1]
    y = bond.left[2]
    @tensoropt (
        k1 => 2,
        k2 => 2,
        k3 => 2,
        k4 => 2,
        k5 => 2,
        k6 => 2,
        k7 => 2,
        k8 => 2,
        b1 => 2,
        b2 => 2,
        b3 => 2,
        b4 => 2,
        b5 => 2,
        b6 => 2,
        b7 => 2,
        b8 => 2,
    ) rv =
        (M[x, y - 1].top)[k1 b1; r2 d4 r1 d1] *
        conj((M[x, y - 1].bot)[k1 b1; s2 e4 r1 d1]) *
        (M[x + 1, y - 1].top)[k2 b2; r3 d6 r2 d2] *
        conj((M[x + 1, y - 1].bot)[k2 b2; r3 e6 s2 d2]) *
        (M[x - 1, y].top)[k3 b3; r6 d8 r5 d3] *
        conj((M[x - 1, y].bot)[k3 b3; s6 d8 r5 d3]) *
        (M[x, y].top)[k4 b4; r7 d9 r6 d4] *
        conj((M[x, y].bot)[k4 b4; s7 e9 s6 e4]) *
        u[r7; d5] *
        v[d5; r8] *
        l[e5; s7] *
        r[s8; e5] *
        (M[x + 1, y].top)[k5 b5; r9 d10 r8 d6] *
        conj((M[x + 1, y].bot)[k5 b5; s9 e10 s8 e6]) *
        (M[x + 2, y].top)[k6 b6; r10 d11 r9 d7] *
        conj((M[x + 2, y].bot)[k6 b6; r10 d11 s9 d7]) *
        (M[x, y + 1].top)[k7 b7; r12 d12 r11 d9] *
        conj((M[x, y + 1].bot)[k7 b7; s12 d12 r11 e9]) *
        (M[x + 1, y + 1].top)[k8 b8; r13 d13 r12 d10] *
        conj((M[x + 1, y + 1].bot)[k8 b8; r13 d13 s12 e10])
    return real(rv)
end
function _directtruncmetric(M, bond)
    x = bond.left[1]
    y = bond.left[2]
    @tensoropt (
        k1 => 2,
        k2 => 2,
        k3 => 2,
        k4 => 2,
        k5 => 2,
        k6 => 2,
        k7 => 2,
        k8 => 2,
        b1 => 2,
        b2 => 2,
        b3 => 2,
        b4 => 2,
        b5 => 2,
        b6 => 2,
        b7 => 2,
        b8 => 2,
    ) rv[r8 s7; r7 s8] :=
        (M[x, y - 1].top)[k1 b1; r2 d4 r1 d1] *
        conj((M[x, y - 1].bot)[k1 b1; s2 e4 r1 d1]) *
        (M[x + 1, y - 1].top)[k2 b2; r3 d6 r2 d2] *
        conj((M[x + 1, y - 1].bot)[k2 b2; r3 e6 s2 d2]) *
        (M[x - 1, y].top)[k3 b3; r6 d8 r5 d3] *
        conj((M[x - 1, y].bot)[k3 b3; s6 d8 r5 d3]) *
        (M[x, y].top)[k4 b4; r7 d9 r6 d4] *
        conj((M[x, y].bot)[k4 b4; s7 e9 s6 e4]) *
        (M[x + 1, y].top)[k5 b5; r9 d10 r8 d6] *
        conj((M[x + 1, y].bot)[k5 b5; s9 e10 s8 e6]) *
        (M[x + 2, y].top)[k6 b6; r10 d11 r9 d7] *
        conj((M[x + 2, y].bot)[k6 b6; r10 d11 s9 d7]) *
        (M[x, y + 1].top)[k7 b7; r12 d12 r11 d9] *
        conj((M[x, y + 1].bot)[k7 b7; s12 d12 r11 e9]) *
        (M[x + 1, y + 1].top)[k8 b8; r13 d13 r12 d10] *
        conj((M[x + 1, y + 1].bot)[k8 b8; r13 d13 s12 e10])
    return real(rv)
end

function directcf(u, v, l, r, ql, qr, M, bond)
    x = bond.left[1]
    y = bond.left[2]
    @tensoropt (
        k1 => 2,
        k2 => 2,
        k3 => 2,
        k4 => 2,
        k5 => 2,
        k6 => 2,
        k7 => 2,
        k8 => 2,
        b1 => 2,
        b2 => 2,
        b3 => 2,
        b4 => 2,
        b5 => 2,
        b6 => 2,
        b7 => 2,
        b8 => 2,
        d1 => D,
        d2 => D,
        d3 => D,
        d4 => D,
        d5 => D,
        d6 => D,
        d7 => D,
        d8 => D,
        d9 => D,
        d10 => D,
        d11 => D,
        d12 => D,
        d13 => D,
        e4 => D,
        e5 => D,
        e6 => D,
        e9 => D,
        e10 => D,
        r1 => D,
        r2 => D,
        r3 => D,
        r4 => D,
        r5 => D,
        r6 => D,
        r7 => D,
        r8 => D,
        r9 => D,
        r10 => D,
        r11 => D,
        r12 => D,
        r13 => D,
        s2 => D,
        s6 => D,
        s7 => D,
        s8 => D,
        s9 => D,
        s12 => D,
    ) rv =
        (M[x, y - 1].top)[k1 b1; r2 d4 r1 d1] *
        conj((M[x, y - 1].bot)[k1 b1; s2 e4 r1 d1]) *
        (M[x + 1, y - 1].top)[k2 b2; r3 d6 r2 d2] *
        conj((M[x + 1, y - 1].bot)[k2 b2; r3 e6 s2 d2]) *
        (M[x - 1, y].top)[k3 b3; r6 d8 r5 d3] *
        conj((M[x - 1, y].bot)[k3 b3; s6 d8 r5 d3]) *
        (ql)[k4 b4; r7 d9 r6 d4] *
        conj(ql[k4 b4; s7 e9 s6 e4]) *
        u[r7; d5] *
        v[d5; r8] *
        l[e5; s7] *
        r[s8; e5] *
        (qr)[k5 b5; r9 d10 r8 d6] *
        conj(qr[k5 b5; s9 e10 s8 e6]) *
        (M[x + 2, y].top)[k6 b6; r10 d11 r9 d7] *
        conj((M[x + 2, y].bot)[k6 b6; r10 d11 s9 d7]) *
        (M[x, y + 1].top)[k7 b7; r12 d12 r11 d9] *
        conj((M[x, y + 1].bot)[k7 b7; s12 d12 r11 e9]) *
        (M[x + 1, y + 1].top)[k8 b8; r13 d13 r12 d10] *
        conj((M[x + 1, y + 1].bot)[k8 b8; r13 d13 s12 e10])
    return real(rv)
end

function costfunc(u, v, l, r, ql, qr, M, bond)
    cf = (u, v, l, r) -> directcf(u, v, l', r', ql, qr, M, bond)
    return norm(1 - cf(u, v, l, r) - cf(l, r, u, v) + cf(l, r, l, r))
end

function getmanifold(u::AbsTen{N,M}) where {N,M}
    return Euclidean(N, M; field=Manifolds.ℂ)::Euclidean{Tuple{N,M},Manifolds.ℂ}
end

function cgoptimize(left0, right0, pepo, bond, trunc, ::Val{D}, ::Val{Dp}) where {D,Dp}
    hs = hilbertschmidt(pepo)

    ql, qr = pepo[bond]

    # initialize with tsvd

    u, s, v = tsvd(left0 * right0; trunc=trunc)

    au0 = (u * sqrt(s)).data
    av0 = (sqrt(s) * v).data

    man_left =
        Euclidean(size(au0)...; field=Manifolds.ℂ)::Euclidean{Tuple{D,Dp},Manifolds.ℂ}
    man_right =
        Euclidean(size(av0)...; field=Manifolds.ℂ)::Euclidean{Tuple{Dp,D},Manifolds.ℂ}

    man = ProductManifold(man_left, man_right)

    f =
        (M, x) -> costfunc(
            TensorMap(x.parts[1], codomain(u), domain(u)),
            TensorMap(x.parts[2], codomain(v), domain(v)),
            left0,
            right0,
            ql,
            qr,
            hs,
            bond,
        )::Float64

    println(f(man, rand(man)))

    g =
        (M, x) -> ProductRepr(
            gradient(y -> f(M, y), x)[1].parts
        )::ProductRepr{Tuple{Matrix{ComplexF64},Matrix{ComplexF64}}}

    rv0 = ProductRepr(au0, av0)

    gradient_descent!(man, f, g, rv0; stopping_criterion=StopWhenCostLess(1e-8))

    println(f(man, rv0))

    u_temp = TensorMap(rv0.parts[1], codomain(u), domain(u))
    v_temp = TensorMap(rv0.parts[2], codomain(v), domain(v))

    uf, s_f64, vf = tsvd!(u_temp * v_temp; trunc=trunc)

    sf = similar(s_f64, eltype(uf))

    copy!(sf, s_f64)
    return TruncationIsometries(uf, sf, vf)
end

# function singlesitecf(u, v, M, x, y)
#     @tensoropt (
#         k1 => 2,
#         k2 => 2,
#         k3 => 2,
#         k4 => 2,
#         k5 => 2,
#         b1 => 2,
#         b2 => 2,
#         b3 => 2,
#         b4 => 2,
#         b5 => 2,
#         d1 => D,
#         d2 => D,
#         d3 => D,
#         d4 => D,
#         d5 => D,
#         d6 => D,
#         d7 => D,
#         d8 => D,
#         e3 => D,
#         e4 => D,
#         r1 => D,
#         r2 => D,
#         r3 => D,
#         r4 => D,
#         r5 => D,
#         r6 => D,
#         r7 => D,
#         r8 => D,
#         s4 => D,
#         s5 => D,
#         x1 => D,
#         x2 => D,
#         x3 => D,
#         x4 => D,
#         x5 => D,
#     ) rv =
#         (M[x, y - 1].top)[k1 b1; r2 d3 r1 d1] *
#         conj((M[x, y - 1].bot)[k1 b1; r2 e3 r1 d1]) *
#         (M[x - 1, y].top)[k2 b2; r4 d5 r3 d2] *
#         conj((M[x - 1, y].bot)[k2 b2; s4 d5 r3 d2]) *
#         u[r4; x1] *
#         v[x1; x2] *
#         (M[x, y].top)[k3 b3; x3 d6 x2 d3] *
#         conj((M[x, y].bot)[k3 b3; s5 e6 s4 e3]) *
#         u[x3; x4] *
#         v[x4; r5] *
#         (M[x + 1, y].top)[k4 b4; r6 d7 r5 d4] *
#         conj((M[x + 1, y].bot)[k4 b4; r6 d7 s5 d4]) *
#         (M[x, y + 1].top)[k5 b5; r8 d8 r7 d6] *
#         conj((M[x, y + 1].bot)[k5 b5; r8 d8 r7 e6])
#     return real(rv)
# end

function singlesitecf(u, s, v, M, x, y)
    @tensoropt (
        k1 => 2,
        k2 => 2,
        k3 => 2,
        k4 => 2,
        k5 => 2,
        b1 => 2,
        b2 => 2,
        b3 => 2,
        b4 => 2,
        b5 => 2,
        d1 => D,
        d2 => D,
        d3 => D,
        d4 => D,
        d5 => D,
        d6 => D,
        d7 => D,
        d8 => D,
        e3 => D,
        e4 => D,
        r1 => D,
        r2 => D,
        r3 => D,
        r4 => D,
        r5 => D,
        r6 => D,
        r7 => D,
        r8 => D,
        s4 => D,
        s5 => D,
        x1 => D,
        x3 => D,
        m1 => D,
        m2 => D,
        m3 => D,
        m4 => D,
    ) rv =
        (M)[k1 b1; r2 d3 r1 d1] *
        conj((M)[k1 b1; r2 e3 r1 d1]) *
        (M)[k2 b2; r4 d5 r3 d2] *
        conj((M)[k2 b2; s4 d5 r3 d2]) *
        u[r4; m1] *
        s[m1; m2] *
        v[m2; x2] *
        (M)[k3 b3; x3 d6 x2 d3] *
        conj((M)[k3 b3; s5 e6 s4 e3]) *
        u[x3; m3] *
        s[m3; m4] *
        v[m4; r5] *
        (M)[k4 b4; r6 d7 r5 d4] *
        conj((M)[k4 b4; r6 d7 s5 d4]) *
        (M)[k5 b5; r8 d8 r7 d6] *
        conj((M)[k5 b5; r8 d8 r7 e6])
    return real(rv)
end

function sscgoptimise!(pepo)
    @info "trying quick truncate"

    site = getstate(pepo)[1, 1]
    phi = getstate(pepo)[1, 1].data

    c = codomain(site)
    d = domain(site)
    Dp_sp = (domain(getevolved(pepo)[1, 1])[1])
    Dp = dim(Dp_sp)
    D = dim(domain(getstate(pepo)[1, 1])[1])

    manif = Euclidean(Dp, Dp; field=Manifolds.ℂ)

    println(Dp_sp)

    pepo_ev = getevolved(pepo)[1, 1]

    f = (M, x) -> fidelitycf(x, pepo_ev, D, 1, 1, Dp_sp)
    g = (M, x) -> gradient(y -> f(M, y), x)[1]

    guess = rand(manif)

    println(f(manif, guess))

    gradient_descent(manif, f, g, guess)

    u, s, v = tsvd(mat; trunc=truncdim(D))

    iso = TruncationIsometries(u, s, v)

    truncate!(pepo, iso)

    return pepo
end

function fidelitycf(mat, M, D, x, y, Dp_sp)
    mat = TensorMap(mat, Dp_sp, Dp_sp)
    u, s, v = tsvd(mat; trunc=truncdim(D))
    # uf = u * sqrt(s)
    # vf = sqrt(s) * v
    # \rho is evolved state
    trϕρ = singlesitecf(u, s, v, M, 1, 1)
    # trρρ =  singlesite(hilbertschmidt(ρ,ρ))
    trϕϕ = singlesitecf(one(mat), one(mat), one(mat), M, 1, 1)

    return 1 - trϕρ^2 / trϕϕ
end
