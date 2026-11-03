# Iterative simple update (FCLS manuscript, Appendix D) for a long-range propagator: the
# `InfinitePEPSO` is contracted onto the state, and every bond is cut against the current projectors
# of the bonds around it, repeating until the bond weights stop changing.

"""
$(TYPEDEF)

How [`iterativestep!`](@ref) cuts the bonds when it applies a long-range propagator.

# Constructors

    IterativeSU(; maxiter = 20, tol = 1e-6, protect = :none, mu = Inf)

- `maxiter`, `tol`: repeat the cuts until no bond weight changes by more than `tol`, at most
  `maxiter` times
- `protect`: `:none` keeps each bond's top singular directions. `:trace` also keeps the trace of the
  state, and `:twosite` the two-site density matrix across the bond, both as belief propagation on
  the traced network sees them through that bond
- `mu`: `Inf` keeps the `protect` quantities exactly; a finite value instead adds their relative
  change, weighted by `mu`, to the error the cut minimises
"""
struct IterativeSU
    maxiter::Int
    tol::Float64
    protect::Symbol
    mu::Float64

    function IterativeSU(maxiter, tol, protect, mu)
        maxiter >= 1 || throw(ArgumentError("`maxiter` must be at least 1, got $maxiter"))
        protect in (:none, :trace, :twosite) ||
            throw(ArgumentError("`protect` must be :none, :trace or :twosite, got $protect"))
        mu >= 0 || throw(ArgumentError("the penalty weight `mu` must be non-negative, got $mu"))
        return new(maxiter, tol, protect, mu)
    end
end

function IterativeSU(; maxiter = 20, tol = 1e-6, protect = :none, mu = Inf)
    return IterativeSU(maxiter, tol, protect, mu)
end

"""
$(TYPEDEF)

A long-range propagator as one entry of a Trotter decomposition (see [`trotterise`](@ref)). It also
holds the projectors and traced-network messages its last application ended with, which the next
application starts from, so [`timestep!`](@ref) mutates it.

The propagator's first virtual state on every leg must be the idle state of its finite-state machine,
the one no interaction path occupies, as [`fsmpepso`](@ref) builds it.

# Constructors

    PEPSOLayer(propagator::InfinitePEPSO)
"""
# `S` is left unbounded: the registry PEPSKit the documentation builds against has no `InfinitePEPSO`,
# so naming the type at load time would stop the package loading there.
mutable struct PEPSOLayer{S}
    propagator::S
    projectors::Any
    messages::Any
end

PEPSOLayer(propagator) = PEPSOLayer(propagator, nothing, nothing)

Base.show(io::IO, layer::PEPSOLayer) = print(io, "PEPSOLayer(", size(layer.propagator), ")")

# Maps between each leg pair of the two-layer network at a site and one fused leg: `Fn` and `Fe` fuse
# the north and east pairs, `Fs` and `Fw` split the south and west legs back into pairs.
function _fusers(ρ::AbstractTensorMap, S::AbstractTensorMap)
    Nr, Er, Sr, Wr = domain(ρ)
    Ns, Es, Ss, Ws = domain(S)
    return (;
        Fn = isomorphism(Nr ⊗ Ns, fuse(Nr ⊗ Ns)),
        Fe = isomorphism(Er ⊗ Es, fuse(Er ⊗ Es)),
        Fs = isomorphism(fuse(Sr' ⊗ Ss'), Sr' ⊗ Ss'),
        Fw = isomorphism(fuse(Wr' ⊗ Ws'), Wr' ⊗ Ws'),
    )
end

"""
    $(TYPEDSIGNATURES)

The site tensor `ρ` with the propagator `S` applied and each pair of virtual legs mapped to one leg by
`Pn`, `Pe`, `Ps`, `Pw` (laid out as PEPSKit's `compress` projectors), shaped as a `PEPOTensor`.
`last` names the leg contracted at the end; passing the leg that carries a fuser rather than a
projector, the largest of the four, keeps the intermediates small.
"""
function _composite(ρ, S, Pn, Pe, Ps, Pw; last::Symbol = :w)
    alloc = TensorOperations.ManualAllocator()
    if last === :w
        @tensor allocator = alloc order = (e2, n2, s1, q1, q2, n1, e1, s2, w1, w2) A[p1 p2; n e sd w] :=
            ρ[q1 q2; n1 e1 s1 w1] * S[p1 p2 q1 q2; n2 e2 s2 w2] *
            Pn[n1 n2; n] * Pe[e1 e2; e] * Ps[sd; s1 s2] * Pw[w; w1 w2]
    elseif last === :e
        @tensor allocator = alloc order = (w2, n2, s1, q1, q2, n1, w1, s2, e1, e2) A[p1 p2; n e sd w] :=
            ρ[q1 q2; n1 e1 s1 w1] * S[p1 p2 q1 q2; n2 e2 s2 w2] *
            Pn[n1 n2; n] * Pe[e1 e2; e] * Ps[sd; s1 s2] * Pw[w; w1 w2]
    elseif last === :n
        @tensor allocator = alloc order = (w2, e2, s1, q1, q2, e1, w1, s2, n1, n2) A[p1 p2; n e sd w] :=
            ρ[q1 q2; n1 e1 s1 w1] * S[p1 p2 q1 q2; n2 e2 s2 w2] *
            Pn[n1 n2; n] * Pe[e1 e2; e] * Ps[sd; s1 s2] * Pw[w; w1 w2]
    elseif last === :s
        @tensor allocator = alloc order = (w2, e2, n1, q1, q2, e1, w1, n2, s1, s2) A[p1 p2; n e sd w] :=
            ρ[q1 q2; n1 e1 s1 w1] * S[p1 p2 q1 q2; n2 e2 s2 w2] *
            Pn[n1 n2; n] * Pe[e1 e2; e] * Ps[sd; s1 s2] * Pw[w; w1 w2]
    else
        throw(ArgumentError("`last` must be :n, :e, :s or :w, got $last"))
    end
    return A
end

# Multiply leg `i` of `t` (codomain legs first) elementwise by `v`, acting on the dense data.
function _scaleleg(t::AbstractTensorMap, i::Int, v::AbstractVector)
    t2 = copy(t)
    a = reshape(t2.data, dims(codomain(t))..., dims(domain(t))...)
    a .*= reshape(v, ntuple(j -> j == i ? length(v) : 1, ndims(a)))
    return t2
end

_matrix(t::AbstractTensorMap) = reshape(convert(Array, t), dim(codomain(t)), dim(domain(t)))

_orthcols(A; rtol = 1e-10) = (F = svd(A); F.U[:, F.S .> rtol * max(F.S[1], eps())])

_normmax(v) = v / v[argmax(abs.(v))]

_weightvalues(w::AbstractTensorMap) = real.(diag(convert(Array, w)))

# R1, R2 with X = Q1 R1 and Y = R2 Q2 across the bond joining domain leg `kx` of `X` to `ky` of `Y`.
function _bondfactors(X, kx, Y, ky)
    nx, ny = numout(X), numout(Y)
    Xp = permute(X, (Tuple(setdiff(1:(nx + 4), nx + kx)), (nx + kx,)))
    Yp = permute(Y, ((ny + ky,), Tuple(setdiff(1:(ny + 4), ny + ky))))
    D1, V1 = eigh_full(Xp' * Xp)
    D1.data .= sqrt.(max.(real.(D1.data), 0))
    D2, V2 = eigh_full(Yp * Yp')
    D2.data .= sqrt.(max.(real.(D2.data), 0))
    return D1 * V1', V2 * D2
end

# Top singular directions of M = R1 R2, as projectors P1 = R2 V S^-½ and P2 = S^-½ U' R1.
function _svdcut(R1, R2; trunc)
    M = R1 * R2
    u, s, vh = svd_trunc!(copy(M); trunc)
    # The largest entry of each singular vector is made real and positive, so the bond basis changes
    # smoothly between steps and the stored projectors remain a good start for the next one.
    ua = convert(Array, u)
    ph = [ua[argmax(abs.(ua[:, j])), j] for j in axes(ua, 2)]
    Φ = TensorMap(Matrix(Diagonal(conj.(ph) ./ abs.(ph))), domain(u) ← domain(u))
    u, vh = u * Φ, Φ' * vh
    isq = PEPSKit.sdiag_pow(s, -0.5)
    ϵ = sqrt(max(norm(M)^2 - norm(s)^2, 0)) / norm(M)
    return R2 * vh' * isq, isq * u' * R1, s, ϵ
end

# Projectors with weights taken from the cut's own singular value decomposition; `A` and `B` hold the
# left and right bases of the cut M̃ = A Σ B' on the two sides of the bond, neither of them isometric.
function _cutprojectors(R1, R2, Wl, Wr, D)
    K = ComplexSpace(D)
    P1 = R2 * TensorMap(Wl, domain(R2) ← K)
    P2 = TensorMap(Wr, K ← codomain(R1)) * R1
    return P1, P2, K
end

# Diagonal phases making the largest entry of each column of `U` real and positive.
_phasefix(U) = Diagonal([(p = U[argmax(abs.(U[:, j])), j]; conj(p) / abs(p)) for j in axes(U, 2)])

# As `_svdcut` with D directions, but the cut Π = P1 P2 also satisfies wL Π = wL and Π wR = wR for the
# rows of `wL` and columns of `wR`; the remaining directions are the top ones of what is left.
function _protectedcut(R1, R2, wL, wR; D)
    R1a, R2a = _matrix(R1), _matrix(R2)
    M = R1a * R2a
    Uk = _orthcols(pinv(R1a; rtol = 1e-12)' * wL')
    Vk = _orthcols(pinv(R2a; rtol = 1e-12) * wR)
    k = min(size(Uk, 2), size(Vk, 2), D)
    k == 0 && return nothing
    Uk, Vk = Uk[:, 1:k], Vk[:, 1:k]
    C = Uk' * M * Vk
    Fr = svd(M - M * Vk * (C \ (Uk' * M)))
    Ut = Matrix(qr([Uk Fr.U[:, 1:(D - k)]]).Q)[:, 1:D]
    Vt = Matrix(qr([Vk Fr.V[:, 1:(D - k)]]).Q)[:, 1:D]
    # the bond weight is the SVD of the cut M̃ = (M Vt) Ct⁻¹ (Ut' M), not of Ct = Ut' M Vt
    Ct = Ut' * M * Vt
    Qa, Qb = qr(M * Vt), qr(M' * Ut)
    Fc = svd(Qa.R * (Ct \ Qb.R'))
    Φ = _phasefix(Matrix(Qa.Q)[:, 1:D] * Fc.U)
    u, v, s = Fc.U * Φ, Fc.V * Φ, Fc.S
    rs = Diagonal(sqrt.(s))
    P1, P2, K = _cutprojectors(R1, R2, Vt * (Qa.R \ (u * rs)), (rs * v') * (Qb.R' \ Ut'), D)
    ϵ = norm(M - M * Vt * (Ct \ (Ut' * M))) / norm(M)
    return P1, P2, DiagonalTensorMap(s, K), ϵ
end

# Penalty form of `_protectedcut`: with wL = Y R1 and wR = R2 Z, the cut M̃ minimises ‖L (M - M̃) R‖ for
# L² = 1 + μ Ŷ'Ŷ and R² = 1 + μ ẐẐ', where Ŷ, Ẑ are scaled so that μ weighs relative changes of wL M, M wR.
function _softcut(R1, R2, wL, wR; D, mu)
    R1a, R2a = _matrix(R1), _matrix(R2)
    M = R1a * R2a
    Yc = wL * pinv(R1a; rtol = 1e-12)
    Zc = pinv(R2a; rtol = 1e-12) * wR
    Yh, Zh = Yc * (norm(M) / norm(Yc * M)), Zc * (norm(M) / norm(M * Zc))
    function weight(G)
        F = eigen(Hermitian(G))
        w = sqrt.(1 .+ mu .* max.(F.values, 0))
        return F.vectors * Diagonal(w) * F.vectors', F.vectors * Diagonal(1 ./ w) * F.vectors'
    end
    (L, Li), (R, Ri) = weight(Yh' * Yh), weight(Zh * Zh')
    F = svd(L * M * R)
    U, S, V = F.U[:, 1:D], F.S[1:D], F.V[:, 1:D]
    # M̃ = (L⁻¹U) S (R⁻¹V)' has non-isometric bases, so the bond weight comes from its own SVD, not S
    Qa, Qb = qr(Li * U), qr(Ri * V)
    Fk = svd(Qa.R * Diagonal(S) * Qb.R')
    Φ = _phasefix(Matrix(Qa.Q)[:, 1:D] * Fk.U)
    u, v, s = Fk.U * Φ, Fk.V * Φ, Fk.S
    rs = Diagonal(sqrt.(s))
    Wl = R * V * Diagonal(1 ./ S) * (Qa.R \ (u * rs))
    Wr = (rs * v') * (Qb.R' \ (Diagonal(1 ./ S) * U' * L))
    P1, P2, K = _cutprojectors(R1, R2, Wl, Wr, D)
    ϵ = norm(M - Li * U * Diagonal(S) * V' * Ri) / norm(M)
    return P1, P2, DiagonalTensorMap(s, K), ϵ
end

# Cut the bond between X (open domain leg `kx`) and Y (`ky`), both already weighted, keeping the
# `protect` quantities of `alg` carried by the protected vectors `wL`, `wR`.
function _cutbond(X, kx, Y, ky, wL, wR, trunc, alg::IterativeSU)
    R1, R2 = _bondfactors(X, kx, Y, ky)
    # dropping exactly vanishing directions keeps S^-½ finite when the bond has rank below D
    cut = _svdcut(R1, R2; trunc = trunc & trunctol(; rtol = 1e-14))
    alg.protect === :none && return cut
    D = dim(space(cut[3], 1))
    M = _matrix(R1) * _matrix(R2)
    count(>(1e-14 * opnorm(M)), svdvals(M)) <= D && return cut
    protected = if isinf(alg.mu)
        _protectedcut(R1, R2, wL, wR; D)
    else
        _softcut(R1, R2, wL, wR; D, mu = alg.mu)
    end
    return something(protected, cut)
end

# Periodic access to a 2 × Nr × Nc array of per-bond values, laid out as `SUWeight`.
_bondat(λ, d, r, c) = λ[d, mod1(r, size(λ, 2)), mod1(c, size(λ, 3))]

# The weights on the north, east, south and west legs of site (r, c).
function _siteweights(λ, r, c)
    return (
        _bondat(λ, 2, r, c), _bondat(λ, 1, r, c), _bondat(λ, 2, r + 1, c), _bondat(λ, 1, r, c - 1),
    )
end

# Site projectors (Pn, Pe, Ps, Pw) with their new legs weighted by √λ of the site's four bonds.
function _weighted(P, λs)
    Pn, Pe, Ps, Pw = P
    return (
        _scaleleg(Pn, 3, sqrt.(λs[1])), _scaleleg(Pe, 3, sqrt.(λs[2])),
        _scaleleg(Ps, 1, sqrt.(λs[3])), _scaleleg(Pw, 1, sqrt.(λs[4])),
    )
end

"""
    $(TYPEDSIGNATURES)

Belief-propagation messages of a traced network, `t[r, c]` being site (r, c) as an n-e-s-w array:
`m[d, r, c]` arrives at site (r, c) on leg `d` (north, east, south, west), scaled to unit largest
entry. Starts from `init` when its lengths fit.
"""
function _tracemessages(t::AbstractMatrix; init = nothing, tol = 1e-10, maxiter = 500)
    Nr, Nc = size(t)
    at(r, c) = t[mod1(r, Nr), mod1(c, Nc)]
    fits = !isnothing(init) && size(init) == (4, Nr, Nc) &&
        all(I -> length(init[I]) == size(t[I[2], I[3]], I[1]), CartesianIndices(init))
    m = fits ? copy(init) : [ones(ComplexF64, size(t[r, c], d)) for d in 1:4, r in 1:Nr, c in 1:Nc]
    msg(d, r, c) = m[d, mod1(r, Nr), mod1(c, Nc)]
    for _ in 1:maxiter
        new = similar(m)
        for r in 1:Nr, c in 1:Nc
            # each message leaves a neighbour through the leg facing (r, c)
            a = at(r - 1, c)
            mn, me, mw = msg(1, r - 1, c), msg(2, r - 1, c), msg(4, r - 1, c)
            @tensor vn[s] := a[n e s w] * mn[n] * me[e] * mw[w]
            a = at(r, c + 1)
            mn, me, ms = msg(1, r, c + 1), msg(2, r, c + 1), msg(3, r, c + 1)
            @tensor ve[w] := a[n e s w] * mn[n] * me[e] * ms[s]
            a = at(r + 1, c)
            me, ms, mw = msg(2, r + 1, c), msg(3, r + 1, c), msg(4, r + 1, c)
            @tensor vs[n] := a[n e s w] * me[e] * ms[s] * mw[w]
            a = at(r, c - 1)
            mn, ms, mw = msg(1, r, c - 1), msg(3, r, c - 1), msg(4, r, c - 1)
            @tensor vw[e] := a[n e s w] * mn[n] * ms[s] * mw[w]
            new[1, r, c], new[2, r, c], new[3, r, c], new[4, r, c] = _normmax.((vn, ve, vs, vw))
        end
        δ = maximum(norm.(new .- m))
        m = new
        δ < tol && break
    end
    return m
end

# Protected bond vectors of the open leg of a composite `X`, from the messages `q` on its other three
# legs: rows of ⟨q, X⟩ for each physical pair (`:twosite`) or their trace (`:trace`).
function _protected(X, open::Int, q, protect::Symbol)
    a = convert(Array, X)
    d1, d2 = size(a, 1), size(a, 2)
    others = filter(!=(open), 1:4)
    b = reshape(permutedims(a, ((2 .+ others)..., 1, 2, 2 + open)), :, d1 * d2 * size(a, 2 + open))
    # `kron` runs its last factor fastest, matching the first of `others` in the column-major `b`
    env = kron(reverse([q[k] for k in others])...)
    v = reshape(transpose(env) * b, d1 * d2, :)
    protect === :twosite && return v
    return reshape(sum(v[(i - 1) * d1 + i, :] for i in 1:min(d1, d2)), 1, :)
end

# Projectors onto the state's bond times the propagator's idle state: the exact cut of the composite
# before any path runs. The machine's other states have entries that continue or turn a path whether
# or not one was started, so a cut ranking them by norm alone would keep directions no path reaches.
function _idleprojectors(ρ, S)
    Nr, Nc = size(ρ)
    function idle(V, W)
        a = zeros(ComplexF64, dim(V), dim(W), dim(V))
        foreach(i -> a[i, 1, i] = 1, 1:dim(V))
        return a
    end
    into(k, r, c) = (V = domain(ρ[r, c])[k]; W = domain(S[r, c])[k]; TensorMap(idle(V, W), V ⊗ W ← V))
    function outof(k, r, c)
        V, W = domain(ρ[r, c])[k]', domain(S[r, c])[k]'
        return TensorMap(permutedims(idle(V, W), (3, 1, 2)), V ← V ⊗ W)
    end
    Pxs = Any[(into(2, r, c), outof(4, r, mod1(c + 1, Nc))) for r in 1:Nr, c in 1:Nc]
    Pys = Any[(into(1, r, c), outof(3, mod1(r - 1, Nr), c)) for r in 1:Nr, c in 1:Nc]
    return Pxs, Pys
end

function _projectorsfit(stored, F, λ)
    isnothing(stored) && return false
    Pxs, Pys = stored
    size(Pxs) == size(F) || return false
    Nr, Nc = size(F)
    return all(Iterators.product(1:Nr, 1:Nc)) do (r, c)
        Pe, Pw = Pxs[r, c]
        Pn, Ps = Pys[r, c]
        codomain(Pe) == codomain(F[r, c].Fe) && domain(Pw) == domain(F[r, mod1(c + 1, Nc)].Fw) &&
            codomain(Pn) == codomain(F[r, c].Fn) && domain(Ps) == domain(F[mod1(r - 1, Nr), c].Fs) &&
            dim(domain(Pe)) == length(λ[1, r, c]) && dim(domain(Pn)) == length(λ[2, r, c])
    end
end

"""
    $(TYPEDSIGNATURES)

Apply the long-range propagator of `layer` to `state` by iterative simple update, updating `state`,
`weights` and the projectors `layer` keeps for the next step. Returns the largest relative
truncation error of the last round of cuts.

Every bond is cut against the projectors the previous round chose for the other legs of its two
sites, those legs weighted by their bond weights as in simple update, and the rounds repeat until no
weight changes by more than `alg.iterative.tol`. The bond dimension comes from `alg.trunc`, or is
kept when that is `nothing`. A round starts from the projectors of the layer's last application
when they fit, and otherwise from projectors onto the propagator's idle state.
"""
function iterativestep!(
        state::InfinitePEPO, weights::SUWeight, layer::PEPSOLayer, alg::DampedUpdate
    )
    isnothing(alg.truncalg) || throw(
        ArgumentError(
            "a long-range layer is cut by iterative simple update and takes no `truncalg`"
        ),
    )
    isinf(alg.zeta) || throw(
        ArgumentError("damping is not implemented for a long-range layer; use `zeta = Inf`"),
    )
    settings = alg.iterative
    Nr, Nc, = size(state)
    G = layer.propagator
    ρ = [state[r, c] for r in 1:Nr, c in 1:Nc]
    S = [G[mod1(r, size(G, 1)), mod1(c, size(G, 2)), 1] for r in 1:Nr, c in 1:Nc]
    F = map(_fusers, ρ, S)
    bonds = CartesianIndices((2, Nr, Nc))
    truncs = map(bonds) do I
        isnothing(alg.trunc) || return alg.trunc
        V = space(weights[Tuple(I)...], 1)
        return truncspace(isdual(V) ? flip(V) : V)
    end

    λ = map(bonds) do I
        s = _weightvalues(weights[Tuple(I)...])
        return s ./ maximum(s)
    end
    Pxs, Pys = _projectorsfit(layer.projectors, F, λ) ? layer.projectors : _idleprojectors(ρ, S)

    msgs = layer.messages
    cuts = nothing
    for _ in 1:settings.maxiter
        P = [PEPSKit._site_projectors(Pxs, Pys, r, c) for r in 1:Nr, c in 1:Nc]
        sP = [_weighted(P[r, c], _siteweights(λ, r, c)) for r in 1:Nr, c in 1:Nc]
        q = nothing
        if settings.protect !== :none
            traced = map(Iterators.product(1:Nr, 1:Nc)) do (r, c)
                Threads.@spawn begin
                    A = convert(Array, _composite(ρ[r, c], S[r, c], P[r, c]...))
                    sum(A[i, i, :, :, :, :] for i in 1:min(size(A, 1), size(A, 2)))
                end
            end
            msgs = _tracemessages(fetch.(traced); init = msgs)
            # the iterate's legs carry √λ, so the messages are divided by it to contract the bare
            # tensors; a direction with zero weight is absent from the state and is dropped
            q = map(CartesianIndices(msgs)) do I
                d, r, c = Tuple(I)
                w = sqrt.(_siteweights(λ, r, c)[d])
                return [abs(b) > 1e-14 ? a / b : zero(a) for (a, b) in zip(msgs[I], w)]
            end
        end
        tasks = map(bonds) do I
            Threads.@spawn _iterativecut(I, ρ, S, F, sP, q, truncs[I], settings)
        end
        cuts = fetch.(tasks)

        δ = maximum(bonds) do I
            s = _weightvalues(cuts[I][2])
            s ./= maximum(s)
            length(s) == length(λ[I]) || return Inf
            return maximum(abs.(sort(s) .- sort(λ[I])))
        end
        for I in bonds
            d, r, c = Tuple(I)
            (d == 1 ? Pxs : Pys)[r, c] = cuts[I][1]
            s = _weightvalues(cuts[I][2])
            λ[I] = s ./ maximum(s)
        end
        δ < settings.tol && break
    end

    P = [PEPSKit._site_projectors(Pxs, Pys, r, c) for r in 1:Nr, c in 1:Nc]
    for r in 1:Nr, c in 1:Nc
        state[r, c] = normalize!(_composite(ρ[r, c], S[r, c], P[r, c]...), Inf)
    end
    for I in bonds
        weights[Tuple(I)...] = normalize!(cuts[I][2], Inf)
    end
    layer.projectors = (Pxs, Pys)
    layer.messages = msgs
    return maximum(I -> cuts[I][3], bonds)
end

# One bond of a round: `I = (1, r, c)` joins (r, c) to (r, c + 1), `I = (2, r, c)` joins (r, c) to
# (r - 1, c). Returns the bond's two projectors, its weight and its truncation error.
function _iterativecut(I, ρ, S, F, sP, q, trunc, settings::IterativeSU)
    Nr, Nc = size(ρ)
    d, r, c = Tuple(I)
    if d == 1
        r2, c2 = r, mod1(c + 1, Nc)
        Pn, _, Ps, Pw = sP[r, c]
        tX = Threads.@spawn _composite(ρ[r, c], S[r, c], Pn, F[r, c].Fe, Ps, Pw; last = :e)
        Pn, Pe, Ps, _ = sP[r2, c2]
        Y = _composite(ρ[r2, c2], S[r2, c2], Pn, Pe, Ps, F[r2, c2].Fw; last = :w)
        X = fetch(tX)
        kx, ky, open1, open2 = 2, 4, F[r, c].Fe, F[r2, c2].Fw
    else
        r2, c2 = mod1(r - 1, Nr), c
        _, Pe, Ps, Pw = sP[r, c]
        tX = Threads.@spawn _composite(ρ[r, c], S[r, c], F[r, c].Fn, Pe, Ps, Pw; last = :n)
        Pn, Pe, _, Pw = sP[r2, c2]
        Y = _composite(ρ[r2, c2], S[r2, c2], Pn, Pe, F[r2, c2].Fs, Pw; last = :s)
        X = fetch(tX)
        kx, ky, open1, open2 = 1, 3, F[r, c].Fn, F[r2, c2].Fs
    end
    wL = wR = nothing
    if settings.protect !== :none
        wL = _protected(X, kx, Tuple(q[k, r, c] for k in 1:4), settings.protect)
        wR = permutedims(_protected(Y, ky, Tuple(q[k, r2, c2] for k in 1:4), settings.protect))
    end
    P1, P2, s, ϵ = _cutbond(X, kx, Y, ky, wL, wR, trunc, settings)
    return (open1 * P1, P2 * open2), s, ϵ
end
