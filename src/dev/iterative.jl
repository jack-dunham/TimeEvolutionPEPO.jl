# Iterative simple update (FCLS manuscript, Appendix D) for a long-range propagator: the
# `InfinitePEPSO` is contracted onto the state, and every bond is cut against the current projectors
# of the bonds around it, repeating until the bond weights stop changing.

"""
$(TYPEDEF)

How [`iterativestep!`](@ref) cuts the bonds when it applies a long-range propagator.

# Constructors

    IterativeSU(; maxiter = 20, tol = 1e-6, mu = 0.0)

- `maxiter`, `tol`: repeat the cuts until the bond weights change by less than `tol`, as PEPSKit's
  `compare_weights` measures it, at most `maxiter` times
- `mu`: `0` keeps each bond's top singular directions. A positive value also penalises the relative
  change, weighted by `mu`, of the two-site density matrix across the bond as belief propagation on
  the traced network sees it; this needs spaces without symmetry

The contractions run in parallel under PEPSKit's scheduler (`PEPSKit.Defaults.set_scheduler!`).
"""
struct IterativeSU
    maxiter::Int
    tol::Float64
    mu::Float64

    function IterativeSU(maxiter, tol, mu)
        maxiter >= 1 || throw(ArgumentError("`maxiter` must be at least 1, got $maxiter"))
        mu >= 0 || throw(ArgumentError("the penalty weight `mu` must be non-negative, got $mu"))
        return new(maxiter, tol, mu)
    end
end

function IterativeSU(; maxiter = 20, tol = 1e-6, mu = 0.0)
    return IterativeSU(maxiter, tol, mu)
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

# Maps between each leg pair of the two-layer network at a site and one fused leg, north to west: the
# north and east maps fuse their pairs, the south and west ones split their legs back into pairs.
function _fusers(ρ::AbstractTensorMap, S::AbstractTensorMap)
    Nr, Er, Sr, Wr = domain(ρ)
    Ns, Es, Ss, Ws = domain(S)
    return (
        isomorphism(Nr ⊗ Ns, fuse(Nr ⊗ Ns)),
        isomorphism(Er ⊗ Es, fuse(Er ⊗ Es)),
        isomorphism(fuse(Sr' ⊗ Ss'), Sr' ⊗ Ss'),
        isomorphism(fuse(Wr' ⊗ Ws'), Wr' ⊗ Ws'),
    )
end

"""
    $(TYPEDSIGNATURES)

The site tensor `ρ` with the propagator `S` applied and each pair of virtual legs mapped to one leg by
`maps = (Pn, Pe, Ps, Pw)`, laid out as PEPSKit's `compress` projectors, shaped as a `PEPOTensor`. The
legs are relabelled so that leg `big` (north, east, south, west) is contracted last; passing the leg
that carries a fuser rather than a projector keeps the intermediates small.
"""
function _composite(ρ, S, maps; big::Int = WEST)
    o = Tuple(circshift(1:4, 4 - big))
    ρo = permute(ρ, ((1, 2), 2 .+ o))
    So = permute(S, ((1, 2, 3, 4), 4 .+ o))
    # every map with its leg pair in the codomain, as `Pn` and `Pe` already have it
    Ma, Mb, Mc, Md = map(k -> k <= EAST ? maps[k] : permute(maps[k], ((2, 3), (1,))), o)
    @tensor allocator = TensorOperations.ManualAllocator() order = (
        b2, a2, c1, q1, q2, a1, b1, c2, d1, d2,
    ) A[p1 p2; a b c d] :=
        ρo[q1 q2; a1 b1 c1 d1] * So[p1 p2 q1 q2; a2 b2 c2 d2] *
        Ma[a1 a2; a] * Mb[b1 b2; b] * Mc[c1 c2; c] * Md[d1 d2; d]
    return permute(A, ((1, 2), 2 .+ invperm(o)))
end

# R1, R2 with X = Q1 R1 and Y = R2 Q2 across the bond joining domain leg `kx` of `X` to `ky` of `Y`.
function _bondfactors(X, kx, Y, ky)
    _, R1 = left_orth!(permute(X, (Tuple(setdiff(1:6, kx + 2)), (kx + 2,))))
    R2, _ = right_orth!(permute(Y, ((ky + 2,), Tuple(setdiff(1:6, ky + 2)))))
    return R1, R2
end

# Cut the bond M = R1 R2 into PEPSKit's `(Pa, s, Pb)`, keeping the top singular directions for
# `mu = 0`, and return the relative truncation error with them.
function _cutbond(R1, R2, wL, wR; trunc, mu)
    iszero(mu) || return _softcut(R1, R2, wL, wR; trunc, mu)
    Pa, s, Pb, ϵ = PEPSKit._proj_from_RL(R1, R2; trunc)
    return Pa, s, Pb, ϵ / norm(R1 * R2)
end

# The cut minimising ‖L (M - M̃) R‖ for L² = 1 + μ Y'Y and R² = 1 + μ ZZ', where Y M = wL R2 and
# M Z = R1 wR are scaled to the norm of M, so that μ weighs the relative change of wL M̃ and M̃ wR.
function _softcut(R1, R2, wL, wR; trunc, mu)
    M = R1 * R2
    Y = wL * pinv(R1; rtol = 1e-12)
    Z = pinv(R2; rtol = 1e-12) * wR
    Y *= norm(M) / norm(Y * M)
    Z *= norm(M) / norm(M * Z)
    L = sqrt(id(domain(Y)) + mu * Y' * Y)
    R = sqrt(id(codomain(Z)) + mu * Z * Z')
    Pa, _, Pb, _ = PEPSKit._proj_from_RL(L * R1, R2 * R; trunc)
    # R1 Pa is not an isometry times √s, so the cut is split again by its own singular values
    _, ra = left_orth(R1 * Pa)
    rb, _ = right_orth(Pb * R2)
    qa, s, qb, _ = PEPSKit._proj_from_RL(ra, rb)
    Pa, Pb = Pa * qa, qb * Pb
    return Pa, s, Pb, norm(M - R1 * Pa * Pb * R2) / norm(M)
end

_normmax(v) = v / v[argmax(abs.(v))]

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

# `X` with the messages `m` (north, east, south, west) contracted onto every leg but `open`: a map
# from the open leg to the site's physical pair, whose rows the soft cut protects.
function _protected(X, open::Int, m)
    others = filter(!=(open), 1:4)
    Xp = permute(X, ((1, 2), (2 .+ others..., 2 + open)))
    env = foldr(⊗, [Tensor(m[k], domain(X)[k]) for k in others]) ⊗ id(domain(X)[open])
    return Xp * env
end

# Projectors onto the state's bond times the propagator's idle state: the exact cut of the composite
# before any path runs. The machine's other states have entries that continue or turn a path whether
# or not one was started, so a cut ranking them by norm alone would keep directions no path reaches.
function _idleprojectors(ρ, S)
    Nr, Nc = size(ρ)
    idle(W) = Tensor([i == 1 ? one(ComplexF64) : zero(ComplexF64) for i in 1:dim(W)], W)
    into((r, c), k) = id(domain(ρ[r, c])[k]) ⊗ idle(domain(S[r, c])[k])
    outof((r, c), k) = id(domain(ρ[r, c])[k]') ⊗ idle(domain(S[r, c])[k]')'
    Pxs = Any[(into((r, c), EAST), outof((r, mod1(c + 1, Nc)), WEST)) for r in 1:Nr, c in 1:Nc]
    Pys = Any[(into((r, c), NORTH), outof((mod1(r - 1, Nr), c), SOUTH)) for r in 1:Nr, c in 1:Nc]
    return Pxs, Pys
end

# Whether the projectors a layer stored act between the same spaces as `start`, so can replace it.
function _samespaces(stored, start)
    return !isnothing(stored) && map.(p -> space.(p), stored) == map.(p -> space.(p), start)
end

"""
    $(TYPEDSIGNATURES)

Apply the long-range propagator of `layer` to `state` by iterative simple update, updating `state`,
`weights` and the projectors `layer` keeps for the next step. Returns the largest relative
truncation error of the last round of cuts.

Every bond is cut against the projectors the previous round chose for the other legs of its two
sites, those legs weighted by their bond weights as in simple update, and the rounds repeat until the
weights change by less than `alg.iterative.tol`. The bond dimension comes from `alg.trunc`, or is
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
    (; maxiter, tol, mu) = alg.iterative
    iszero(mu) || sectortype(state[1, 1]) == Trivial ||
        throw(ArgumentError("the soft cut (`mu > 0`) needs spaces without symmetry"))
    Nr, Nc, = size(state)
    ρ = [state[r, c] for r in 1:Nr, c in 1:Nc]
    S = [layer.propagator[r, c, 1] for r in 1:Nr, c in 1:Nc]
    F = map(_fusers, ρ, S)
    bonds = CartesianIndices((2, Nr, Nc))
    truncs = map(bonds) do I
        trunc = if isnothing(alg.trunc)
            V = space(weights[Tuple(I)...], 1)
            truncspace(isdual(V) ? flip(V) : V)
        else
            alg.trunc
        end
        # dropping vanishing directions keeps the bond at the rank of the cut when that is below D
        return trunc & trunctol(; rtol = 1e-14)
    end

    start = _idleprojectors(ρ, S)
    Pxs, Pys = _samespaces(layer.projectors, start) ? layer.projectors : start
    λ, msgs, cuts = weights, layer.messages, nothing
    for _ in 1:maxiter
        cuts, msgs = _iterativeround(ρ, S, F, _siteprojectors(Pxs, Pys), λ, msgs, truncs, mu)
        for I in bonds
            d, r, c = Tuple(I)
            (d == 1 ? Pxs : Pys)[r, c] = cuts[I][1]
        end
        λ′ = SUWeight(map(cut -> normalize(cut[2], Inf), cuts))
        converged = PEPSKit.compare_weights(λ′, λ) < tol
        λ = λ′
        converged && break
    end

    P = _siteprojectors(Pxs, Pys)
    As = PEPSKit.dtmap(I -> normalize!(_composite(ρ[I], S[I], P[I]), Inf), CartesianIndices(P))
    for r in 1:Nr, c in 1:Nc
        state[r, c] = As[r, c]
    end
    for I in bonds
        weights[Tuple(I)...] = λ[Tuple(I)...]
    end
    layer.projectors = (Pxs, Pys)
    layer.messages = msgs
    return maximum(cut -> cut[3], cuts)
end

function _siteprojectors(Pxs, Pys)
    return [PEPSKit._site_projectors(Pxs, Pys, Tuple(I)...) for I in CartesianIndices(Pxs)]
end

# One round: the traced-network messages when `mu > 0`, then every bond cut against projectors `P`.
function _iterativeround(ρ, S, F, P, λ, msgs, truncs, mu)
    m = if iszero(mu)
        msgs
    else
        traced = PEPSKit.dtmap(CartesianIndices(P)) do I
            A = convert(Array, _composite(ρ[I], S[I], P[I]))
            return sum(A[i, i, :, :, :, :] for i in axes(A, 1))
        end
        _tracemessages(traced; init = msgs)
    end
    cuts = PEPSKit.dtmap(CartesianIndices(truncs)) do I
        return _iterativecut(I, ρ, S, F, P, λ, m, truncs[I], mu)
    end
    return cuts, m
end

# One bond of a round: `I = (1, r, c)` joins (r, c) to (r, c + 1), `I = (2, r, c)` joins (r, c) to
# (r - 1, c). Returns the bond's two projectors, its weight and its truncation error.
function _iterativecut(I, ρ, S, F, P, λ, msgs, trunc, mu)
    Nr, Nc = size(ρ)
    d, r, c = Tuple(I)
    # the legs facing each other across the bond, and the sites they belong to
    kx, ky = d == 1 ? (EAST, WEST) : (NORTH, SOUTH)
    x, y = (r, c), d == 1 ? (r, mod1(c + 1, Nc)) : (mod1(r - 1, Nr), c)
    X, Y = PEPSKit.dtmap([(x, kx), (y, ky)]) do (site, k)
        maps = ntuple(j -> j == k ? F[site...][k] : P[site...][j], 4)
        return _composite(ρ[site...], S[site...], maps; big = k)
    end
    closed(k) = Tuple(filter(!=(k), 1:4))
    R1, R2 = _bondfactors(
        absorb_weight(X, λ, x..., closed(kx)), kx, absorb_weight(Y, λ, y..., closed(ky)), ky
    )
    wL = wR = nothing
    if !iszero(mu)
        wL = _protected(X, kx, msgs[:, x...])
        wR = permute(_protected(Y, ky, msgs[:, y...]), ((3,), (1, 2)))
    end
    Pa, s, Pb, ϵ = _cutbond(R1, R2, wL, wR; trunc, mu)
    return (F[x...][kx] * Pa, Pb * F[y...][ky]), s, ϵ
end
