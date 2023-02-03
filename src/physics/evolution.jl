abstract type AbstractSimulation{TruncAlg<:AbstractTruncationAlgorithm} end

struct TimeEvolutionSimulation{TruncAlg,P<:AbstractPEPO,L<:AbstractSuperOperator} <:
       AbstractSimulation{TruncAlg}
    state::P
    liouvillian::L
    duration::Float64
    numsteps::Int
    truncalg::TruncAlg # one of SU, FET, or NTU. 
    initialstate::P
end

function simulate!(sim::TimeEvolutionSimulation)
    t = sim.duration
    n = sim.numsteps
    P = sim.state
    L = sim.liouvillian
    alg = sim.truncalg

    τ = t / n # Calculate the trotter timestep.
    # Op = exp.(τ .* L) # Construct the time evolution operator from the semi-group generator.

    Lx = L.x
    Ly = L.y

    Op_x = @. twolocal_decompose(exp(τ * Lx))
    Op_y = @. twolocal_decompose(exp(τ * Ly))

    propagate!(P, Op_x, Op_y, n, alg)

    return P
end

function propagate!(P, Op_x, Op_y, n, alg::FET{Alg}) where {Alg}
    # Unroll algorithm parameters.
    recycle = alg.recycle   # Whether to recycle the metric
    gauge = alg.gauge       # Whether to fix the gauge of the metric
    ξ = alg.ξ               # Correlation length scaling parameter
    param = alg.boundary    # Parameters of the boundary algorithm

    # Z = TensorMap(pauli[1], ℂ^2, ℂ^2)
    # Z_exp = tr.(Ref(Z) .* onelocal(P))
    # @warn "Initial mag: $((Z_exp))"
    # First time step.
    Bx_i = By_i = x -> initialise_state(Alg, x, param)
    # Apply the time evolution operator, calculate the metric, and then truncate.
    P, Bx, By = trotterstep!(P, Op_x, Op_y, Bx_i, By_i; ξ=ξ, gauge=gauge)

    # Z = TensorMap(pauli[1], ℂ^2, ℂ^2)
    # Z_exp = tr.(Ref(Z) .* onelocal(P))
    # @warn "$((Z_exp))"
    # Remaining time steps.
    for i in 2:(n)
        @info "Trotter step $i"
        # If we wish to recycle the metric, set the function that returns the initial guess
        # for the boundary to simply return the boundary of the previous time step.
        if recycle == true
            Bx_i = x -> initialise_state(Bx, x)
            By_i = x -> initialise_state(By, x)
        end

        # Apply the time evolution operator, calculate the metric, and then truncate.
        P, Bx, By = trotterstep!(P, Op_x, Op_y, Bx_i, By_i; ξ=ξ, gauge=gauge)

        # Measure 
        X = TensorMap(pauli[1], ℂ^2, ℂ^2)
        X_exp = tr.(Ref(X) .* onelocal(P))
        @warn "X: $((X_exp))"

        Y = TensorMap(pauli[2], ℂ^2, ℂ^2)
        Y_exp = tr.(Ref(Y) .* onelocal(P))
        @warn "X: $((Y_exp))"

        Z = TensorMap(pauli[3], ℂ^2, ℂ^2)
        Z_exp = tr.(Ref(Z) .* onelocal(P))
        @warn "Z: $((Z_exp))"
    end

    # Return the PEPO state after n applications of the operator.
    return P
end

function trotterstep!(pepo, op_tuple_x, op_tuple_y, bx_i, by_i; kwargs...)
    pepo, bx = axistrotterstep!(pepo, op_tuple_x, bx_i; kwargs...)

    perm = (2, 1, 3, 5, 4, 6)
    pepo = _permute.(pepo, Ref(perm))

    pepo, by = axistrotterstep!(pepo, op_tuple_y, by_i; kwargs...)

    pepo = _permute.(pepo, Ref(invperm(perm)))

    return pepo, bx, by
end

function axistrotterstep!(pepo, op_tuple, get_initialboundary; kwargs...)
    pepo_hs = hilbertschmidt(pepo)

    boundary = get_initialboundary(pepo_hs)
    boundary = calculate!(boundary)

    updatestep!(pepo, op_tuple, boundary, truncspace(codomain(pepo[1, 1])[1]))

    return pepo, boundary
end

# m[s_b s_ph' s_ph; s_b s_ph' s_ph] (l,b,k ; r b k)
function _evolve(
    p::AbstractTensorMap{S,3,3}, m::AbstractTensorMap{S,3,3}
) where {S<:IndexSpace}
    return @tensor pe[w w2 s k; e e2 n b] := m[w2 b k; e2 mb mk] * p[w s mk; e n mb]
end
function _evolve(pa, pb, L)
    @tensoropt pab[w sa ka na ba; sb kb e nb bb] :=
        pa[w sa mka; x na mba] * pb[x sb mkb; e nb mbb] * L[ba ka bb kb; mba mka mbb mkb]
end

function splitab(pab; trunc=notrunc())
    pa, s, pb = tsvd(pab; trunc=trunc)
    return permute(pa, (1, 2, 3), (6, 4, 5)), s, permute(pb, (1, 2, 3), (4, 5, 6))
end

# Left Q . R
function pepoqr(pe_a)
    Q, R = leftorth(pe_a, (1, 2, 3, 4, 7, 8), (5, 6))
    return Q, R
end

# Right L . Q
function pepolq(pe_a)
    L, Q = rightorth(pe_a, (1, 2), (3, 4, 5, 6, 7, 8))
    return L, Q
end

function evolvetrunc(p, m; trunc=truncerr(1e-10))
    pe = _evolve(p, m)
    Qr, R = leftorth(pe, (1, 2, 3, 4, 7, 8), (5, 6))
    Qrp = permute(Qr, (1, 2, 3, 4), (7, 5, 6))
    L, Ql = rightorth(Qrp, (1, 2), (3, 4, 5, 6, 7))
    Q = permute(Ql, (1, 2, 3), (4, 5, 6))
    U, S, V = tsvd(R * L; trunc=trunc)
    println("dim: ", dim(codomain(S)))
    Mr = U * sqrt(S)
    Ml = sqrt(S) * V
    @tensoropt p_n[-1 -2 k; -4 -5 b] := Q[1 -2 k; 4 -5 b] * Ml[-1; 1] * Mr[4; -4]
    return p_n
end

function evolve(
    p::AbstractTensorMap{S,3,3}, m::AbstractTensorMap{S,3,3}
) where {S<:IndexSpace}
    t = _evolve(p, m)
    return merge_bonds(t)
end

evolve(P::AbstractPEPO, M) = evolve.(P, M)

function evolveleft(pepo_left, op_left)
    Q, L = leftorth(_evolveleft(pepo_left, op_left))
    return permute(Q, (1, 2, 3), (6, 4, 5)), L
end
function evolveleft_alt(pepo_left, op_left)
    pepo_new = _evolveleft(pepo_left, op_left)

    Q, L = leftorth(pepo_new)
    Q = permute(Q, (1, 2, 3), (6, 4, 5))

    pp = permute(pepo_new, (1, 2, 3), (6, 7, 4, 5))
    c = codomain(pp)
    d = domain(pp)
    A = TensorMap(pp.data, c, fuse(d[1] * d[2]) * d[3] * d[4])

    return A, Q, L
end

function evolveright(pepo_right, op_right)
    R, Q = rightorth(_evolveright(pepo_right, op_right))
    return R, permute(Q, (1, 2, 3), (4, 5, 6))
end
function evolveright_alt(pepo_right, op_right)
    pepo_new = _evolveright(pepo_right, op_right)

    R, Q = rightorth(pepo_new)
    Q = permute(Q, (1, 2, 3), (4, 5, 6))

    pp = permute(pepo_new, (1, 2, 3, 4), (5, 6, 7))
    c = codomain(pp)
    d = domain(pp)
    B = TensorMap(pp.data, fuse(c[1] * c[2]) * c[3] * c[4], d)

    return B, R, Q
end

function _evolveleft(pl, op)
    @tensoropt pl_n[w s k n b; e1 e2] := pl[w s mk; e1 n mb] * op[b k; e2 mb mk]
    return pl_n
end
function _evolveright(pr, mr)
    @tensoropt pr_n[w1 w2; s k e n b] := pr[w1 s mk; e n mb] * mr[w2 b k; mb mk]
    return pr_n
end
