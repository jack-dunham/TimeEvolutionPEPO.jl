abstract type AbstractSimulation{TruncAlg<:AbstractTruncationScheme} end

# function initsimulation(
#     ::Type{<:AbstractSimulation},
#     size::HilbertSpace,
#     model::Model,
#     initialstate,
#     # Simulation specific parameters
#     numsteps=100,
#     timestep=0.1,
#     d=BondSpace(size, ℂ^2), # Either IndexSpace, Matrix{IndexSpace}
#     D=ℂ^2, # Either an IndexSpace, Tuple{IndexSpace}, Matrix{Tuple{IndexSpace}}
#     initialstate=[1, 0, 0],
#     truncalg=FET{VUMPS},
# ) end

struct TimeEvolution{TruncAlg,P<:AbstractPEPO,L<:AbstractSuperOperator} <:
       AbstractSimulation{TruncAlg}
    # Necessary
    state::P
    liouvillian::L
    # Have defaults
    timestep::Float64       # Timestep
    numsteps::Int           # Number of time steps.
    tol::Float64
    truncalg::TruncAlg      # one of SU, FET, or CLU
    initialstate::P         # Initial state
    # Simulation agnostic
    info::ConvergenceInfo   # Information about convergence.
    # 0: Save nothing
    # 1: Save just the convergence information at each time step
    # 2: Save just the quantum state at each time step 
    # 3: Save both the quantum state and the convergence info at each time step
    save::Int
    outfile::String
    outdir::String
end

function TimeEvolution(
    state,
    liouvillian;
    #KWARGS
    timestep=0.01,
    numsteps=100,
    tol=0.0,
    truncalg=SU(),
    kwargs...,
)
    return TimeEvolution(
        state, liouvillian, timestep, numsteps, tol, truncalg, state; kwargs...
    )
end

function (::Type{Sim})(
    vargs...; save=0, outfile="", outdir=pwd()
) where {TruncAlg,Sim<:AbstractSimulation{<:TruncAlg}}
    return Sim(vargs..., ConvergenceInfo(), save, outfile, outdir)
end

# function initsimulation(
#     ::Type{TimeEvolutionSimulation},
#     initialstate,
#     liouvillian,
#     timestep::Float64,
#     numsteps::Int,
#     truncalg::TruncAlg;
#     save::Int=0,
#     outfile::String="",
#     outdir::String=pwd(),
# )
#     rv = TimeEvolutionSimulation(
#         initialstate,
#         liouvillian,
#         timestep,
#         numsteps,
#         truncalg,
#         initialstate,
#         ConvergenceInfo(),
#         save,
#         outfile,
#     )
#     return rv
# end

function simulate!(sim::TimeEvolution)
    τ = sim.timestep
    n = sim.numsteps
    P = sim.state
    L = sim.liouvillian
    alg = sim.truncalg
    info = sim.info

    # Op = exp.(τ .* L) # Construct the time evolution operator from the semi-group generator.

    Lx = L.x
    Ly = L.y

    Op_x = @. twolocal_decompose(exp(τ * Lx))
    Op_y = @. twolocal_decompose(exp(τ * Ly))

    Nx, Ny = size(P)

    if Nx == 1 && Ny == 1
        twosite_lattice = Infinite{2,2}()
        P = OnLattice(tensorstyle(P), twosite_lattice, P[1, 1])
        Nx = Ny = 2
    end

    out = propagate!(P, Op_x, Op_y, n, alg)

    return out
end

function propagate!(P, Op_x, Op_y, n, alg::CLU{NTU})
    # Unroll algorithm parameters
    gauge = alg.gauge
    ξ = alg.ξ

    Bx = By = initialise_state(NTU, P)

    out = _propagate!(P, Op_x, Op_y, Bx, By, n; gauge=gauge, ξ=ξ)
    return out
end
function propagate!(P, Op_x, Op_y, n, alg::FET{Alg}) where {Alg}
    # Unroll algorithm parameters
    gauge = alg.gauge
    ξ = alg.ξ
    recycle = alg.recycle   # Whether to recycle the metric
    param = alg.boundary    # Parameters of the boundary algorithm

    # Get the initial FET boundary
    Bx = By = initialise_state(Alg, hilbertschmidt(P), param)

    out = _propagate!(P, Op_x, Op_y, Bx, By, n; recycle=recycle, gauge=gauge, ξ=ξ)

    return out
end

function _propagate!(P, Op_x, Op_y, Bx, By, n; kwargs...)
    Nx, Ny = size(P)

    S_x = Matrix{Matrix{Float64}}(undef, Nx, Ny)
    S_y = Matrix{Matrix{Float64}}(undef, Nx, Ny)

    @info "Trotter step: 1"
    P, Bx, By = trotterstep!(P, S_x, S_y, Op_x, Op_y, Bx, By; init=true, kwargs...)

    P0 = onelocal(P)

    X = TensorMap(pauli[1], ℂ^2, ℂ^2)
    X_exp = tr.(Ref(X) .* P0)
    @warn "X: $((X_exp))"

    Y = TensorMap(pauli[2], ℂ^2, ℂ^2)
    Y_exp = tr.(Ref(Y) .* P0)
    @warn "Y: $((Y_exp))"

    Z = TensorMap(pauli[3], ℂ^2, ℂ^2)
    Z_exp = tr.(Ref(Z) .* P0)
    @warn "Z: $((Z_exp))"

    obs_X = Vector{typeof(X_exp)}([])
    obs_Y = Vector{typeof(Y_exp)}([])
    obs_Z = Vector{typeof(Z_exp)}([])

    append!(obs_X, Ref(X_exp))
    append!(obs_Y, Ref(Y_exp))
    append!(obs_Z, Ref(Z_exp))

    for i in 2:n
        @info "Trotter step: $i"

        P, Bx, By = trotterstep!(P, S_x, S_y, Op_x, Op_y, Bx, By; kwargs...)

        P0 = onelocal(P)

        X = TensorMap(pauli[1], ℂ^2, ℂ^2)
        X_exp = tr.(Ref(X) .* P0)
        @warn "X: $((X_exp))"

        Y = TensorMap(pauli[2], ℂ^2, ℂ^2)
        Y_exp = tr.(Ref(Y) .* P0)
        @warn "Y: $((Y_exp))"

        Z = TensorMap(pauli[3], ℂ^2, ℂ^2)
        Z_exp = tr.(Ref(Z) .* P0)
        @warn "Z: $((Z_exp))"

        append!(obs_X, Ref(X_exp))
        append!(obs_Y, Ref(Y_exp))
        append!(obs_Z, Ref(Z_exp))
    end
    return obs_X, obs_Y, obs_Z
end

function __propagate!(P, Op_x, Op_y, n, alg::FET{Alg}) where {Alg}
    # Unroll algorithm parameters.
    recycle = alg.recycle   # Whether to recycle the metric
    gauge = alg.gauge       # Whether to fix the gauge of the metric
    ξ = alg.ξ               # Correlation length scaling parameter
    param = alg.boundary    # Parameters of the boundary algorithm

    # Z = TensorMap(pauli[1], ℂ^2, ℂ^2)
    # Z_exp = tr.(Ref(Z) .* onelocal(P))
    # @warn "Initial mag: $((Z_exp))"

    Nx, Ny = size(P)

    if Nx == 1 && Ny == 1
        twosite_lattice = Infinite{2,2}()
        P = OnLattice(tensorstyle(P), twosite_lattice, P[1, 1])
        Nx = Ny = 2
    end

    S_x = Matrix{Matrix{Float64}}(undef, Nx, Ny)
    S_y = Matrix{Matrix{Float64}}(undef, Nx, Ny)

    err = []

    # First time step.
    @info "Trotter step 1"
    Bx_i = By_i = x -> initialise_state(Alg, x, param)
    # Apply the time evolution operator, calculate the metric, and then truncate.
    P, Bx, By = trotterstep!(P, S_x, S_y, Op_x, Op_y, Bx_i, By_i; ξ=ξ, gauge=gauge)

    X = TensorMap(pauli[1], ℂ^2, ℂ^2)
    X_exp = tr.(Ref(X) .* onelocal(P))
    @warn "X: $((X_exp))"

    Y = TensorMap(pauli[2], ℂ^2, ℂ^2)
    Y_exp = tr.(Ref(Y) .* onelocal(P))
    @warn "Y: $((Y_exp))"

    Z = TensorMap(pauli[3], ℂ^2, ℂ^2)
    Z_exp = tr.(Ref(Z) .* onelocal(P))
    @warn "Z: $((Z_exp))"

    obs_x = Vector{typeof(X_exp)}([])
    obs_y = Vector{typeof(Y_exp)}([])
    obs_z = Vector{typeof(Z_exp)}([])

    append!(obs_x, Ref(X_exp))
    append!(obs_y, Ref(Y_exp))
    append!(obs_z, Ref(Z_exp))

    # Remaining time steps.

    for i in 2:(n)
        @info "Trotter step $i"

        # Apply the time evolution operator, calculate the metric, and then truncate.
        S_x_old = deepcopy(S_x)
        S_y_old = deepcopy(S_y)

        P, Bx, By = trotterstep!(P, S_x, S_y, Op_x, Op_y, Bx, By; ξ=ξ, gauge=gauge)

        # ϵ_x = findconvergence.(S_x, S_x_old)
        # ϵ_y = findconvergence.(S_y, S_y_old)
        # #
        # ϵ = max(ϵ_x..., ϵ_y...)

        @info "Converged to: $ϵ"

        # Measure 
        X = TensorMap(pauli[1], ℂ^2, ℂ^2)
        X_exp = tr.(Ref(X) .* onelocal(P))
        @warn "X: $((X_exp))"

        Y = TensorMap(pauli[2], ℂ^2, ℂ^2)
        Y_exp = tr.(Ref(Y) .* onelocal(P))
        @warn "Y: $((Y_exp))"

        Z = TensorMap(pauli[3], ℂ^2, ℂ^2)
        Z_exp = tr.(Ref(Z) .* onelocal(P))
        @warn "Z: $((Z_exp))"

        append!(obs_x, Ref(X_exp))
        append!(obs_y, Ref(Y_exp))
        append!(obs_z, Ref(Z_exp))
        # append!(err, ϵ)
    end

    # Return the PEPO state after n applications of the operator.
    return obs_x, obs_y, obs_z, err
end

function findconvergence(s, s_old)
    return max(abs.(s .- s_old)...)
end

function trotterstep!(
    pepo, coeffs_x, coeffs_y, op_tuple_x, op_tuple_y, bx_i, by_i; kwargs...
)
    pepo, bx = axistrotterstep!(pepo, coeffs_x, op_tuple_x, bx_i; kwargs...)

    perm = (2, 1, 3, 5, 4, 6)
    pepo = _permute.(pepo, Ref(perm))

    pepo, by = axistrotterstep!(pepo, coeffs_y, op_tuple_y, by_i; kwargs...)

    pepo = _permute.(pepo, Ref(invperm(perm)))

    return pepo, bx, by
end

function axistrotterstep!(
    pepo, coeffs, op_tuple, boundary; recycle=false, init=false, kwargs...
)
    # If this is the first trotterstep, then then hilbertschmidt tensor has already 
    # been calculated, so skip it.
    if !init
        pepo_hs = hilbertschmidt(pepo)
        if recycle
            boundary = initialise_state(boundary, pepo_hs) # Get initial guess of boundary
        else
            boundary = randomstate(boundary, pepo_hs)
        end
    end

    boundary = calculate!(boundary)

    D = codomain(pepo[1, 1])[1]

    updatestep!(pepo, coeffs, op_tuple, boundary, truncspace(D); kwargs...)

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

# Update each site simultaneously, then truncate each bond simultaneously. Can be used with
# a trivial unit cell, however requires a GULT metric. The optimisation step can be perfomed
# on the isometries only.
# L is a lattice of MPO's
function siteupdate!(P, L, boundary::AbstractEnvironment; gauge=2, ξ=Inf)
    PE = evolve(P, L)

    G = metric(PE, boundary) # OnLattice

    left0, right0 = isometryinit(G)
    U, S, V = metricoptimise(left0, right0, G; EAT=false)

    # Convert isometries to the weighted trace gauge if desired.
    wtg!(U, S, V, G; gauge=gauge)
    if !(ξ == Inf)
        rescale!(S, ξ)
        wtg!(U, S, V, G; gauge=gauge)
    end

    U = @. U * sqrt(S)
    V = sqrt.(circshift(S, (0, 1))) .* V

    isometryupdate!.(P, PE, U, V)

    return P
end

# Update and truncate each bond separately, and in sequence. Not compatible with a trivial 
# unit cell. Requires a LULT metric. The optimisation step can be performed on isometries,
# or the tensors directly.
function bondupdate!(P, L, boundary::AbstractEnvironment; kwargs...)
    # We need to devectorise everything as bond update is sequential.
    for y in axes(P, 2)
        for x in axes(P, 1)
            bondupdate!(P[x, y], P[x + 1, y], L[x, y], boundary, trunc, x, y; kwargs...)
        end
        # If we choose to enforce translational symmetry, then don't bother updating 
        # mirror bonds.
        if latticestyle(P).issymetric == true
            for x in 1:(Nx - 1)
                P[x, y + 1], P[x + 1, y + 1] = P[x + 1, y], P[x + 2, y]
            end
            break
        end
    end

    return P
end

function bondupdate!(
    pepo_left, pepo_right, op_tuple, boundary, trunc::TruncationScheme, x, y; ξ=Inf, gauge=2
)
    # Get the operators that will apply to the left and right sites
    op_left = op_tuple[1]
    op_right = op_tuple[2]

    # Evolve the tensors
    q_left, l = evolveleft(pepo_left, op_left)
    r, q_right = evolveright(pepo_right, op_right)

    # If doing tensor optimisation, then decompose into QL, and QR

    # If doing isometry optimisation, then get initial guesses for L & R

    G = metric(q_left, q_right, boundary, x, y)

    opt.maxiter
    opt.tol
    opt.EAT
    u, s, v = metricoptimise(l, r, G, trunc; maxiter=opt.maxiter, tol=opt.tol, EAT=opt.EAT)

    u, s, v = wtg(u, s, v, G; kwargs...)

    isometryupdateleft!(pepo_left, q_left, sqrt(s) * u)
    isometryupdateright!(pepo_right, q_right, v * sqrt(s))

    return pepo_left, pepo_right
end

abstract type AbstractOptimisation end

# Optimise just the isometries.
struct IsometryOptimisation <: AbstractOptimisation
    maxiter::Int
    tol::Float64
end

# Optimise the reduced tensors as isometries. 
struct TensorOptimisation <: AbstractOptimisation
    maxiter::Int
    tol::Float64
    EAT::Bool
end

function sequentialoptimise(alg::AbstractOptimisation, a, b, boundary, x, y)
    return alg(a, b, boundary, x, y)
end
function paralleloptimise(PE, boundary, alg::IsometryOptimisation)
    return alg()
end

function (alg::IsometryOptimisation)(args...)
    G = metric(args)
    left0, right0 = isometryinit(G)
    U, S, V = metricoptimise(left0, right0, G; EAT=false)
    return U, S, V
end

# Only works in sequence
function (alg::TensorOptimisation)(A, B, boundary, x, y)
    QL, left0 = leftrankreduce(A)
    right0, QR = rightrankreduce(B)
    G = metric(QL, QR, boundary, x, y)
    U, S, V = metricoptimise(left0, right0, G; alg.kwargs...)
    return U, S, V
end

# struct HilbertSchmidt{TType<:AbstractTensorMap}
#     P::TType
#     Q::TType
# end
#
# struct Trace{TType<:AbstractTensorMap}
#     T::TType
# end

# function metric(P::PEPO, boundary::AbstractBoundary) end
# function metric(P1, P2, boundary::AbstractBoundary) end

# Returns an initalised boundary. Use `calculate!` to calculate the boundary.
# Returns an environment state 
function initenvironment(boundarytype::AbstractEnvironment, P) end
