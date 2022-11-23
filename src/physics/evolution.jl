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

    Op_x = @. exp(τ * Lx)
    Op_y = @. exp(τ * Ly)

    Op = SuperOperator(Op_x, Op_y)

    propagate!(P, Op, n, alg)

    return P
end

function propagate!(P, Op, n, alg::FET{VUMPS})
    # Current VUMPS implementation using MPOs rather than trotter gates. 
    Op = to_mpo(Op)

    _propagate!(P, Op, n, alg)
    return P
end

function _propagate!(P, Op, n, alg::FET{Alg}) where {Alg}
    # Unroll algorithm parameters.
    recycle = alg.recycle   # Whether to recycle the metric
    gauge = alg.gauge       # Whether to fix the gauge of the metric
    ξ = alg.ξ               # Correlation length scaling parameter
    param = alg.boundary    # Parameters of the boundary algorithm

    # First time step.
    Bx_i = By_i = x -> initialise_state(Alg, x, param)
    # Apply the time evolution operator, calculate the metric, and then truncate.
    P, Bx, By = trotterstep!(P, Op, Bx_i, By_i; ξ=ξ, gauge=gauge)

    # Remaining time steps.
    for _ in 1:(n - 1)
        # If we wish to recycle the metric, set the function that returns the initial guess
        # for the boundary to simply return the boundary of the previous time step.
        if recycle == true
            Bx_i = _ -> Bx
            By_i = _ -> By
        end
        # Apply the time evolution operator, calculate the metric, and then truncate.
        P, Bx, By = trotterstep!(P, Op, Bx_i, By_i; ξ=ξ, gauge=gauge)
    end

    # Return the PEPO state after n applications of the operator.
    return P
end

function trotterstep!(P, Op, Bx_i, By_i; kwargs...)
    # Apply the time evolution operator along x-axis. 
    P, Bx = half_trotterstep!(P, Op.x, Bx_i; kwargs...)

    # Permute the y- and x-axis of the PEPO.
    perm = (2, 1, 3, 5, 4, 6)
    P = _permute(P, perm)

    # Apply the time evolution operator along x-axis (now the y-axis.)
    P, By = half_trotterstep!(P, Op.y, Bx_i; kwargs...)

    # Undo the previous permutation.
    P = _permute(P, invperm(perm))

    return P, Bx, By
end

# Evolves the PEPO state P according to Op along a single axis 
function half_trotterstep!(P, Op, get_initialboundary; kwargs...)
    # Apply time evolution operator to the state.
    PE = evolve(P, Op)

    println(space.(Op))

    # Get the single bond space for each site.
    Dη = getindex.(domain.(PE), 1)

    # Calculate the hilbertschmidt tensor tr(ρρ'). 
    PE_hs = hilbertschmidt(PE)

    # Get the initial guess for the boundary.
    boundary = get_initialboundary(PE_hs)

    # Run the boundary algorithm to obtain the boundary.
    boundary = calculate!(boundary)

    # Form the truncation metric tensor from the boundary.
    Y = truncmetric(Dη, boundary)

    # Truncate the bond.
    P = truncate!(P, PE, Y; kwargs...)

    return P, boundary
end

# m[s_b s_ph' s_ph; s_b s_ph' s_ph] (l,b,k ; r b k)
function _evolve(
    p::AbstractTensorMap{S,3,3}, m::AbstractTensorMap{S,3,3}
) where {S<:IndexSpace}
    return @tensor pe[w w2 s k; e e2 n b] := m[w2 b k; e2 mb mk] * p[w s mk; e n mb]
end

function evolve(
    p::AbstractTensorMap{S,3,3}, m::AbstractTensorMap{S,3,3}
) where {S<:IndexSpace}
    t = _evolve(p, m)
    return merge_bonds(t)
end

evolve(P::AbstractPEPO, M) = evolve.(P, M)
