abstract type AbstractTimeEvolution end

Base.show(io::IO, te::AbstractTimeEvolution) = dump(te)

# TODO We only need one Model struct. Change this soon.
struct Simulation{T<:AbstractTimeEvolution,P<:AbstractPEPO,V,M<:AbstractModel}
    state::P
    initial_state::P
    model::M
    alg::T
    output::V
    info::ConvergenceInfo
end

# Sruct MPOWII
@kwdef struct WII{TAlg<:AbstractTruncationScheme} <: AbstractTimeEvolution
    timestep::Float64 = 0.01
    numsteps::Int = 100
    breaktol::Float64 = 0.0
    order::Int = 1
    truncalg::TAlg = SU()
    verbose::Bool = true
end

function initialize(
    model,
    initial_state,
    space,
    eastbond,
    southbond=eastbond;
    alg::AbstractTimeEvolution,
    output=nothing,
)
    pepo = PEPO(initial_state, space, eastbond, southbond)

    return Simulation(pepo, deepcopy(pepo), model, alg, output, ConvergenceInfo())
end

function Base.show(io::IO, sim::Simulation)
    return show(sim.alg)
    # print("timestep: "); dump(tebd.timestep);
    # print("numsteps: "); dump(tebd.numsteps);
    # print("breaktol: "); dump(tebd.breaktol);
    # print("order: "); dump(tebd.order);
    # print(" adasd"); dump(tebd.truncalg);
end

const LocalTimeDependantPEPO{P<:LocalUpdatablePEPO} = TimeDependantPEPO{P}
const GlobalTimeDependantPEPO{P<:GlobalUpdatablePEPO} = TimeDependantPEPO{P}

function initenvironment(network::Trace, alg::AbstractBoundaryAlgorithm; kwargs...)
    return ITC.initialize(alg, network.data; kwargs...)
end

"""
    simulate(sim::Simulation)
    simulate(hook::Function, sim::Simulation)

Equivalent to `simulate!(hook, deepcopy(sim))`. 
"""
simulate(sim::Simulation) = simulate(identity, sim)
simulate(hook, sim::Simulation) = simulate!(hook, deepcopy(sim))

simulate!(sim::Simulation) = simulate!(identity, sim)
function simulate!(hook, sim::Simulation)
    # @info "Running simulation of model: $tebd.model"

    tebd = sim.alg

    simulate!(hook, sim, Val(tebd.order))

    return sim
end

function simulate!(callback, sim, order::Val{N}) where {N}
    evoalg = sim.alg
    pepo = sim.state
    model = sim.model

    info = sim.info

    truncalg = evoalg.truncalg
    numsteps = evoalg.numsteps
    timestep = evoalg.timestep
    breaktol = evoalg.breaktol

    space = physicalspace(pepo)::AbstractLattice

    # Get the dynamical map from the model
    dynamical_map = dynamicalmap(model, space, evoalg)

    # Make PEPO time-dependant so we can evolve it.
    td_pepo = initialize_evolution(pepo, dynamical_map, evoalg)

    # Initialise the truncators
    truncators = initialize_truncators(td_pepo, dynamical_map, truncalg)

    # Calculate the environments from the initial guess

    # if isa(truncalg.updatescheme, LULT)
    #     renormalize!(truncators)
    # else
    #     nothing
    # end

    numiter = 0
    tol = Inf

    # Need to copy the weights otherwise they will mutate
    prev_weights = deepcopy(getweights(pepo))

    while numiter < numsteps && tol ≥ breaktol
        timestep!(td_pepo, truncators, dynamical_map, order; truncalg=truncalg)

        # If we are doing full renorm at each step, we don't need to recalc
        # if isa(alg, SU) #|| alg.renorm === :full
        #     nothing
        # else
        #     calculate!(envs)
        # end

        # renormalize!(envs)

        # Calculate the difference in pepo singular values between timesteps
        tol = tolerance!(prev_weights, getweights(pepo), timestep)

        numiter += 1

        evoalg.verbose && @info "Steady state convergence: $tol after $numiter iterations."

        info.error = tol
        info.iterations = numiter

        # User specified function for post-step processing
        callback(sim)
    end
    return sim
end

function testthermal(D)
    s2 = ℂ^2
    Ds = ℂ^D

    Z = Matrix{ComplexF64}([1 0; 0 -1])
    ZZ = kron(Z, Z)

    βc = log(1 + sqrt(2)) / 2

    data = -1 / 2 * βc * ((kron(ZZ, one(ZZ)) + kron(one(ZZ), ZZ)))

    model = Model(data, data)

    talg = SU()
    # talg = FET(; alg=VUMPS(; bonddim=4))

    sim = initialize(
        model,
        [0, 0, 0],
        fill(s2, 2, 2),
        fill(Ds, 2, 2);
        output=[],
        alg=TEBD(; timestep=0.02, numsteps=100, truncalg=talg),
    )

    rmul!.(getstate(sim.state), 0.5)

    simulate!(measurehook!, sim)

    return sim.output
end
function measurehook!(sim::Simulation)
    # if sim.info.iterations < 1
    #     return sim
    # end

    pepo = sim.state

    alg = VUMPS(; maxiter=100, bonddim=10, verbose=true)

    obs_env = newcontraction(alg, trace(pepo).data)

    run!(obs_env)

    one_local = onelocal(pepo, obs_env; normalize=true)

    magn = measure(one_local, σ1)

    push!(sim.output, magn.data[1, 1])

    return sim
end
function errorhook!(sim::Simulation)
    push!(sim.output, sim.info.error)

    return sim
end

function testising()
    s2 = ℂ^2
    Ds = ℂ^2

    model = PEPOKit.Models.ising(; J=1.0, h=0.2)

    # talg = SU()
    talg = FET(; alg=VUMPS(; bonddim=4))

    sim = initialize(
        model,
        [0, 0, 1],
        fill(s2, 2, 2),
        fill(Ds, 2, 2);
        output=[],
        alg=TEBD(; timestep=0.2, numsteps=200, truncalg=talg),
    )

    simulate!(errorhook!, sim)

    return sim
end

function testxyz()
    s2 = ℂ^2

    # L = Lattice{1,1,Infinite}([s2;;])
    L = Lattice{2,2,Infinite}([s2 s2; s2 s2], false)
    # L = Lattice{3,3,Infinite}([s2 s2 s2; s2 s2 s2; s2 s2 s2])

    pepo = PEPO(L, [1, 0, 0], ℂ^2)

    Z = Matrix{ComplexF64}([1 0; 0 -1])

    Jx = 0.5
    Jy = 1.2
    Jz = 1.0
    Γ = 0.25

    mx, my = @infliouvillian Jx * X[i] * X[i +1] +
        Jy * Y[i] * Y[i + 1] +
        Jz * Z[i] * Z[i + 1] Γ * M[i]

    out = trotterevolution!(pepo, fill(mx, L), fill(my, L))

    return out
end
