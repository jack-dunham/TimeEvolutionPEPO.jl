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

@kwdef struct DMPO{TAlg<:AbstractTruncationScheme} <: AbstractTimeEvolution
    timestep::Float64 = 0.01
    numsteps::Int = 100
    breaktol::Float64 = 0.0
    trotterorder::Int = 1
    truncalg::TAlg = SU()
    extend::Bool = true
    compress::Bool = true
    approx::Bool = false
    order::Int = 1
    verbose::Bool = true
end

function tensorify(method::DMPO, model::AbstractUnitCell, physical::AbstractUnitCell)
    # Calculate the time-evolution mpo for each unique mpo
    tempo = map(x -> TimeEvolutionMPO(x, method.timestep), model)

    rv = broadcast(eachindex(physical), physical) do ind, p
        return dmpo2tensor(tempo[ind[1], ind[2]], p)
    end

    return rv
end

function tensorify(method::TEBD, model::AbstractUnitCell, physical::AbstractUnitCell)
    # Calculate the time-evolution mpo for each unique mpo
    tempo = map(x -> TimeEvolutionMPO(x, method.timestep), model)

    u1, v2 = tebd2tensor(tempo[1, 1], physical[1, 1], physical[2, 1])

    u = similar(physical, typeof(u1))
    v = similar(physical, typeof(v2))

    skipfirst = true

    foreach(eachindex(physical)) do ind
        if skipfirst
            skipfirst = false
            continue
        end
        x = ind[1]
        y = ind[2]
        u1, v2 = tebd2tensor(tempo[x, y], physical[x, y], physical[x + 1, y])
        u[x, y] = u1
        v[x + 1, y] = v2
    end

    return u, v
end

function tebd2tensor(model::LindbladEquation, p1::IndexSpace, p2::IndexSpace; trunc)
    ⊗ = kron

    # Vectorise the superoperators; we do this the non-julian way (row major) as the rest
    # of the code assumes (ket, bra) ordering (whoops).
    c2 = H -> (-im * (H ⊗ one(H) - one(H) ⊗ transpose(H)))
    d2 = L -> L ⊗ transpose(L') - 1 / 2 * (L' * L ⊗ one(L) + one(L) * transpose(L' * L))

    # We need both local terms that have support on two sites, and therefore need to divide
    # by two to avoid double counting in the brickwork pattern
    c1 = H -> 1 / 2 * (c2(one(H) ⊗ H) + c2(H ⊗ one(H)))
    d1 = L -> 1 / 2 * (d2(one(L) ⊗ L) + d2(L ⊗ one(L)))

    data = Matrix(ComplexF64, p^4, p^4)

    # We divide by half again for local terms as in 2D we effectively apply these twice
    data += 1 / 2 * sum(c1, model.onelocal)
    data += 1 / 2 * sum(d1, model.linops)

    data += sum(c2, model.twolocal)

    space = p1 * p2 * p1' * p2'

    tensor = TensorMap(exp(dt * data), space, space)

    # move site one to codomain and site two to domain
    tensor = permute(tensor, (1, 3, 5, 7), (2, 4, 6, 8))

    u, s, v = tsvd(tensor; trunc=trunc)

    return u * sqrt(s), _transpose(sqrt(s) * v)
end

function dmpo2tensor(mpo, p::S) where {S<:IndexSpace}
    d = dim(p)
    D = prod(x -> size(x, 1), mpo)
    s = oneunit(S)^D
    array = zeros(ComplexF64, d, d, d, d, D, D)

    for (slice, block) in zip(eachslice(array; dims=(5, 6)), mpo)
        slice .= reshape(block, size(slice))
    end

    tensor = TensorMap(array, p * p' * p' * p, s * s')
    return tensor
end

unitcellsize(x::AbstractMatrix) = size(x)
unitcellsize(_) = (1, 1)

dimtospace(d::Int) = ℂ^d
dimtospace(s::IndexSpace) = s

similar_matrix(A::AbstractMatrix, outsize) = Matrix{eltype(A)}(undef, outsize)
similar_matrix(A, outsize) = Matrix{A}(undef, outsize)

function promote_matrix_size(args...)
    xs = map(x -> size(x, 1), args)
    ys = map(x -> size(x, 2), args)
    outsize = (lcm(xs...), lcm(ys...))

    outmat = map(x -> similar_matrix(x, outsize), args)

    foreach(outmat, args) do mat, arg
        sx, sy = isa(arg, AbstractMatrix) ? size(arg) : (1, 1)
        nx, ny = size(mat)

        for y in 1:sy:ny
            for x in 1:sx:nx
                mat[x:sx, y:sy] = arg
            end
        end
    end

    return outmat
end

function Simulation(
    model::LindbladEquation,
    eastdim::Int,
    southdim::Int=eastdim;
    physical,
    unitcell=(1, 1),
    kwargs...,
)
    eastspace = fill(ℂ^eastdim, unitcell)
    southspace = fill(ℂ^southdim, unitcell)

    d = trivial_physical_space(model)

    return Simulation(model, d, eastspace, southspace; kwargs...)
end

function Simulation(
    model::AbstractMatrix{LindbladEquation}, eastdim::Int, southdim::Int=eastdim; kwargs...
)
    eastspace = fill(ℂ^eastdim, unitcell)
    southspace = fill(ℂ^southdim, unitcell)

    d = physicaldim(model)

    physicalspace = ℂ^d

    return Simulation(model, d, eastspace, southspace; kwargs...)
end

# Assumes IndexSpace
function Simulation(
    model::Union{LindbladEquation,AbstractMatrix{LindbladEquation}},
    rho0,
    horizontal::Union{S,AbstractMatrix{S}},
    vertical::Union{S,AbstractMatrix{S}}=horizontal;
    physical=physicalspace(S, model),
    output=nothing,
    method,
) where {S}
    mod, phy, hor, ver = promote_matrix_size(model, physical, horizontal, vertical)

    pepo = PEPO(rho0, dimtospace.(phy), dimtospace.(hor), dimtospace.(ver))

    dynamicalmap = tensorify(method, model, phy)

    return Simulation(pepo, deepcopy(pepo), mod, dynamicalmap, output, ConvergenceInfo())
end

function Simulation(
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
