abstract type AbstractTimeEvolution end

Base.show(io::IO, te::AbstractTimeEvolution) = dump(te)

abstract type AbstractUpdateStyle end

struct LocalUpdate <: AbstractUpdateStyle end
struct GlobalUpdate <: AbstractUpdateStyle end

struct Simulation{E<:AbstractTimeEvolution,P<:AbstractPEPO,L<:AbstractTimeStepper,T}
    initialstate::Tuple{P,P}
    info::ConvergenceInfo
    stepper::Tuple{L,L}
    method::E
    state::Tuple{P,P}
    truncators::Tuple{T,T}
end

function Base.show(io::IO, m::MIME"text/plain", sim::Simulation)
    println("METHOD")
    println("≡≡≡≡≡≡")
    show(io, m, sim.method)
    println("INFO")
    print("≡≡≡≡")
    for name in fieldnames(ConvergenceInfo)
        println("")
        print(name, ": ", getfield(sim.info, name))
    end
    return nothing
end
getstate(sim::Simulation) = sim.state[1]

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

showheader(::Type{<:DMPO}) = "Dynamical Matrix Product Operator (DMPO)"


function Base.show(io::IO, m::MIME"text/plain", sim::AbstractTimeEvolution)
    shownested(io, m, sim)
    return nothing
end

@kwdef struct TEBD{
    TAlg<:AbstractTruncationScheme,S<:AbstractUpdateStyle,C<:TensorKit.TruncationScheme
} <: AbstractTimeEvolution
    timestep::Float64 = 0.01
    numsteps::Int = 100
    breaktol::Float64 = 0.0
    trotterorder::Int = 1
    truncalg::TAlg = SU()
    verbose::Bool = true
    updatestyle::S = LocalUpdate()
    compress::C = truncerr(eps())
end

showheader(::Type{<:TEBD}) = "Time-evolving block decimation (TEBD)"

# Explicitly isotropic
function dynamicalmap(
    method::AbstractTimeEvolution,
    model::AbstractMatrix{<:LindbladEquation},
    physical::AbstractMatrix{<:IndexSpace},
)
    stepper = timestepper(method, model, physical)
    return stepper, stepper
end

# Anisotropic
function dynamicalmap(
    method::AbstractTimeEvolution,
    model::NTuple{2,AbstractMatrix{<:LindbladEquation}},
    physical::AbstractMatrix{<:IndexSpace},
)
    horizontal = timestepper(method, model[1], physical)
    vertical = timestepper(method, model[2], physical)

    return horizontal, vertical
end

function timestepper(method::DMPO, model::AbstractUnitCell, physical::AbstractUnitCell)
    tau = method.timestep / method.trotterorder

    # Calculate the time-evolution mpo for each unique mpo
    tempo = broadcast(x -> dynamicalmpo(x, tau, method.order), model)

    t = broadcast(eachindex(physical), physical) do ind, p
        return dmpotensor(tempo[ind[1], ind[2]], p)
    end

    return GlobalStepper(t)
end

function timestepper(method::TEBD, model::AbstractUnitCell, physical::AbstractUnitCell)
    tau = method.timestep / method.trotterorder
    compress = method.compress

    u1, v2 = tebdtensor(model[1, 1], physical[1, 1], physical[2, 1], tau; compress=compress)

    u = similar(physical, typeof(u1))
    v = similar(physical, typeof(v2))

    u[1, 1] = u1
    v[2, 1] = v2

    skipfirst = true

    for ind in eachindex(physical)
        if skipfirst
            skipfirst = false
            continue
        end
        x = ind[1]
        y = ind[2]
        u1, v2 = tebdtensor(
            model[x, y], physical[x, y], physical[x + 1, y], tau; compress=compress
        )
        u[x, y] = u1
        v[x + 1, y] = v2
    end

    return LocalStepper(u, v)
end

vectorise(term::AbstractTerm) = vectorise(typeof(term))(Matrix{ComplexF64}(term))

vectorise(::Type{HamiltonianTerm{1}}) = twolocal(vectorise(HamiltonianTerm{2}))

# Vectorise the superoperators; we do this the non-julian way (row major) as the rest
# of the code assumes (ket, bra) ordering (whoops).
function vectorise(::Type{HamiltonianTerm{2}})
    return H -> -im * (kron(H, one(H)) - kron(one(H), transpose(H)))
end
function vectorise(::Type{LindbladTerm})
    anticomm = L -> kron(L' * L, one(L)) + kron(one(L), transpose(L' * L))

    return twolocal(L -> kron(L, transpose(L')) - 1 / 2 * anticomm(L))
end

# We need both local terms that have support on two sites, and therefore need to divide
# by two to avoid double counting in the brickwork pattern
twolocal(f) = H -> 1 / 2 * (f(kron(one(H), H)) + f(kron(H, one(H))))

function tebdtensor(model::LindbladEquation, p1::IndexSpace, p2::IndexSpace, dt; compress)

    # c2 = H -> (-im * (H ⊗ one(H) - one(H) ⊗ transpose(H)))
    # d2 = L -> L ⊗ transpose(L') - 1 / 2 * (L' * L ⊗ one(L) + one(L) * transpose(L' * L))
    #
    # # We need both local terms that have support on two sites, and therefore need to divide
    # # by two to avoid double counting in the brickwork pattern
    # c1 = H -> 1 / 2 * (c2(one(H) ⊗ H) + c2(H ⊗ one(H)))
    # d1 = L -> 1 / 2 * (d2(one(L) ⊗ L) + d2(L ⊗ one(L)))

    space = p1 * p2 * p1' * p2'

    data = zeros(ComplexF64, dim(space), dim(space))

    # We divide by half again for local terms as in 2D we effectively apply these twice
    data = 1 / 2 * sum(vectorise, model.onelocal; init=data)
    data = 1 / 2 * sum(vectorise, model.linops; init=data)

    data = sum(vectorise, model.twolocal; init=data)

    tensor = TensorMap(exp(dt * data), space, space)

    # move site one to codomain and site two to domain
    tensor = permute(tensor, (1, 3, 5, 7), (2, 4, 6, 8))

    u, s, v = tsvd(tensor; trunc=compress)

    return u * sqrt(s), _transpose(sqrt(s) * v)
end

function dmpotensor(mpo, p::S) where {S<:IndexSpace}
    d = dim(p)
    D = sum(x -> size(x, 1), mpo[:, 1])
    s = ⊕(oneunit(S)^D...)
    array = zeros(ComplexF64, d, d, d, d, D, D)

    # for (slice, block) in zip(eachslice(array; dims=(5, 6)), mpo)
    #     println(size(slice))
    #     println(size(block[1,1]))
    #     slice .= reshape(block, size(slice))
    # end

    for (ind1, block) in pairs(mpo)
        i = ind1[1]
        j = ind1[2]
        for (ind2, op) in pairs(block)
            n = i + ind2[1] - 1
            m = j + ind2[2] - 1
            slice = view(array, :, :, :, :, n, m)
            slice .= reshape(op, size(slice))
        end
    end

    tensor = TensorMap(array, p * p' * p' * p, s * s')
    return tensor
end

unitcellsize(x::AbstractMatrix) = size(x)
unitcellsize(_) = (1, 1)

dimtospace(d::Int) = ℂ^d
dimtospace(s::IndexSpace) = s

similar_matrix(A::AbstractMatrix, outsize) = Matrix{eltype(A)}(undef, outsize)
similar_matrix(A, outsize) = Matrix{typeof(A)}(undef, outsize)

function promotespaces(
    model, args...; geometry::G=Square(), issymmetric::Bool=false
) where {G}
    shape = (x, i...) -> isa(x, AbstractMatrix) ? size(x, i...) : size([x;;], i...)

    xs = map(x -> shape(x, 1), (model, args...))
    ys = map(x -> shape(x, 2), (model, args...))
    outsize = (lcm(xs...), lcm(ys...))

    outmat = map(x -> similar_matrix(x, outsize), args)

    outlat = map(outmat, args) do mat, arg
        sx, sy = shape(arg)
        nx, ny = size(mat)

        for y in 1:sy:ny
            for x in 1:sx:nx
                copyspaceto!(view(mat, x:(x + sx - 1), y:(y + sy - 1)), arg)
            end
        end

        return Lattice{G}(mat, issymmetric)
    end

    model = isa(model, AbstractMatrix) ? model : [model;;]

    lattice = Lattice{Square}(model, issymmetric)

    return lattice, outlat...
end

copyspaceto!(mat, arg::AbstractMatrix) = copyto!(mat, arg)
copyspaceto!(mat, arg::TensorSpace) = copyto!(mat, [arg;;])

function Simulation(
    model::Union{LindbladEquation,AbstractMatrix{LindbladEquation}},
    rho0,
    horizontal::Union{S,AbstractMatrix{S}},
    vertical::Union{S,AbstractMatrix{S}}=horizontal;
    method::AbstractTimeEvolution,
    physical=physicalspace(S, model),
) where {S}
    model, physical, horizontal, vertical = promotespaces(
        model, physical, horizontal, vertical
    )

    space = dimtospace.(physical)

    pepo = PEPO(rho0, space, dimtospace.(horizontal), dimtospace.(vertical))

    map = dynamicalmap(method, model, space)

    # Construct the time evolving state
    state = initevolution(pepo, map, method)

    # Construct the truncators
    truncators = inittruncators(state, map, method.truncalg)

    initialstate = deepcopy(state)

    sim = Simulation(initialstate, ConvergenceInfo(), map, method, state, truncators)

    return sim
end

# function Simulation(
#     model,
#     initial_state,
#     space,
#     eastbond,
#     southbond=eastbond;
#     alg::AbstractTimeEvolution,
#     output=nothing,
# )
#     pepo = PEPO(initial_state, space, eastbond, southbond)
#
#     return Simulation(pepo, deepcopy(pepo), model, alg, output, ConvergenceInfo())
# end
#
# function Base.show(io::IO, sim::Simulation)
#     return show(sim.method)
#     # print("timestep: "); dump(tebd.timestep);
#     # print("numsteps: "); dump(tebd.numsteps);
#     # print("breaktol: "); dump(tebd.breaktol);
#     # print("order: "); dump(tebd.order);
#     # print(" adasd"); dump(tebd.truncalg);
# end

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

function simulate!(callback, sim::Simulation)
    method = sim.method
    pepo = sim.state
    stepper = sim.stepper
    truncator = sim.truncators

    info = sim.info

    truncalg = method.truncalg
    numsteps = method.numsteps
    timestep = method.timestep
    breaktol = method.breaktol
    order = method.trotterorder

    # space = physicalspace(pepo)::AbstractLattice
    #
    # # Get the dynamical map from the model
    # dynamical_map = dynamicalmap(model, space, evoalg)
    #
    # # Make PEPO time-dependant so we can evolve it.
    # td_pepo = initevolution(pepo, dynamical_map, evoalg)
    #
    # # Initialise the truncators
    # truncators = inittruncators(td_pepo, dynamical_map, truncalg)

    # Calculate the environments from the initial guess

    # if isa(truncalg.updatescheme, LULT)
    #     renormalize!(truncators)
    # else
    #     nothing
    # end

    numiter = 0
    tol = Inf

    # Need to copy the weights otherwise they will mutate
    prev_weights = deepcopy(getweights(pepo[1]))

    while numiter < numsteps && tol ≥ breaktol
        timestep!(pepo, truncator, stepper, order; truncalg=truncalg)

        # If we are doing full renorm at each step, we don't need to recalc
        # if isa(alg, SU) #|| alg.renorm === :full
        #     nothing
        # else
        #     calculate!(envs)
        # end

        # renormalize!(envs)

        # Calculate the difference in pepo singular values between timesteps
        tol = tolerance!(prev_weights, getweights(pepo[1]), timestep)

        numiter += 1

        method.verbose && @info "Steady state convergence: $tol after $numiter iterations."

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

    pepo = getstate(sim).parent

    println(typeof(pepo))

    alg = VUMPS(; maxiter=100, bonddim=10, verbose=true)

    obs_env = newcontraction(alg, trace(pepo).data)

    run!(obs_env)

    one_local = onelocal(pepo, obs_env; normalize=true)

    magn = measure(one_local, σ1)

    return magn.data[1, 1]
end

measurehook!(output::AbstractVector) = sim -> push!(output, measurehook!(sim))

errorhook!(output::AbstractVector) = sim -> push!(output, sim.info.error)

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
