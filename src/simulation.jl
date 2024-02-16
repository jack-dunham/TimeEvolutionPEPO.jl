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
    println("")
    println("INFO")
    print("≡≡≡≡")
    for name in fieldnames(ConvergenceInfo)
        println("")
        print(name, ": ", getfield(sim.info, name))
    end
    return nothing
end

"""
    getpepo(obj::Union{AbstractPEPO, Simulation}) -> PEPO

Return the unwrapped `PEPO` object of `obj`. If `obj isa PEPO` then `getpepo(obj) === obj`.
"""
getpepo(sim::Simulation) = getpepo(sim.state[1])

"""
    latticetensors(obj::Union{AbstractPEPO, DensityMatrix, Simulation}) -> AbstractLattice

Return the `AbstractLattice` of underyling pepo tensors associated with `obj`.
"""
latticetensors(sim::Simulation) = latticetensors(getpepo(sim))

unitcellsize(x::AbstractMatrix) = size(x)
unitcellsize(_) = (1, 1)

similar_matrix(A::AbstractMatrix, outsize) = Matrix{eltype(A)}(undef, outsize)
similar_matrix(A, outsize) = Matrix{typeof(A)}(undef, outsize)

function promotelattice(
    model, args...; geometry::G=Square(), issymmetric::Bool=true
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

    return outlat
end

copyspaceto!(mat, arg::AbstractMatrix) = copyto!(mat, arg)
copyspaceto!(mat, arg::TensorSpace) = copyto!(mat, [arg;;])

function Simulation(
    model,
    rho0,
    horizontal::Union{S,AbstractMatrix{S}},
    vertical::Union{S,AbstractMatrix{S}}=horizontal;
    method::AbstractTimeEvolution,
    physical=physicalspace(S, model),
    issymmetric=true,
) where {S}
    physical, horizontal, vertical = promotelattice(
        model, physical, horizontal, vertical; issymmetric=issymmetric
    )

    pepo = PEPO(rho0, physical, horizontal, vertical)

    stepper = newstepper(pepo, model, method)

    # Construct the time evolving state
    state = newevolution(pepo, stepper, method)

    # Construct the truncators
    truncators = newtruncators(state, stepper, method.truncalg)

    initialstate = deepcopy(state)

    sim = Simulation(initialstate, ConvergenceInfo(), stepper, method, state, truncators)

    return sim
end

function initenvironment(network, alg::AbstractBoundaryAlgorithm; kwargs...)
    return ITC.initialize(alg, network; kwargs...)
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

    numiter = 0
    tol = Inf

    # Need to copy the weights otherwise they will mutate
    prev_weights = deepcopy(getweights(pepo[1]))

    while numiter < numsteps && tol ≥ breaktol
        timestep!(pepo, truncator, stepper, order; truncalg=truncalg)

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

    rmul!.(latticetensors(sim.state), 0.5)

    simulate!(measurehook!, sim)

    return sim.output
end
function measurehook!(sim::Simulation, A=PAULI[1])

    pepo = getpepo(sim)

    alg = VUMPS(; maxiter=200, bonddim=16, verbose=true)

    dm = DensityMatrix(pepo, alg)
     
    rhos = map(ind -> ReducedDensityMatrix(dm, ind), uniqueindices(pepo))

    expt = map(x -> expval(x, A), rhos)

    return expt[1], expt[2]
end

function measurehook!(output::Tuple, args...)
    f = sim -> begin 
        exp1, exp2 = measurehook!(sim, args...)
        push!(output[1], exp1)
        push!(output[2], exp2)
        return nothing
    end
end

errorhook!(output::AbstractVector) = sim -> push!(output, sim.info.error)

function negativityhook!(sim::Simulation)
    # if sim.info.iterations < 1
    #     return sim
    # end

    pepo = latticetensors(sim)

    alg = VUMPS(; maxiter=100, bonddim=10, verbose=true)

    obs_env = newcontraction(alg, trace(pepo).data)

    run!(obs_env)

    normalize!(pepo, obs_env.runtime)

    reduceddm = trace(latticetensors(pepo), obs_env, 1, 2)

    return negativity(reduceddm)
end
function negativityhook!(output::AbstractVector)
    return sim -> push!(output, negativityhook!(sim))
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
