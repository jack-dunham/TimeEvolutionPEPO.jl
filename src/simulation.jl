struct Simulation{
    E<:AbstractTimeEvolution,P<:AbstractPEPO,PB<:PEPOWithBuffer,L<:AbstractTimeStepper,T
}
    initialstate::P
    info::SimulationInfo
    stepper::Schedule{L}
    method::E
    state::PB
    truncators::Tuple{T,T}
end

function Base.show(io::IO, m::MIME"text/plain", sim::Simulation)
    # pepo = getpepo(sim)
    # map(b -> bondspace(pepo ,b ), uniquebonds(pepo)) |> display
    println("METHOD")
    println("≡≡≡≡≡≡")
    show(io, m, sim.method)
    println("")
    println("INFO")
    print("≡≡≡≡")
    for name in fieldnames(SimulationInfo)
        println("")
        print(name, ": ", getfield(sim.info, name))
    end
    return nothing
end

"""
    getpepo(obj::Union{AbstractPEPO, Simulation}) -> PEPO

Return the unwrapped `PEPO` object of `obj`. If `obj isa PEPO` then `getpepo(obj) === obj`.
"""
pepo(sim::Simulation) = sim.state.parent

"""
    latticetensors(obj::Union{AbstractPEPO, DensityMatrix, Simulation}) -> AbstractLattice

Return the `AbstractLattice` of underyling pepo tensors associated with `obj`.
"""
tensors(sim::Simulation) = tensors(pepo(sim))

unitcellsize(x::AbstractMatrix) = size(x)
unitcellsize(_) = (1, 1)

similar_matrix(A::AbstractMatrix, outsize) = Matrix{eltype(A)}(undef, outsize)
similar_matrix(A, outsize) = Matrix{typeof(A)}(undef, outsize)

function promotelattice(model::NTuple, args...; geometry::G=Square()) where {G}
    shape = (x, i...) -> isa(x, AbstractMatrix) ? size(x, i...) : size([x;;], i...)

    xs = map(x -> shape(x, 1), (model..., args...))
    ys = map(x -> shape(x, 2), (model..., args...))
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

        return Lattice{G}(mat)
    end

    return outlat
end

geometrytype(::Type{G}) where {G<:ITC.AbstractUnitCellGeometry} = G
geometrytype(::Type{<:AbstractLattice{G}}) where {G} = G
geometrytype(::Type{<:AbstractMatrix}) = Square
geometrytype(T::DataType) = Square
geometrytype(g) = geometrytype(typeof(g))

promote_geometry_type(A::Type{G}, B::Type{G}) where {G<:ITC.AbstractUnitCellGeometry} = G
promote_geometry_type(args...) = mapreduce(geometrytype, promote_geometry_type, args)
function promote_geometry(A::Type, B::Type)
    throw(ErrorException("TODO: incompatible lattice geometries"))
end

copyspaceto!(mat, arg::AbstractMatrix) = copyto!(mat, arg)
copyspaceto!(mat, arg::TensorSpace) = copyto!(mat, [arg;;])

model2matrix(m::AbstractVector) = map(model2matrix, m)
model2matrix(m::AbstractMatrix) = m
model2matrix(m::AbstractMatrix{<:Number}) = [m;;]
model2matrix(m) = [m;;]

function model2tuple(m::NTuple{N}) where {N}
    println(typeof(m))
    if N in (1, 2)
        return map(model2matrix, m)
    else
        throw(ArgumentError("model tuple as too many elements for 2D simulation; got $N"))
    end
end
model2tuple(m) = model2tuple((m,))

function Simulation(
    model,
    rho0::Union{NTuple{3},Matrix,ThermalState},
    horizontal::Union{S,AbstractMatrix{S}},
    vertical::Union{S,AbstractMatrix{S}}=horizontal;
    method::AbstractTimeEvolution,
    physical=S,
) where {S}
    model = model2tuple(model)

    # physical, horizontal, vertical = promotelattice(model, physical, horizontal, vertical)

    G = promote_geometry_type(physical, horizontal, vertical)

    println(typeof(physical))

    stepper, physicalspace = newstepper(Lattice{G}([physical;;]), model, method)

    pepo = PEPO([rho0;;], [physicalspace;;], [horizontal;;], [vertical;;])

    # Construct the time evolving state
    state = newevolution(pepo, stepper, method)

    # Construct the truncators
    truncators = newtruncators(state, stepper, method.truncalg)

    initialstate = deepcopy(state)

    sim = Simulation(initialstate, SimulationInfo(), stepper, method, state, truncators)

    return sim
end
function Simulation(model, pepo::PEPO, method::AbstractTimeEvolution)
    model = model2tuple(model)

    stepper = newstepper(pepo, model, method)

    # Construct the time evolving state
    state = newevolution(pepo, stepper, method)

    # Construct the truncators
    truncators = newtruncators(state, stepper, method.truncalg)

    initialstate = deepcopy(state)

    sim = Simulation(initialstate, SimulationInfo(), stepper, method, state, truncators)

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

function simulate!(callback, sim::Simulation; maxshots = 100)

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

    r = range(start = 1, step = round(Int, numsteps/maxshots, RoundUp), stop = numsteps)
    shotsat = collect(r)

    numiter = 0
    tol = Inf

    # Need to copy the weights otherwise they will mutate
    prev_weights = deepcopy(weights(pepo))

    info.iterations == 0 && callback(sim)
    
    timestep!(pepo, truncator, only(stepper.start), truncalg)

    doshot = false

    while numiter < numsteps && tol ≥ breaktol

        numiter += 1

        doshot = numiter in shotsat

        itr = doshot ? stepper.finish : stepper.bulk

        for stepper in itr
            timestep!(pepo, truncator, stepper, truncalg)
        end

        # Calculate the difference in pepo singular values between timesteps
        tol = tolerance!(prev_weights, weights(pepo), timestep)
        # tol = 0.0

        method.verbose && @info "Steady state convergence: $tol after $numiter iterations."

        info.convergence = tol
        info.iterations += 1

        # User specified function for post-step processing
        info.convergence < breaktol ? info.isconverged = true : info.isconverged = false

        if doshot
            callback(sim)
            timestep!(pepo, truncator, only(stepper.start), truncalg)
            doshot = false
        end
    end

    for stepper in stepper.finish
        timestep!(pepo, truncator, stepper, truncalg)
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

    rmul!.(PEPOKit.tensors(sim.state), 0.5)

    simulate!(measurehook!, sim)

    return sim.output
end
function measurehook!(sim::Simulation, A=PAULI[1]; env=nothing)
    pepo = PEPOKit.pepo(sim)

    alg = VUMPS(; tol=1e-10, maxiter=200, bonddim=16, verbose=true)
    # alg = CTMRG(; tol=1e-8, maxiter=200, bonddim=16, verbose=true)
    #
    # env = newcontraction(alg, contractphysical(PEPOKit.tensors(pepo)))
    # renormalize!(env, contractphysical(PEPOKit.tensors(pepo)))

    dm = DensityMatrix(pepo, alg)
    # dm = DensityMatrix(pepo, env)

    rhos = map(ind -> ReducedDensityMatrix(dm, ind), uniqueindices(pepo))

    expt = map(x -> expval(x, A), rhos)
    return expt

    # corr = [correlator(dm, A, (1, 1), A, (i, 1)) for i in 2:20]
    # return corr
end

function measurehook!(output::Vector, sim, args...)

    # init
    tensors = PEPOKit.tensors(sim.state.parent)

    alg = CTMRG(; tol=1e-8, maxiter=200, bonddim=16, verbose=true)
    # alg = VUMPS(; tol=1e-10, maxiter=200, bonddim=16, verbose=true)
    # alg = CTMRG(; tol=1e-10, maxiter=100, bonddim=16, verbose=true)

    env = newcontraction(alg, contractphysical(tensors))
    runcontraction!(env)

    r = round(Int, sim.method.numsteps / 100, RoundUp)

    println(r)

    f = sim -> begin
        if mod(sim.info.iterations, 0:(r - 1)) == 0
            exp1 = measurehook!(sim, args...; env=env)
            push!(output, exp1)
        end
        return nothing
    end

    return f
end

function errorhook!(output)
    f = sim -> begin
        push!(output, sim.info.convergence)
    end
    return f
end
function negativityhook!(sim::Simulation)
    # if sim.info.iterations < 1
    #     return sim
    # end

    pepo = tensors(sim)

    alg = CTMRG(; maxiter=100, bonddim=16, verbose=true)

    dm = DensityMatrix(pepo, alg)
    rdm1 = PEPOKit.ReducedDensityMatrix(dm, (1, 1), (2, 1))
    rdm2 = PEPOKit.ReducedDensityMatrix(dm, (1, 1), (1, 2))

    return (negativity(rdm1), negativity(rdm2))
end

function negativityhook!(output, sim::Simulation)

    return sim -> push!(output, negativityhook!(sim))
end

function negativityhook!(output::AbstractVector)
    return sim -> push!(output, negativityhook!(sim))
end


function test_p2s!(out, op, mpo, us, vw, vn)
    @tensoropt (
        k => 2,
        b => 2,
        kk => 2,
        bb => 2,
        e1 => 4,
        w1 =>4,
        s1 => 4,
        n1 =>4,
        w =>4,
        s => 4,
        n =>4,
        e2 => D,
        w2 =>D,
        s2 => D,
        n2 =>D,
    ) out[k b; e1 e2 s w n ] = op[kk bb; e1 s1 w1 n1] * 
    mpo[k b kk bb; e2 s2 w2 n2] * us[s1 s2; s] * vw[w; w1 w2] * vn[n; n1 n2]
    return out
end
function test_p2s(op, mpo, us, vw, vn)
    @tensoropt (
        k => 2,
        b => 2,
        kk => 2,
        bb => 2,
        e1 => 4,
        w1 =>4,
        s1 => 4,
        n1 =>4,
        w =>4,
        s => 4,
        n =>4,
        e2 => D,
        w2 =>D,
        s2 => D,
        n2 =>D,
    ) out[k, b, e1, e2, s,  w,  n ] := op[kk, bb, e1, s1, w1, n1] * 
    mpo[k, b, kk, bb, e2, s2, w2, n2] * us[s1, s2, s] * vw[w, w1, w2] * vn[n, n1, n2]
    return out
end
