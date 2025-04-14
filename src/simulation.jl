"""
$(TYPEDEF)

Concrete type representing mutable state of a simulation. Note, the fields of `Simulation`
are internal and not part of the public API.

# Constructors

    $(FUNCTIONNAME)(model::AbstractModel, rho, physical, virtual, [vertical_virtual=virtual]; method::AbstractMethod, timestep::Float64, kwargs...)
    $(FUNCTIONNAME)(model::AbstractModel, pepo::PEPO; method::AbstractMethod, timestep::Float64, kwargs...)

Initialise a `Simulation` of the model `model` with initial quantum state defined by `rho`,
as any valid input into the constructor `PEPO(rho, physical, virtual, virtual_vertical)`, or
given by a pre-existing `PEPO` object `pepo`. A `method` and `timestep` must be supplied as keyword 
arguments. An addition, the following optional keywords can also be used:

## Optional Keywords

- `order::Int=1`: order of the trotter decomposition (if any).
- `storeall::Bool=true`: this should be set to false if one does not wish to store objects
    that can be reconstructed; useful if, for example, very large tensors are used for the
    time evolution, but repeated calls to `simulate!` will be slower.

See also: [`simulate!`](@ref)
"""
struct Simulation{E<:AbstractTimeEvolution,P<:AbstractPEPO,L<:AbstractTimeStepper,M,T}
    info::SimulationInfo
    method::E
    model::M
    state::P
    stepper::TrotterDecomposition{L}
    truncators::Tuple{T,T}
    initialstate::P
    function Simulation(
        info::SimulationInfo,
        method::E,
        model::M,
        state::P,
        stepper::TrotterDecomposition{L},
        truncators::Tuple{T,T},
        initialstate::P,
    ) where {E,P,L,M,T}
        return new{E,P,L,M,T}(info, method, model, state, stepper, truncators, initialstate)
    end
    function Simulation(
        info::SimulationInfo,
        method::E,
        model::M,
        state::P,
        stepper::TrotterDecomposition{L},
        truncators::Tuple{T,T},
        ::Nothing,
    ) where {E,P,L,M,T}
        return new{E,P,L,M,T}(info, method, model, state)
    end
end

"""
    quantumstate(obj) -> PEPO

Return the quantum state associated with object `obj`.
"""
quantumstate(s::Simulation) = quantumstate(s.state)
quantumstate(p::PEPO) = p
quantumstate(p::AxialPEPO) = p.parent

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

        return UnitCell{G}(mat)
    end

    return outlat
end

geometrytype(::Type{G}) where {G<:ITC.AbstractUnitCellGeometry} = G
geometrytype(::Type{<:AbstractUnitCell{G}}) where {G} = G
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
    rho0,
    physical,
    virtual,
    vertical_virtual=virtual;
    # Kwargs
    timestep::Float64,
    method::AbstractTimeEvolution,
    order::Int=1,
    storeall::Bool=true,
)
    # model = model2tuple(model)

    # physical, horizontal, vertical = promotelattice(model, physical, horizontal, vertical)

    G = promote_geometry_type(physical, virtual, vertical_virtual)

    stepper, physicalspace = newstepper(
        UnitCell{G}([physical;;]), model, method; timestep=timestep, order=order
    )

    pepo = PEPO(rho0, physical, virtual, vertical_virtual)

    # Construct the time evolving state
    state = newevolution(pepo, stepper, method)

    # Construct the truncators
    truncators = newtruncators(state, stepper, method.truncalg)

    initialstate = storeall ? deepcopy(state) : nothing

    info = SimulationInfo()

    sim = Simulation(info, method, model, state, stepper, truncators, initialstate)

    return sim
end

function Simulation(
    model, pepo::PEPO, method::AbstractTimeEvolution, info=SimulationInfo(); storeall=true
)
    stepper, _ = newstepper(physicalspace(pepo), model, method)

    state = newevolution(pepo, stepper, method)

    truncators = newtruncators(state, stepper, method.truncalg)

    initialstate = storeall ? deepcopy(state) : nothing

    sim = Simulation(info, method, model, state, stepper, truncators, initialstate)

    return sim
end

function initenvironment(network, alg::AbstractBoundaryAlgorithm; kwargs...)
    return ITC.initialize(alg, network; kwargs...)
end

simulate!(sim::Simulation; kwargs...) = simulate!(identity, sim; kwargs...)

"""
    simulate([callback = identity], sim::Simulation; numsteps::Int, kwargs...)

Run the simulation defined in `sim` for `numsteps` iterations. Note, `numsteps` is a keyword
argument. The callable `callback` takes a single argument, the `sim` object itself and is
executed during every shot (default behaviour results in a shot at the end of each step).

# Optional Keywords
- `breaktol::Integer=0.0`: break the simulation when convergence falls below this value.
- `maxshots::Integer=numsteps`: `callback` will be executed at most `maxshots` number of times
    at evenly spaced intervals throughout the time evolution
- `maxtime::Float64=Inf`: maximum wall clock time allowed. If the wall clock time since the
    start `simulate` was called exceeds this value, then no more iterations occur.
- `verbose::Bool=true`: if `true`, log simulation progress at the `@info` log level

See also: [`Simulation`](@ref).
"""
function simulate!(
    callback,
    sim::Simulation;
    numsteps::Integer,
    maxshots::Integer=numsteps,
    maxtime::Float64=Inf,
    breaktol::Float64=0.0,
    verbose::Bool=true,
)
    t0 = time()

    method = sim.method
    pepo = sim.state
    model = sim.model

    # Get simulation infomation and set appropriate fields
    info = sim.info
    setfield!(info, :finished, false)
    setfield!(info, :converged, convergence(info) < breaktol)

    truncalg = method.truncalg

    # Shots are taken at these iterations
    r = range(; start=0, step=round(Int, numsteps / maxshots, RoundUp), stop=numsteps)
    shotsat = collect(r)

    numiter = 0
    tol = Inf

    # Need to copy the weights otherwise they will mutate
    prev_weights = deepcopy(weights(pepo))

    # Only do the first callback if the simulation is just starting, otherwise it will
    # duplicate
    info.iterations == 0 && callback(sim)

    if isdefined(sim, :stepper)
        stepper = sim.stepper
    else
        stepper, _ = newstepper(physicalspace(pepo), model, method)
    end

    if isdefined(sim, :truncators)
        truncator = sim.truncators
    else
        truncator = newtruncators(pepo, stepper, truncalg)
    end

    timestep = sim.stepper.timestep

    while numiter < numsteps
        for (i, step) in enumerate(stepper)
            if eltype(truncator) <: AbstractVector
                trunc = getindex.(truncator, i)
            else
                trunc = truncator
            end

            timestep!(pepo, trunc, step, truncalg)
        end
        # Calculate the difference in pepo singular values between timesteps
        tol = tolerance!(prev_weights, weights(pepo), timestep)
        # tol = 0.0

        numiter += 1
        verbose && @info "Steady state convergence: $tol after $numiter iterations."

        info.convergence = tol
        info.iterations += 1

        # User specified function for post-step processing
        if numiter in shotsat
            pepo isa AxialPEPO && setaxis!(pepo, 1)
            callback(sim)
        end

        if time() - t0 > maxtime
            break
        elseif info.convergence < breaktol
            info.converged = true
            info.finished = true
            @info "Simulation reached break tolerance $(info.convergence) < $breaktol."
            break
        end
    end

    pepo isa AxialPEPO && setaxis!(pepo, 1)

    info.finished = true

    return sim
end

"""
    $(TYPEDSIGNATURES)

Return `sim` with the time step mutated to `timestep`.
"""
function updatetimestep!(sim::Simulation, timestep::Float64)
    space = physicalspace(sim)

    head, body, tail = newstepper(
        space, sim.model, sim.method; order=trotterorder(sim), timestep=timestep
    )

    sim.stepper.head = head
    sim.stepper.body = body
    sim.stepper.tail = tail

    sim.stepper.timestep = timestep

    return sim
end

function dampinghook!(sim::Simulation, xi, output)
    alg = VUMPS(; tol=1e-10, maxiter=50, bonddim=16, verbose=true)

    X, Y, Z = PAULI

    if sim.info.convergence < 1e-6
        pepo = TimeEvolutionPEPO.quantumstate(sim)
        dm = DensityMatrix(pepo, alg)

        rhos = map(ind -> ReducedDensityMatrix(dm, ind), uniqueindices(pepo))

        expt = map(x -> expval(x, X), rhos)

        push!(xi, sim.method.truncalg.ξ)
        push!(output, expt)

        sim.method.truncalg.ξ += 20.0
    end

    return nothing
end

function measurehook!(sim::Simulation, A=PAULI[1]; env=nothing)
    pepo = TimeEvolutionPEPO.quantumstate(sim)

    # alg = VUMPS(; tol=1e-8, maxiter=200, bonddim=32, verbose=true)
    alg = CTMRG(; tol=1e-8, maxiter=200, bonddim=16, verbose=true, randinit=false)
    #
    # env = newcontraction(alg, contractphysical( TimeEvolutionPEPO.tensors(pepo)))
    # renormalize!(env, contractphysical( TimeEvolutionPEPO.tensors(pepo)))

    # dm = DensityMatrix(trivialise(pepo), alg)
    dm = DensityMatrix(pepo, alg)
    rhos = map(ind -> partialtrace(dm, ind), uniqueindices(pepo))
    # dm = DensityMatrix(pepo, env)

    expt = map(x -> expval(x, A), rhos)
    return expt

    # corr = [correlator(dm, A, (1, 1), A, (i, 1)) for i in 2:20]
    # return corr
end

function measurehook!(output::Vector, sim, args...)

    # init

    # alg = CTMRG(; tol=1e-8, maxiter=200, bonddim=16, verbose=true)
    # alg = VUMPS(; tol=1e-10, maxiter=200, bonddim=20, verbose=true)
    alg = CTMRG(; tol=1e-10, maxiter=100, bonddim=16, verbose=true, randinit=true)

    env = nothing #newcontraction(alg, contractphysical(tensors))
    # runcontraction!(env)

    f = sim -> begin
        exp1 = measurehook!(sim, args...; env=env)
        push!(output, exp1)
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

function correlatorhook!(output)
    # alg = VUMPS(; tol=1e-10, maxiter=200, bonddim=16, verbose=true)
    alg = CTMRG(; tol=1e-10, maxiter=200, bonddim=16, verbose=true, randinit=false)
    f = sim -> begin
        dm = DensityMatrix(quantumstate(sim), alg)
        try
            xi = PEPOKit.correlationlength(dm)
            push!(output, xi)
        catch e
            push!(output, NaN)
        end
    end
    return f
end

function fullcorrelatorhook!(output)
    f =
        sim -> begin
            ds, es = PEPOKit.correlationlength(
                D -> VUMPS(; bonddim=D), PEPOKit.quantumstate(sim), 10:1:20
            )
            push!(output, (ds, es))
        end
    return f
end

function negativityhook!(sim::Simulation)
    # if sim.info.iterations < 1
    #     return sim
    # end

    pepo = getaxis(sim)

    alg = VUMPS(; maxiter=100, bonddim=16, verbose=true)

    dm = DensityMatrix(pepo, alg)
    rdm1 = TimeEvolutionPEPO.ReducedDensityMatrix(dm, (1, 1), (2, 1))
    rdm2 = TimeEvolutionPEPO.ReducedDensityMatrix(dm, (1, 1), (3, 1))

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
        w1 => 4,
        s1 => 4,
        n1 => 4,
        w => 4,
        s => 4,
        n => 4,
        e2 => D,
        w2 => D,
        s2 => D,
        n2 => D,
    ) out[k b; e1 e2 s w n] =
        op[kk bb; e1 s1 w1 n1] *
        mpo[k b kk bb; e2 s2 w2 n2] *
        us[s1 s2; s] *
        vw[w; w1 w2] *
        vn[n; n1 n2]
    return out
end
function test_p2s(op, mpo, us, vw, vn)
    @tensoropt (
        k => 2,
        b => 2,
        kk => 2,
        bb => 2,
        e1 => 4,
        w1 => 4,
        s1 => 4,
        n1 => 4,
        w => 4,
        s => 4,
        n => 4,
        e2 => D,
        w2 => D,
        s2 => D,
        n2 => D,
    ) out[k, b, e1, e2, s, w, n] :=
        op[kk, bb, e1, s1, w1, n1] *
        mpo[k, b, kk, bb, e2, s2, w2, n2] *
        us[s1, s2, s] *
        vw[w, w1, w2] *
        vn[n, n1, n2]
    return out
end
