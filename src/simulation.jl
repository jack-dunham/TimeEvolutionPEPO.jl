"""
$(TYPEDEF)

Concrete type representing mutable state of a simulation. Note, the fields of `Simulation`
are internal and not part of the public API.

# Constructors

    $(FUNCTIONNAME)(pepo::PEPO, model, method::AbstractMethod; timestep::Float64, kwargs...)

Initialise a `Simulation` of the model `model` using `method` with initial quantum state 
defined by `pepo`. A `timestep` must be supplied as a keyword argument. An addition, the 
following optional keywords can also be used:

## Optional Keywords

- `order::Int=1`: order of the trotter decomposition (if any).
- `storeall::Bool=true`: this should be set to false if one does not wish to store objects
    that can be reconstructed; useful if, for example, very large tensors are used for the
    time evolution. Repeated calls to `simulate!` will be slower if this is set to `true`.

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
        ::TrotterDecomposition{L},
        ::Tuple{T,T},
        ::Nothing,
    ) where {E,P,L,M,T}
        return new{E,P,L,M,T}(info, method, model, state)
    end
end

function Base.show(io::IO, m::MIME"text/plain", sim::Simulation)
    println(io, "SIMULATION")
    println(io, "≡≡≡≡≡≡≡≡≡≡")
    println(io, "size: $(size(quantumstate(sim)))")
    println(io, "timestep: $(timestep(sim))")
    println(io, "METHOD")
    println(io, "≡≡≡≡≡≡")
    show(io, m, sim.method)
    println(io, "")
    println(io, "INFO")
    print(io, "≡≡≡≡")
    for name in fieldnames(SimulationInfo)
        println(io, "")
        print(io, name, ": ", getfield(sim.info, name))
    end
    return nothing
end

unitcellsize(x::AbstractMatrix) = size(x)
unitcellsize(_) = (1, 1)

similar_matrix(A::AbstractMatrix, outsize) = Matrix{eltype(A)}(undef, outsize)
similar_matrix(A, outsize) = Matrix{typeof(A)}(undef, outsize)

geometrytype(::Type{G}) where {G<:TRGroups.AbstractUnitCellGeometry} = G
geometrytype(::Type{<:AbstractUnitCell{G}}) where {G} = G
geometrytype(::Type{<:AbstractMatrix}) = Square
geometrytype(T::DataType) = Square
geometrytype(g) = geometrytype(typeof(g))

function promote_geometry_type(
    A::Type{G}, B::Type{G}
) where {G<:TRGroups.AbstractUnitCellGeometry}
    return G
end
promote_geometry_type(args...) = mapreduce(geometrytype, promote_geometry_type, args)
function promote_geometry(A::Type, B::Type)
    throw(ErrorException("TODO: incompatible lattice geometries"))
end

copyspaceto!(mat, arg::AbstractMatrix) = copyto!(mat, arg)
copyspaceto!(mat, arg::TensorSpace) = copyto!(mat, [arg;;])

@deprecate(
    Simulation(model, rho0, physical, virtual; method, kwargs...),
    Simulation(PEPO(rho0, physical, virtual), model, method; kwargs...)
)

function Simulation(
    pepo::PEPO,
    model,
    method::AbstractTimeEvolution,
    info=SimulationInfo();
    timestep,
    storeall=true,
    order=1,
)
    stepper = newstepper(physicalspace(pepo), model, method; order=order, timestep=timestep)

    state = newevolution(pepo, stepper, method)

    truncators = newtruncators(state, stepper, method.alg)

    initialstate = storeall ? deepcopy(state) : nothing

    sim = Simulation(info, method, model, state, stepper, truncators, initialstate)

    return sim
end

function initenvironment(network, alg::AbstractBoundaryAlgorithm; kwargs...)
    return Renormalization(network, alg; kwargs...)
end

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
- `numinit::Integer=0`: skip this many iterations. Useful if you are resuming a simulation but wish 
    to maintain the same sot schedule
- `verbose::Bool=true`: if `true`, log simulation progress at the `@info` log level

See also: [`Simulation`](@ref).
"""
simulate!(sim::Simulation; kwargs...) = simulate!(identity, sim; kwargs...)
function simulate!(
    callback,
    sim::Simulation;
    numsteps::Integer,
    maxshots::Integer=numsteps,
    maxtime::Float64=Inf,
    breaktol::Float64=0.0,
    verbose::Bool=true,
    numinit::Integer=0,
)
    initial_wall_time = time()

    method = sim.method
    pepo = sim.state
    model = sim.model

    # Get simulation infomation and set appropriate fields
    info = getfield(sim, :info)
    info.finished = false
    info.converged = convergence(info) < breaktol

    total_wall_duration = info.walltime
    total_sim_duration = info.simtime

    truncalg = method.alg

    # Shots are taken at these iterations
    r = range(; start=0, step=round(Int, numsteps / maxshots, RoundUp), stop=numsteps)
    shotsat = collect(r)

    # Need to copy the weights otherwise they will mutate
    prev_weights = deepcopy(bondweights(pepo))

    # Only do the first callback if the simulation is just starting, otherwise it will
    # duplicate
    info.iterations == 0 && callback(sim)

    if isdefined(sim, :stepper)
        stepper = sim.stepper
    else
        stepper = newstepper(physicalspace(pepo), model, method)
    end

    if isdefined(sim, :truncators)
        truncator = sim.truncators
    else
        truncator = newtruncators(pepo, stepper, truncalg)
    end

    numinit > 0 && @info "Resuming simulation from $numinit iterations."
    numiter = numinit

    tol = Inf

    while numiter < numsteps
        timestep!(pepo, truncator, stepper, truncalg)
        # Calculate the difference in pepo singular values between timesteps
        tol = tolerance!(prev_weights, bondweights(pepo), timestep(sim))

        numiter += 1
        verbose && @info "Steady state convergence: $tol after $numiter iterations."

        info.convergence = tol
        info.iterations += 1

        # We compute the delta's like this to avoid floating point rounding errors you would obtain if
        # incrementing the total times. Less important for wall time, but for sim time you often expect 
        # an exact final time.
        sim_time_delta = (numiter - numinit) * timestep(sim)
        info.simtime = sim_time_delta + total_sim_duration

        wall_time_delta = time() - initial_wall_time
        info.walltime = wall_time_delta + total_wall_duration

        # User specified function for post-step processing
        if numiter in shotsat
            pepo isa AxialPEPO && setaxis!(pepo, 1)
            callback(sim)
        end

        if wall_time_delta > maxtime
            @info "Maximum simulation time $wall_time_delta > $maxtime exceeded after $numiter iterations."

            pepo isa AxialPEPO && setaxis!(pepo, 1)
            callback(sim)

            # Return early
            return sim
        elseif info.convergence < breaktol
            info.converged = true

            @info "Converged."

            pepo isa AxialPEPO && setaxis!(pepo, 1)
            callback(sim)

            break
        end
    end

    pepo isa AxialPEPO && setaxis!(pepo, 1)

    info.finished = true

    @info "Finished."
    @info "Steady state convergence: $tol after $numiter iterations."
    @info "Total iterations for this Simulation: $(info.iterations))."

    return sim
end

"""
    $(TYPEDSIGNATURES)

Return `sim` with the time step mutated to `timestep`.
"""
function updatetimestep!(sim::Simulation, timestep::Float64)
    space = physicalspace(quantumstate(sim))

    new_decomposition = newstepper(
        space, sim.model, sim.method; order=trotterorder(sim), timestep=timestep
    )

    sim.stepper.head = new_decomposition.head
    sim.stepper.body = new_decomposition.body
    sim.stepper.tail = new_decomposition.tail

    sim.stepper.timestep = timestep

    return sim
end

# function Base.getproperty(::Simulation, ::Symbol)
#     throw(ArgumentError("field access is an unsupported interface for `Simulation` objects"))
#     return nothing
# end

## INTERFACE
"""
    $(TYPEDSIGNATURES)

Return the quantum state associated with object a simulation.
"""
quantumstate(s::Simulation) = quantumstate(getfield(s, :state))
quantumstate(p::PEPO) = p
quantumstate(p::AxialPEPO) = p.parent

"""
    $(TYPEDSIGNATURES)

Return the order of the Trotter decomposition of a simulation.
"""
trotterorder(sim::Simulation) = sim.stepper.order

"""
    $(TYPEDSIGNATURES)

Return the time step used by a simulation.
"""
timestep(sim::Simulation) = sim.stepper.timestep

"""
    $(TYPEDSIGNATURES)

Return a named tuple containing simulation information. See [`SimulationInfo`](@ref)
for details of the containted fields.
"""
function simulationinfo(s::Simulation)
    info = getfield(s, :info)

    names = fieldnames(SimulationInfo)

    values = map(fname -> getfield(info, fname), names)

    return NamedTuple{names}(values)
end

### REMOVE THESE BELOW

function dampinghook!(sim::Simulation, xi, output)
    alg = VUMPS(; tol=1e-10, maxiter=50, bonddim=16, verbose=true)

    X, Y, Z = PAULI

    if sim.info.convergence < 1e-6
        pepo = TimeEvolutionPEPO.quantumstate(sim)
        dm = DensityMatrix(pepo, alg)

        rhos = map(ind -> ReducedDensityMatrix(dm, ind), uniqueindices(pepo))

        expt = map(x -> expval(x, X), rhos)

        push!(xi, sim.method.alg.zeta)
        push!(output, expt)

        sim.method.alg.zeta += 20.0
    end

    return nothing
end

function measurehook!(sim::Simulation, A=PAULI[1]; env=nothing)
    pepo = TimeEvolutionPEPO.quantumstate(sim)

    # alg = VUMPS(; tol=1e-8, maxiter=200, bonddim=16, verbose=true)
    alg = CTMRG(; tol=1e-7, maxiter=200, bonddim=16, verbose=true, randinit=false)
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
    alg = CTMRG(; tol=1e-10, maxiter=100, bonddim=16, verbose=true, randinit=false)

    env = nothing #newcontraction(alg, contractphysical(tensors))
    # runcontraction!(env)

    f = sim -> begin
        exp1 = measurehook!(sim, args...; env=env)
        push!(output, exp1)
        return nothing
    end

    return f
end

function vumpshook!(output, A)
    return sim -> begin
        pdm = PurifiedDensityMatrix(quantumstate(sim), VUMPS(; bonddim=16, tol=1e-8))

        rdm = partialtrace(pdm, (1, 1))

        m = abs(expval(rdm, A))
        l = correlationlength(pdm)

        push!(output, (m, l))

        return nothing
    end
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
