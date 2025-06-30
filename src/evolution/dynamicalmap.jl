mutable struct TrotterDecomposition{T}
    head::Vector{T}
    body::Vector{T}
    tail::Vector{T}
    timestep::Float64
    const order::Integer
end

Base.eltype(::TrotterDecomposition{T}) where {T} = T

function TrotterDecomposition(head, body, tail; timestep, order)
    return TrotterDecomposition(head, body, tail, timestep, order)
end

initialize(trotter::TrotterDecomposition) = trotter.head
finalize(trotter::TrotterDecomposition) = trotter.body

Base.iterate(sch::TrotterDecomposition, state=1) = iterate(sch.body, state)

Base.size(t::TrotterDecomposition) = (length(t),)
Base.length(t::TrotterDecomposition) = length(t.body)

function newtruncators(
    pepo, sequence::TrotterDecomposition, truncalg::AbstractTruncationScheme
)
    rv = map(sequence.body) do propagator
        newtruncators(pepo, propagator, truncalg)
    end
    return rv
end

# function construct_stepper_list(physical, model, method)
#     model = Iterators.flatten(model)
#
#     trotterlist = map(model) do matrix_of_callables
#
#         G = promote_geometry_type(physical, matrix_of_callables)
#
#         S = stepper_type(typeof(matrix_of_callables), method)
#
#         rvf =
#             dt -> begin
#                 data = broadcast(physical, matrix_of_callables) do space, callable
#                     return callable(space, method)(dt)
#                 end
#
#                 lat = Lattice{G}(data)
#
#                 return S(lat)
#             end
#
#         return rvf
#     end
#
#     tau =method.timestep
#
#     bulk = map(t -> t(tau), trotterlist)
#
#     maxsize = size(first(bulk))
#
#     physicalspace = broadcast((p, _) -> p, physical, CartesianIndices(maxsize))
#
#     T = eltype(bulk)
#
#
#     return Schedule(T[], bulk, T[], method.numsteps), physicalspace
# end

onvector(v::Vector) = v
function onvector(t)
    v = typeof(t)[]
    push!(v, t)
    return v
end

function trotterlayers(model::Hamiltonian, theterms)
    for term in theterms
        if isexpandable(term)
            model += sum(trotterlayers(model, terms(term)))
        else
            model += TrotterLayer(zero(model) + term)
        end
    end

    return model
end

function trotterlayers(model::Hamiltonian)
    rv = Hamiltonian((), oneterms(model))

    return trotterlayers(rv, model)
end

# Obtain the trotter layers from the Liouvillian
function trotterlayers(model::Liouvillian)
    ham = trotterlayers(model.hamiltonian)
    return Liouvillian(ham, model.linops)
end

# function trotterlayers(model::Liouvillian) 
#
#     H = Hamiltonian((), onelocalterms(model), false)
#
#     L = Liouvillian(H, linops(model), false)
#
#     return trotterlayers(L, model.hamiltonian)
# end

function distributelocal(layers, localops, ::Nothing)
    numl = length(layers)

    @debug "distributing local operators:" numlayers = numl numlocal = length(localops)

    if !isempty(localops)
        layers = map(step -> step + 1 / numl * sum(localops), layers)
    end

    return layers
end

function distributelocal(layers, localops, linops)
    distributelocal(layers, localops, nothing)

    numl = length(layers)

    foreach(layer -> addlinops(layer, sqrt(1 / numl) .* linops), layers)

    return layers
end

function _trotterise(physical, layers, method, localops, linops)

    # @warn length(localops)

    steps = distributelocal(layers, localops, linops)

    trotterlist = _trotterise(physical, steps, method)

    return trotterlist
end

function _trotterise(physical, steps, method, ::InReverse)
    return _trotterise(physical, map(InReverse, steps), method)
end

function _trotterise(physical, steps, method, _=nothing)
    @debug "Trotterising over $(length(steps)) layers"

    trotterlist = map(steps) do modelstep
        S = applykwargs(stepper_type(method), modelstep)

        rvf =
            dt -> begin
                lat = broadcastmodel(geometrytype(physical), modelstep, physical, method)(dt)

                v = onvector(lat)

                return map(S, v)
            end

        return rvf
    end

    return trotterlist
end

function trotterise(
    physical, model, method::Union{GlobalTEBD,GlobalClusterMPO}, reverse=nothing
)
    steps = combineaxial(model.layers)

    trotterlist = _trotterise(physical, steps, method, reverse)

    return trotterlist
end

function trotterise(
    physical, model, method::Union{TEBD,AxialTEBD,ClusterMPO}, reverse=nothing
)
    @debug "seperating..."
    steps = seperateaxis(model.layers)

    trotterlist = _trotterise(physical, steps, method, reverse)

    return trotterlist
end

function trotterise(physical, model, method::ClusterPEPO, reverse=nothing)

    # oneloc = oneterms(model)

    trotterlist = _trotterise(physical, model.layers, method, reverse)

    return trotterlist
end

# Given a sequence of trotter-layers, combine consequtive layers if on different axis
function combineaxial(steps)
    steps = [steps...]

    rv = Combine[]

    while !isempty(steps)
        op = popfirst!(steps)

        if op.axis != 0
            if !isempty(steps) && op.axis != steps[begin]
                # If on different axis, then they can be combined
                opnext = popfirst!(steps)
                push!(rv, Combine(op, opnext))
            else
                # Otherwise they cannot be so combine with identity
                opid = TrotterLayer(zero.(op.data), mod(op.axis + 1, 1:2))
                push!(rv, Combine(op, opid))
            end
        else
            push!(rv, Combine(changeaxis(op, 1), changeaxis(op, 2)))
        end
    end
    return rv
end

# Given a sequence of trotter-layers, extract them into axial components
function seperateaxis(steps)
    if any(p -> p.axis != 0, steps)
        # Symmetry broken 
        rv = TrotterLayer[]
        for op in steps
            if op.axis != 0
                push!(rv, op)
            else
                append!(rv, [changeaxis(op, 1; a=0.5), changeaxis(op, 2; a=0.5)])
            end
        end
    else
        rvx = TrotterLayer[]
        rvy = TrotterLayer[]

        for op in steps
            push!(rvx, changeaxis(op, 1; a=0.5))
            push!(rvy, changeaxis(op, 2; a=0.5))
        end

        rv = vcat(rvx, rvy)
    end

    return rv
end

# TODO this is hacky and convoluted and needs more thought
function newstepper(physical, model, method; timestep, order)
    trotterlist = trotterise(physical, model, method)

    tau = timestep
    T = stepper_type(method, physical)

    if length(trotterlist) == 1 && method isa ClusterPEPO
        # No trotterisation required if we have made it this far with only one operator 
        # using ClusterPEPO
        # in the list
        theop = only(only(trotterlist)(tau))

        return TrotterDecomposition(T[], T[theop], T[]; timestep=timestep, order=order)
    end

    if order == 1
        @info "First-order trotter decomposition..."
        body = mapreduce(op -> op(tau), vcat, trotterlist)
    elseif order == 2
        @info "Second-order trotter decomposition..."
        body = mapreduce(op -> op(tau / 2), vcat, trotterlist)
        reversed = reverse(trotterise(physical, model, method, InReverse))
        append!(body, mapreduce(op -> op(tau / 2), vcat, reversed))
    end

    oplist = TrotterDecomposition(T[], T[body...], T[]; timestep=timestep, order=order)

    @info "...of $(length(body)) operators"

    return oplist
end

function timestepper(
    physical::AbstractUnitCell,
    model::AbstractUnitCell,
    method::ClusterMPO;
    tau=method.timestep,
)
    # Calculate the time-evolution mpo for each unique mpo
    tempo = broadcast(
        x -> clustermpo(
            x,
            tau,
            method.order;
            extend=method.extend,
            compress=method.compress,
            approx=method.approx,
        ),
        model,
    )

    t = broadcast(eachindex(physical), physical) do ind, p
        return clustermpotensor(tempo[ind[1], ind[2]], p)
    end

    rv = [(t, t)]

    return rv
end

function timestepper(
    physical::AbstractUnitCell, model::AbstractUnitCell, method::TEBD; tau=method.timestep
)
    update = method.updatestyle
    order = method.trotterorder
    timestep = method.timestep
    compress = method.compress

    rv = tebdtimestepper(update, model, physical; tau=tau, compress=compress)

    return rv
end

# function timestepper(physical::AbstractUnitCell, model::AbstractUnitCell, method::TETNO)
#     newmethod = ClusterMPO(timestep=method.timestep, numsteps=method.numsteps)
#
#     # e,w
#     # s,n
#
#     stepper = timestepper(physical, model, newmethod).t[1]
#
#
#     newstepper = broadcast(stepper) do t
#         @tensoropt begin
#             op[ook oob ik ib; e s w n] := t[ok ob ik ib; e w] * t[ook oob ok ob; s n]
#         end
#         return op
#     end
#     return FullStepper(newstepper)
# end

function timestepper(
    physical::AbstractUnitCell, model::AbstractUnitCell, method::ClusterPEPO; tau
)
    return FullStepper(model)
end
# function timestepper(physical::AbstractUnitCell, model::AbstractUnitCell, method::TETNO)
#     return SplitStepper(model)
# end

function tebdtimestepper(
    style::LocalUpdate, model::AbstractUnitCell{G,L}, physical::AbstractUnitCell; kwargs...
) where {G,L}
    u, v = generic_tebd_time_stepper(model, physical; kwargs...)

    rv = map(identity, zip(u, v))

    return rv
end
function tebdtimestepper(
    style::GlobalUpdate, model::AbstractUnitCell{G,L}, physical::AbstractUnitCell; kwargs...
) where {G,L}
    u, v = generic_tebd_time_stepper(model, physical; kwargs...)

    rv = map(twobondstepper, u, v)

    return rv
end

function generic_tebd_time_stepper(
    model::AbstractUnitCell{G,L}, physical::AbstractUnitCell; tau, compress
) where {G,L}
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

    return [u], [v]
end

# This method takes a matrix as an input
function tebdtensor(model::AbstractMatrix, p1::IndexSpace, p2::IndexSpace, dt; compress)
    spacecod = p1 * p2 * p1' * p2'
    spacedom = p1 * p2 * p1' * p2'

    symdata = zeros(ComplexF64, dims(spacecod * spacedom))

    copyto!(symdata, exp(dt * model))

    tensor = TensorMap(symdata, spacecod, spacedom)
    # move site one to codomain and site two to domain
    tensor = permute(tensor, (1, 3, 5, 7), (2, 4, 6, 8))

    u, s, v, err = tsvd(tensor; trunc=compress)

    @info "Compression error: $err for η=$(dim(codomain(s)))"

    return u * sqrt(s), _transpose(sqrt(s) * v)
end

function clustermpotensor(mpo, p::S) where {S<:IndexSpace}
    d = dim(p)
    D = sum(x -> size(x, 1), mpo[:, 1])

    @info "MPO bond dimension: $D"

    s = ⊕(oneunit(S)^D...)
    array = zeros(ComplexF64, d, d, d, d, D, D)

    L = size(mpo, 1)

    matrix = reduce(vcat, [reduce(hcat, mpo[i, :]) for i in 1:L])

    for (ind, op) in pairs(matrix)
        slice = view(array, :, :, :, :, ind)
        slice .= reshape(op, size(slice))
    end

    tensor = TensorMap(array, p * p' * p' * p, s * s')

    return permutedom(tensor, (2, 1))
end
