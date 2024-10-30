struct TrotterDecomposition{T}
    head::Vector{T}
    body::Vector{T}
    tail::Vector{T}
end

Base.eltype(::TrotterDecomposition{T}) where {T} = T

Base.iterate(sch::TrotterDecomposition, state=1) = iterate(sch.body, state)

function Base.size(t::TrotterDecomposition)
    return maximum(size, Iterators.flatten((t.head, t.body, t.tail)))
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

function trotterise(physical, steps, method)
    trotterlist = map(steps) do modelstep
        G = promote_geometry_type(physical, modelstep)

        S = applykwargs(stepper_type(method), modelstep)

        rvf = dt -> begin
            lat = broadcastmodel(G, modelstep, physical, method)(dt)

            v = onvector(lat)

            return map(S, v)
        end

        return rvf
    end

    return trotterlist
end

# TODO this is hacky and convoluted and needs more thought
function newstepper(physical, model, method)
    order = method.trotterorder

    # Apply wrappers etc as needed and makes sure everything is on matrices
    steps = processmodel(model, method)

    trotterlist = trotterise(physical, steps, method)

    tau = method.timestep

    # if length(trotterlist) == 1 
    #     # No trotterisation required if we have made it this far with only one operator
    #     # in the list
    #     theop = only(only(trotterlist)(tau))
    #
    #     physicalspace = broadcast((p, _) -> p, physical, CartesianIndices(size(theop)))
    #
    #     T = typeof(theop)
    #
    #     return TrotterDecomposition(T[], [theop], T[]), physicalspace
    # end

    if method.trotterorder == 1
        @info "First-order trotter decomposition..."
        body = mapreduce(op -> op(tau), vcat, trotterlist)
    elseif method.trotterorder == 2
        @info "Second-order trotter decomposition..."
        body = mapreduce(op -> op(tau / 2), vcat, trotterlist)
        reversed = reverse(trotterise(physical, map(InReverse, steps), method))
        append!(body, mapreduce(op -> op(tau / 2), vcat, reversed))
    end

    T = eltype(body)

    oplist = TrotterDecomposition(T[], body, T[])

    physicalspace = broadcast((p, _) -> p, physical, CartesianIndices(size(oplist)))

    @info "...of $(length(body)) operators"

    return oplist, physicalspace
end

# function construct_stepper_list(physical, model, method)
#
#     if !(eltype(model)<:AbstractMatrix)
#         model = map(m -> [m;;], model)
#     end
#
#     trotterlist = broadcast(model) do matrix_of_callables
#
#         G = promote_geometry_type(physical, matrix_of_callables)
#
#         S = stepper_type(eltype(matrix_of_callables), method)
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
#     tau = method.timestep
#
#     if length(trotterlist) == 1
#         pepsof = only(trotterlist)(tau)
#
#         maxsize = size(pepsof)
#
#         physicalspace = broadcast((p, _) -> p, physical, CartesianIndices(maxsize))
#
#         T = typeof(pepsof)
#
#         return Schedule(T[], [pepsof], T[], method.numsteps), physicalspace
#     end
#
#     startend = first(trotterlist)(tau / 2)
#     bulkend = first(trotterlist)(tau)
#
#     bulkops = typeof(startend)[]
#
#     foreach(op -> push!(bulkops, op(tau / 2)), trotterlist[(begin + 1):(end - 1)])
#     rbulkops = reverse(bulkops)
#
#     push!(bulkops, last(trotterlist)(tau))
#     append!(bulkops, rbulkops)
#     push!(bulkops, bulkend)
#
#     # bulkitr = Iterators.repeated(bulkops, method.numsteps - 1)
#
#     restitr = push!(bulkops[begin:(end - 1)], startend)
#
#     maxsize = maximum(size, restitr)
#     physicalspace = broadcast((p, _) -> p, physical, CartesianIndices(maxsize))
#
#     return Schedule([startend], bulkops, restitr, method.numsteps), physicalspace
# end

function second_order_trotter(physical, model, method; tau=method.timestep)
    xsteps, ysteps = construct_stepper_list(physical, model, method; tau=tau / 2)

    println(length(xsteps))

    steps = copy(xsteps)

    append!(steps, ysteps)
    append!(steps, map(reverse, reverse(ysteps)))
    append!(steps, map(reverse, reverse(xsteps)))

    return steps
end

function fourth_order_trotter(physical, model, method; tau=method.timestep)
    tau1 = tau2 = tau / (4 - cbrt(4))
    tau3 = tau - 2 * tau1 - 2 * tau2

    alltaus = (tau1, tau2, tau3)

    u1, u2, u3 = map(alltaus) do t
        return second_order_trotter(physical, model, method; tau=t)
    end

    return [u1..., u2..., u3..., u2..., u1...]
end

function _newstepper(
    hori::L, vert::L, method::AbstractTimeEvolution
) where {L<:AbstractLattice}
    order = method.trotterorder

    steps = map(s -> stepper(s, method; axis=1), hori)
    append!(steps, map(s -> stepper(s, method; axis=2), vert))

    if order == 2
        append!(steps, map(reverse, reverse(steps)))
    end

    return steps
end

# Interleaved
function newstepper(
    physical::AbstractMatrix{<:IndexSpace},
    model::NTuple{N,AbstractMatrix},
    method::AbstractTimeEvolution,
) where {N}
    order = trotterorder(method)
    tau = method.timestep

    if order == 1
        return construct_stepper_list(physical, model, method)
    elseif order == 2
        return second_order_trotter(physical, model, method)
    elseif order == 4
        return fourth_order_trotter(physical, model, method)
    end
end

stepper(step, method::TEBD; kwargs...) = stepper(step, method.updatestyle; kwargs...)
function stepper(step, ::LocalUpdate; axis, isreversed=false)
    return OneBondStepper(step, axis, isreversed)
end
function stepper(step, ::Union{GlobalUpdate,DMPO}; axis, isreversed=false)
    return TwoBondStepper(step, axis, isreversed)
end

# function timestepper(physical::AbstractUnitCell, model::NTuple{1, T}, method::AbstractTimeEvolution; kwargs...) where {T}
#     steps = timestepper(physical, only(model), method; kwargs...)
#     return steps, steps
# end
# function timestepper(physical::AbstractUnitCell, model::NTuple{2, T}, method::AbstractTimeEvolution; kwargs...) where {T}
#
#     xmod,ymod = model
#
#     xsteps = timestepper(physical, xmod, method; kwargs...)
#     ysteps = timestepper(physical, ymod, method; kwargs...)
#
#     return xsteps, steps
# end

function timestepper(
    physical::AbstractUnitCell, model::AbstractUnitCell, method::DMPO; tau=method.timestep
)
    # Calculate the time-evolution mpo for each unique mpo
    tempo = broadcast(
        x -> dynamicalmpo(
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
        return dmpotensor(tempo[ind[1], ind[2]], p)
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
#     newmethod = DMPO(timestep=method.timestep, numsteps=method.numsteps)
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
    physical::AbstractUnitCell, model::AbstractUnitCell, method::Union{TETNO,TEPEPSO}; tau
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

function dmpotensor(mpo, p::S) where {S<:IndexSpace}
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
