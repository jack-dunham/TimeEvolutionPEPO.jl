function newstepper(
    ::PEPO, stepper::NTuple{N,<:AbstractMatrix{T}}, method
) where {N,T<:AbstractTimeStepper}
    return stepper
end

function newstepper(physical::Lattice, model::NTuple, method 
                    ) where {S<:TensorSpace}
    return construct_stepper_list(physical, model, method)
end

function newstepper(pepo::PEPO{G}, model::Tuple, method) where {G}
    model = map(Lattice{G}, model)
    return newstepper(physicalspace(pepo), model, method)
end

# Explicitly isotropic
function newstepper(
    physical::AbstractMatrix{<:IndexSpace},
    model::NTuple{1,AbstractMatrix},
    method::Union{TETNO,TEPEPSO},
)

    println(typeof(model))

    onlystep = timestepper(physical, only(model), method; tau=method.timestep)
    return [onlystep]
end

#
# function newstepper(
#     physical::AbstractMatrix{<:IndexSpace},
#     model::NTuple{2,AbstractMatrix},
#     method::AbstractTimeEvolution,
# )
#
#     hori = timestepper(physical, model[1], method)
#     vert = timestepper(physical, model[2], method)
#
#     rv = newstepper(hori, vert, method)
#
#     return rv
# end

function construct_stepper_list(
    physical, model::NTuple{2}, method; tau=method.timestep
)
    xtensorsteps = timestepper(physical, model[1], method; tau=tau)

    xsteps = map(xtensorsteps) do s
        return stepper(s, method; axis=1)
    end

    if N == 1
        ytensorsteps = xtensorsteps
    elseif N == 2
        ytensorsteps = timestepper(physical, model[2], method; tau=tau)
    end

    ysteps = map(ytensorsteps) do s
        return stepper(s, method; axis=2)
    end

    return xsteps, ysteps
end

mutable struct Schedule{T}
    start::Vector{T}
    bulk::Vector{T}
    finish::Vector{T}
    numiter::Int
end

Base.length(sch::Schedule) =length(sch.start) + (sch.numiter - 1) * length(sch.bulk) + length(sch.finish)

Base.eltype(::Schedule{T}) where {T} = T

function Base.iterate(sch::Schedule{T}, state = ()) where {T}

    reps = Iterators.repeated(sch.bulk, sch.numiter - 1)
    itr = Iterators.flatten((sch.start, reps, sch.finish))

    return Base.iterate(itr, state)
end

function construct_stepper_list(physical, model::NTuple{1, <:Vector}, method::TEPEPSO)
    trotterlist = map(only(model)) do matrix_of_callables
        G = promote_geometry_type(physical, matrix_of_callables)

        rvf = dt -> begin 
            data = broadcast(physical, matrix_of_callables) do space, callable
                return callable(space, method)(dt)
            end

            lat = Lattice{G}(data)

            return FullStepper(lat)
        end
    end

    tau = method.timestep

    if length(trotterlist) == 1
        pepsof = only(trotterlist)
        return only(trotterlist)(tau)
    end

    startend = first(trotterlist)(tau / 2)
    bulkend = first(trotterlist)(tau)

    bulkops = typeof(startend)[]

    foreach(op -> push!(bulk, op(tau / 2)), trotterlist[begin+1:end-1])
    rbulkops = reverse(bulkops)

    push!(bulkops, last(trotterlist)(tau))
    append!(bulkops, rbulkops)
    push!(bulkops, bulkend)

    # bulkitr = Iterators.repeated(bulkops, method.numsteps - 1)

    restitr = push!(bulkops[begin:end - 1], startend)

    maxsize = maximum(size, restitr)
    physicalspace = broadcast((p,_) -> p, physical, CartesianIndices(maxsize))

    return Schedule([startend], bulkops, restitr, method.numsteps), physicalspace
end

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
) where {G,L<:AbstractLindbladEquation}
    u, v = generic_tebd_time_stepper(model, physical; kwargs...)

    rv = map(identity, zip(u, v))

    return rv
end
function tebdtimestepper(
    style::GlobalUpdate, model::AbstractUnitCell{G,L}, physical::AbstractUnitCell; kwargs...
) where {G,L<:AbstractLindbladEquation}
    u, v = generic_tebd_time_stepper(model, physical; kwargs...)

    rv = map(twobondstepper, u, v)

    return rv
end

function generic_tebd_time_stepper(
    model::AbstractUnitCell{G,L}, physical::AbstractUnitCell; tau, compress
) where {G,L<:AbstractLindbladEquation}
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

localvectorise(term::AbstractTerm) = localvectorise(typeof(term))(Matrix{ComplexF64}(term))
function localvectorise(::Type{LindbladTerm})
    anticomm = L -> kron(L' * L, one(L)) + kron(one(L), transpose(L' * L))

    return L -> kron(L, transpose(L')) - 1 / 2 * anticomm(L)
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

    # ket1 * ket2 * bra1 * bra2
    spacecod = p1 * p2 * p1' * p2'
    spacedom = p1 * p2 * p1' * p2'

    data = zeros(ComplexF64, dim(spacecod), dim(spacedom))
    symdata = zeros(ComplexF64, dims(spacecod * spacedom))

    # We divide by half again for local terms as in 2D we effectively apply these twice
    data = 1 / 2 * sum(vectorise, model.onelocal; init=data)
    data = 1 / 2 * sum(vectorise, model.linops; init=data)

    data = sum(vectorise, model.twolocal; init=data)

    copyto!(symdata, exp(dt * data))

    tensor = TensorMap(symdata, spacecod, spacedom)

    # move site one to codomain and site two to domain
    tensor = permute(tensor, (1, 3, 5, 7), (2, 4, 6, 8))

    u, s, v = tsvd(tensor; trunc=compress)

    return u * sqrt(s), _transpose(sqrt(s) * v)
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
