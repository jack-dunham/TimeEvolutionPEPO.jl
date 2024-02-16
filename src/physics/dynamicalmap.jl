function newstepper(pepo::PEPO{G}, model, method) where {G}
    model = Lattice{G}(isa(model, AbstractMatrix) ? model : [model;;], issymmetric(pepo))

    return newstepper(physicalspace(pepo), model, method)
end
# Explicitly isotropic
function newstepper(
    physical::AbstractMatrix{<:IndexSpace},
    model::AbstractMatrix,
    method::AbstractTimeEvolution,
)
    stepper = timestepper(physical, model, method)
    return stepper, stepper
end

# Anisotropic
function newstepper(
    physical::AbstractMatrix{<:IndexSpace},
    model::NTuple{2,AbstractMatrix},
    method::AbstractTimeEvolution,
)
    horizontal = timestepper(physical, method, model[1])
    vertical = timestepper(physical, method, model[2])

    return horizontal, vertical
end

function timestepper(physical::AbstractUnitCell, model::AbstractUnitCell, method::DMPO)
    tau = method.timestep / method.trotterorder

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

    return GlobalStepper(t)
end

function timestepper(physical::AbstractUnitCell, model::AbstractUnitCell, method::TEBD)
    update = method.updatestyle
    order = method.trotterorder
    timestep = method.timestep
    compress = method.compress
    return tebdtimestepper(
        update, model, physical; order=order, timestep=timestep, compress=compress
    )
end
function tebdtimestepper(
    style::Local, model::AbstractUnitCell, physical::AbstractUnitCell; kwargs...
)
    u, v = tebdtimestepper(model, physical; kwargs...)
    return LocalStepper(u, v)
end
function tebdtimestepper(
    style::Global, model::AbstractUnitCell, physical::AbstractUnitCell; kwargs...
)
    u, v = tebdtimestepper(model, physical; kwargs...)
    return GlobalStepper(u, v)
end
function tebdtimestepper(
    model::AbstractUnitCell, physical::AbstractUnitCell; order, timestep, compress
)
    tau = timestep / order

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

    return u, v
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

    # ket1 * ket1 * bra2 * bra2
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

    @info "MPO bond dimension: $D"

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
