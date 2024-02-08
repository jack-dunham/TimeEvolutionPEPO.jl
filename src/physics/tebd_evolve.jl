struct SequentialUpdate end
struct ParallelUpdate end

function initevolution(pepo::AbstractPEPO, map, method)
    xpepo = _initevolution(pepo, map[1])
    ypepo = _initevolution(changeaxis(pepo), map[2])
    return xpepo, ypepo
end
# function _initevolution(pepo::AbstractPEPO, map::TrotterGates, method::TEBD)
#     return _initevolution(pepo, map, method.updatestyle)
# end

function _initevolution(pepo::AbstractPEPO, map::LocalStepper)
    evolved = broadcast(getstate(pepo), map.u, map.v) do p, u, v
        D1, D2, D3, D4 = domain(p)

        TL = promote_type(eltype(p), eltype(u))
        TR = promote_type(eltype(p), eltype(v))

        left = TensorMap(undef, TL, codomain(p), fuse(D1 * domain(u)) * D2 * D3 * D4)
        right = TensorMap(undef, TR, codomain(p), D1 * D2 * fuse(D3 * domain(v))' * D4)

        return left, right
    end

    firstbond = Ref(Bond(CartesianIndex(1, 1), CartesianIndex(2, 1)))

    return LocalUpdatablePEPO(pepo, evolved, firstbond, Ref(false))
end

function _initevolution(pepo::AbstractPEPO, map::TrotterGates, ::GlobalUpdate)
    function useful_f(x, l, r)
        D = domain(x)
        lη = domain(l)
        rη = domain(r)
        left = TensorMap(
            undef,
            promote_type(eltype(x), eltype(l)),
            codomain(x),
            fuse(D[1] * lη) * D[2] * fuse(D[3] * rη)' * D[4],
        )
        return left
    end

    evolved = useful_f.(getstate(pepo), map.east, circshift(map.west, (1, 0)))

    return GlobalUpdatablePEPO(pepo, evolved, Ref(false))
end
function _initevolution(pepo::AbstractPEPO, map::GlobalStepper)
    function useful_f(x, t)
        D = domain(x)
        lη = domain(t)[1]
        rη = domain(t)[2]
        left = TensorMap(
            undef,
            promote_type(eltype(x), eltype(t)),
            codomain(x),
            fuse(D[1] * lη) * D[2] * fuse(D[3] * rη)' * D[4],
        )
        return left
    end

    evolved = useful_f.(getstate(pepo), map.t)

    return GlobalUpdatablePEPO(pepo, evolved, Ref(false))
end

# One full cycle of the trotter evolution. That is, t -> t + τ
function timestep!(pepo, env, map, order; kwargs...)
    trotterstep!(pepo, env, map; kwargs...)

    if order == 2
        trotterstep!(pepo, env, map; reverse=true, kwargs...)
    elseif order == 1
        changeaxis!(pepo[1], pepo[2])
    end

    return pepo
end

function trotterstep!(pepo, env, map; reverse::Bool=false, kwargs...)
    in = 1 + reverse
    out = 2 - reverse

    updatestep!(pepo[in], env[in], map[in]; kwargs...)

    # If possible, renormalize such that xenv -> yenv
    #axisrenormalize!(yenv, xenv, xpepo)
    changeaxis!(pepo[out], pepo[in]) # x -> y

    # Update along the y-directions
    updatestep!(pepo[out], env[out], map[out]; kwargs...)

    return pepo
end

# Parallel 
function updatestep!(pepo::GlobalUpdatablePEPO, truncators, map; truncalg, kwargs...)
    # Evolve, all sites
    pepo = evolve!(pepo, map)

    # # Post-env
    renormalize!(truncators)

    sequentialtruncate(pepo, truncators; truncalg)

    quicknormalize!(pepo)

    hermitianize!(pepo)

    return pepo
end

# Sequential, Requires atleast a 2x2 unit cell. I think?
function updatestep!(pepo::LocalUpdatablePEPO, truncators, map; reverse=false, kwargs...)
    # if size(getstate(pepo)) < (2, 2)
    #     throw(
    #         ArgumentError(
    #             "Sequential update only supported for lattice sizes of at least (2,2). Lattice sized provided: $(size(pepo))",
    #         ),
    #     )
    # end

    issymmetric = true

    bonds = getbonds(pepo)

    renormalize!(truncators)

    if reverse
        axisbonds = Base.reverse(bonds)
    else
        axisbonds = bonds
    end

    for col in eachcol(axisbonds)
        for bond in col

            # Update the states lying on the bond
            updatebond!(pepo, truncators, map, bond; kwargs...)

            # Quick normalization for algorithm stabilty. Does not ensure tr(ρ) = 1
            hermitianize!(pepo)

            quicknormalize!(pepo)

            if issymmetric
                tensors = getstate(pepo)

                for (c, col) in enumerate(eachcol(tensors))
                    if c == 1
                        continue
                    end
                    copy!.(tensors[:, c], circshift(tensors[:, 1], -(c - 1)))
                end
            end

            renormalize!(truncators)
        end

        if issymmetric
            break
        end
    end

    # renormalize!(truncators)
    return pepo, truncators
end

function updatebond!(
    pepo::LocalUpdatablePEPO, truncators, operator, bond; truncalg, kwargs...
)
    # Evolve the sites on the bond given by `bond` according to `operator`.
    evolve!(pepo, operator, bond)

    # Find the truncation isometries.
    isoms = isometries(pepo, truncators, bond, truncalg; kwargs...)

    # Truncate the pepo according to the truncation isometries
    truncate!(pepo, isoms)

    return pepo
end

function evolve!(pepo::LocalUpdatablePEPO, operator, bond)
    # Apply the trotter gate, updating the evolved tensors on `bond`
    setcurrentbond!(pepo, bond)
    applytrotter!(pepo, operator, bond)
    # Set the current bond of the pepo to `bond`
    setisevolved!(pepo, true)

    return pepo
end

function evolve!(pepo::GlobalUpdatablePEPO, operator)
    applybothtrotter!(pepo, operator)
    setisevolved!(pepo, true)
    return pepo
end

function applybothtrotter!(pepo::GlobalUpdatablePEPO, operator)
    setisevolved!(pepo, false)

    pin = getstate(pepo)
    pev = getevolved(pepo)

    applybothtrotter!(pev, pin, operator)

    return pepo
end
function applybothtrotter!(pev, pin, gates::TrotterGates)
    return _applybothtrotter!.(pev, pin, gates.east, gates.west)
end
function applybothtrotter!(pev, pin, gates::SimpleGates)
    return _applybothtrotter!.(pev, pin, gates.east, gates.west, gates.loc)
end
function applybothtrotter!(pev, pin, mpo::GlobalStepper)
    return _applybothtrotter!.(pev, pin, mpo.t)
end

function _applybothtrotter!(pev, pin, op_l, op_r)
    @tensoropt begin
        p_out[ook oob; e r s w l n] :=
            op_r[ok ob ik ib; l] * pin[ik ib; e s w n] * op_l[ook oob ok ob; r]
    end
    mergeinto!(pev, p_out)
    return pev
end
function _applybothtrotter!(pev, pin, op_l, op_r, loc)
    # west -> easts -> loc
    @tensoropt begin
        p_out[ook oob; e r s w l n] :=
            op_r[ok ob ik ib; l] * pin[ik ib; e s w n] * op_l[ook oob ok ob; r]
    end
    mergeinto!(pev, loc * p_out)
    return pev
end
function _applybothtrotter!(pev, pin, mpo)
    # west -> easts -> loc
    @tensoropt begin
        p_out[ok ob; e ee s w ww n] := mpo[ok ob ik ib; ee ww] * pin[ik ib; e s w n]
    end

    mergeinto!(pev, p_out)
    return pev
end

function applytrotter!(pepo::LocalUpdatablePEPO, operator, bond)
    # Get the current left and right pepo tensors
    setisevolved!(pepo, false)

    p_l, p_r = pepo[bond]
    op_l, op_r = operator[bond]

    # Get the evolved pepo tensors such that we can write into them
    pe_l, pe_r = evolvedpair(pepo)

    @tensoropt begin
        q_l[ok ob; e x s w n] := op_l[ok ob ik ib; x] * p_l[ik ib; e s w n]
        q_r[ok ob; e s w x n] := op_r[ok ob ik ib; x] * p_r[ik ib; e s w n]
    end

    mergeinto!(pe_l, q_l)
    mergeinto!(pe_r, q_r)

    return pepo
end
