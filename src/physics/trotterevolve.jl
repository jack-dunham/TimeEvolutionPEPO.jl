struct SequentialUpdate end
struct ParallelUpdate end

function initevolution(pepo::AbstractPEPO, map::DynamicalMap, updatescheme)
    xpepo = _initevolution(pepo, map.x, updatescheme)
    ypepo = _initevolution(changeaxis(pepo), map.y, updatescheme)
    return TimeDependantPEPO(xpepo, ypepo)
end

function _initevolution(pepo::AbstractPEPO, map, updatescheme::LULT)
    function useful_f(x, l, r)
        D = domain(x)
        lη = domain(l)
        rη = domain(r)
        left = TensorMap(
            undef,
            promote_type(eltype(x), eltype(l)),
            codomain(x),
            fuse(D[1] * lη) * D[2] * D[3] * D[4],
        )
        right = TensorMap(
            undef,
            promote_type(eltype(x), eltype(r)),
            codomain(x),
            D[1] * D[2] * fuse(D[3] * rη)' * D[4],
        )
        return left, right
    end

    evolved = useful_f.(getstate(pepo), map.east, map.west)

    first_bond = Ref(Bond(CartesianIndex(1, 1), CartesianIndex(2, 1)))

    return LocalUpdatablePEPO(pepo, evolved, first_bond, Ref(false))
end

function _initevolution(pepo::AbstractPEPO, map, updatescheme::GULT)
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
function _initevolution(pepo::AbstractPEPO, map::WIIMPO, updatescheme::GULT)
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

    evolved = useful_f.(getstate(pepo), map.data)

    return GlobalUpdatablePEPO(pepo, evolved, Ref(false))
end

function timestep!(
    pepo::TimeDependantPEPO,
    env::Environments,
    map::DynamicalMap, #{<:TrotterGates},
    order::Val{N};
    kwargs...,
) where {N}
    xpepo = pepo.x
    ypepo = pepo.y

    xenv = env.x
    yenv = env.y

    xmap = map.x
    ymap = map.y

    trotterstep!(xpepo, ypepo, xenv, yenv, xmap, ymap, order; kwargs...)

    # mutation of xpepo_td and ypepo_td should have mutated pepo accordingly.
    return pepo
end

# One full cycle of the trotter evolution. That is, t -> t + τ
function trotterstep!(xpepo, ypepo, xenv, yenv, xmap, ymap, ::Val{1}; kwargs...)
    updatestep!(xpepo, xenv, xmap; kwargs...)

    # If possible, renormalize such that xenv -> yenv
    #axisrenormalize!(yenv, xenv, xpepo)
    changeaxis!(ypepo, xpepo) # x -> y

    # Update along the y-directions
    updatestep!(ypepo, yenv, ymap; kwargs...)

    # Finally, write tensors back into xpepo
    changeaxis!(xpepo, ypepo) # y -> x

    return xpepo, ypepo
end
# Second order trotter decomposition
# exp(t/2 Lx) exp(t Ly) exp(t/2 Lx)
function trotterstep!(xpepo, ypepo, xenv, yenv, xmap, ymap, ::Val{2}; kwargs...)
    updatestep!(xpepo, xenv, xmap; kwargs...)

    #axisrenormalize!(yenv, xenv, xpepo)
    changeaxis!(ypepo, xpepo)

    updatestep!(ypepo, yenv, ymap; kwargs...)
    updatestep!(ypepo, yenv, ymap; reverse=true, kwargs...)

    #axisrenormalize!(xenv, yenv, ypepo)
    changeaxis!(xpepo, ypepo)

    updatestep!(xpepo, xenv, xmap; reverse=true, kwargs...)
    return xpepo, ypepo
end

# function updatestep!(pepo, environment, map; kwargs...)
#     # Now we either do parallelupdate or sequentialupdate
#     sequentialupdate!(pepo, environment, map; kwargs...)
#     return pepo, environment
# end
# function updatestep!(pepo, environment, map; kwargs...)
#     # Now we either do parallelupdate or sequentialupdate
#     parallelupdateupdate!(pepo, environment, map; kwargs...)
#     return pepo, environment
# end

# Parallel 
function updatestep!(pepo::GlobalUpdatablePEPO, environment, map; truncalg=NTU(), kwargs...)
    # Evolve, all sites
    pepo = evolve!(pepo, map)

    # # Post-env
    ITC.calculate!(environment)

    sequentialtruncate(pepo, environment; xi=truncalg.ξ, gauge=truncalg.gauge)

    normalize!.(getstate(pepo))
    # Truncate each site with respect the isometries
    # truncate!(pepo, iso)

    return pepo
end

# Sequential, Requires atleast a 2x2 unit cell. I think?
function updatestep!(pepo::LocalUpdatablePEPO, environment, map; reverse=false, kwargs...)
    # if size(getstate(pepo)) < (2, 2)
    #     throw(
    #         ArgumentError(
    #             "Sequential update only supported for lattice sizes of at least (2,2). Lattice sized provided: $(size(pepo))",
    #         ),
    #     )
    # end

    bonds = getbonds(pepo)

    if reverse
        axisbonds = Base.reverse(bonds)
    else
        axisbonds = bonds
    end

    for col in eachcol(axisbonds)
        for (i, bond) in enumerate(col)
            # Update the states lying on the bond
            updatebond!(pepo, environment, map, bond; kwargs...)

            # Quick and dirty normalization for algorithm stabilty. Does not ensure tr(ρ) = 1
            normalize!.(getstate(pepo))

            # renormalize(hilbertschmidt(pepo), environment)
            #renormalize!(environment, pepo; renorm=renorm)
            if true #issymmetric(map.east)
                println("hello?")
                for (j, _) in enumerate(eachcol(axisbonds)[2:end])
                    new_left, new_right = pepo[bond]
                    l = Lattice(axisbonds)
                    old_left, old_right = pepo[l[i - j, j + 1]]
                    copy!(old_left, new_left)
                    copy!(old_right, new_right)
                end
            end
            environment!(environment)
        end
        if true#issymmetric(map.east)
            break
        end
    end
    return pepo, environment
end

function updatebond!(pepo::LocalUpdatablePEPO, environment, operator, bond; kwargs...)
    # Evolve the sites on the bond given by `bond` according to `operator`.
    evolve!(pepo, operator, bond)

    # Find the truncation isometries.
    truncators = isometries(pepo, environment, bond; kwargs...)

    # Truncate the pepo according to the truncation isometries
    truncate!(pepo, truncators)

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
function applybothtrotter!(pev, pin, mpo::WIIMPO)
    return _applybothtrotter!.(pev, pin, mpo.data)
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
