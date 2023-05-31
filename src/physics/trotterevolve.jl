struct SequentialUpdate end
struct ParallelUpdate end

function initevolution(
    pepo::AbstractPEPO, map, updatescheme::SequentialUpdate=SequentialUpdate()
)
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

    evolved = useful_f.(getstate(pepo), map.left, map.right)

    first_bond = Ref(Bond(CartesianIndex(1, 1), CartesianIndex(2, 1)))

    return LocallyEvolvingPEPO(pepo, evolved, first_bond, Ref(false))
end
function initevolution(
    pepo::AbstractPEPO, map, updatescheme::ParallelUpdate=ParallelUpdate()
)
    function useful_f(x, l, r)
        D = domain(x)
        lη = domain(l)
        rη = domain(r)
        left = TensorMap(
            undef,
            promote_type(eltype(x), eltype(l)),
            codomain(x),
            fuse(D[1] * lη) * D[2] * fuse(D[3] * rη) * D[4],
        )
        return left, right
    end

    evolved = useful_f.(getstate(pepo), map.left, circshift(map.right, (1, 0)))

    return GloballyEvolvingPEPO(pepo, evolved, Ref(false))
end

function trotterevolution!(
    pepo::PEPO,
    xmodel,
    ymodel;
    # alg::AbstractBoundaryAlgorithm=VUMPS(;
    #     bonddim=5, verbosity=true, tol=1e-12, maxiter=200
    # ),
    alg=NeighbourhoodAlgorithm(),
    updatescheme::SequentialUpdate=SequentialUpdate(),
    numsteps=50,
    timestep=0.1,
    order::Int=2,
    obsalg::AbstractBoundaryAlgorithm=VUMPS(; bonddim=5),
    renorm::Symbol=:full,
)
    L = lattice(pepo)

    tau = timestep / order

    xpepo = pepo
    ypepo = changeaxis(pepo)

    # Get the trotter gates from the liouvillian
    xmap = trottergates(L, xmodel, tau)
    ymap = trottergates(L, ymodel, tau)

    # Initialise pre-environments to iterate on each pepo.
    xenv = initenvironment(HilbertSchmidt(xpepo), alg)
    yenv = initenvironment(HilbertSchmidt(ypepo), alg)

    # Enviroment for the obseverables
    obsenv = initenvironment(Trace(pepo), obsalg)

    # Calculate pre enviroments before first step
    calculate!(xenv)
    calculate!(yenv)

    # Make pepo time-dependant
    xpepo_td = initevolution(xpepo, xmap, updatescheme)
    ypepo_td = initevolution(ypepo, ymap, updatescheme)

    obs = Vector{Matrix{Float64}}()

    # Perform the trotter iteration.
    for step in 1:numsteps
        println("Iteration: $step")
        trotterstep!(xpepo_td, ypepo_td, xenv, yenv, xmap, ymap, Val(order))

        # Only if not doing full renormalization
        if !(renorm == :full)
            calculate!(xenv)
            calculate!(yenv)
        end

        calculate!(obsenv)

        oneloc = onelocal(pepo, obsenv; normalize=true)

        magn = measure(oneloc, σ1)

        push!(obs, (real.(magn)))
    end

    # mutation of xpepo_td and ypepo_td should have mutated pepo accordingly.
    return obs
end

# One full cycle of the trotter evolution. That is, t -> t + τ
function trotterstep!(xpepo, ypepo, xenv, yenv, xmap, ymap, ::Val{1})
    updatestep!(xpepo, xenv, xmap)

    # If possible, renormalize such that xenv -> yenv
    #axisrenormalize!(yenv, xenv, xpepo)
    changeaxis!(ypepo, xpepo) # x -> y

    # Update along the y-directions
    updatestep!(ypepo, yenv, ymap)

    # Finally, write tensors back into xpepo
    changeaxis!(xpepo, ypepo) # y -> x

    return xpepo, ypepo
end
# Second order trotter decomposition
# exp(t/2 Lx) exp(t Ly) exp(t/2 Lx)
function trotterstep!(xpepo, ypepo, xenv, yenv, xmap, ymap, ::Val{2})
    updatestep!(xpepo, xenv, xmap)

    #axisrenormalize!(yenv, xenv, xpepo)
    changeaxis!(ypepo, xpepo)

    updatestep!(ypepo, yenv, ymap)
    updatestep!(ypepo, yenv, ymap; reverse=true)

    #axisrenormalize!(xenv, yenv, ypepo)
    changeaxis!(xpepo, ypepo)

    updatestep!(xpepo, xenv, xmap; reverse=true)
    return xpepo, ypepo
end

function updatestep!(pepo, environment, map; kwargs...)
    # Now we either do parallelupdate or sequentialupdate
    sequentialupdate!(pepo, environment, map; kwargs...)
    return pepo, environment
end
# function updatestep!(pepo, environment, map; kwargs...)
#     # Now we either do parallelupdate or sequentialupdate
#     parallelupdateupdate!(pepo, environment, map; kwargs...)
#     return pepo, environment
# end

function parallelupdate!(pepo::GloballyEvolvingPEPO, operator, environment)
    # Evolve, all sites
    pepo = evolve!(pepo, operator)

    calculate!(environment)

    # Get isometries for each bond
    iso = optimise(pepo, environment)

    # Truncate each site with respect the isometries
    truncate!(pepo, iso)

    return pepo
end

# Requires atleast a 2x2 unit cell. I think?
function sequentialupdate!(
    pepo::LocallyEvolvingPEPO, environment, map; reverse=false, renorm=:full
)
    # if size(getstate(pepo)) < (2, 2)
    #     throw(
    #         ArgumentError(
    #             "Sequential update only supported for lattice sizes of at least (2,2). Lattice sized provided: $(size(pepo))",
    #         ),
    #     )
    # end

    bonds = getbonds(lattice(pepo))

    if reverse
        axisbonds = Base.reverse(bonds)
    else
        axisbonds = bonds
    end

    for col in eachcol(axisbonds)
        for bond in col
            # Update the states lying on the bond
            updatebond!(pepo, environment, map, bond)

            # Quick and dirty normalization for algorithm stabilty. Does not ensure tr(ρ) = 1
            normalize!.(getstate(pepo))

            calculate!(environment)

            #renormalize!(environment, pepo; renorm=renorm)
        end
        # if issymmetric(lattice(pepo))
        #     # update remaining cols
        #     break
        # end
    end
    return pepo, environment
end

function updatebond!(pepo::LocallyEvolvingPEPO, environment, operator, bond; kwargs...)
    # Evolve the sites on the bond given by `bond` according to `operator`.
    evolve!(pepo, operator, bond)

    # Find the optimial truncation isometries.
    isometries = optimise(pepo, environment, bond; kwargs...)

    # Truncate the pepo according to iso
    truncate!(pepo, isometries)

    return pepo
end

# Vectorised 
function optimise(pepo_evolved, environment; kwargs...)
    axisbonds = getbonds(pepo_evolved)[1]

    iso = optimise.(pepo_evolved, environment, axisbonds; kwargs...)
    return iso
end

# Scalar
function optimise(pepo, environment, bond::Bond; xi=Inf, gauge=1, alg=TensorOptimisation())
    # Get the isometries and the truncation metric generated by `alg`.
    # @info "Optimising along: $bond"
    iso, metric = alg(pepo, environment.tensors, bond)
    wtg!(iso, metric, Val(gauge))
    if xi < Inf
        copy!(iso.s, rescale(iso.s, xi))
        wtg!(iso, metric, Val(gauge))
    end
    return iso
end

function evolve!(pepo::LocallyEvolvingPEPO, operator, bond)
    # Apply the trotter gate, updating the evolved tensors on `bond`
    setcurrentbond!(pepo, bond)
    applytrotter!(pepo, operator, bond)
    # Set the current bond of the pepo to `bond`
    setisevolved!(pepo, true)

    return pepo
end

function evolve!(pepo::GloballyEvolvingPEPO, operator)
    applybothtrotter!(pepo, operator)
    return setisevolved!(pepo, true)
end

function applybothtrotter!(pepo::GloballyEvolvingPEPO, operator)
    setisevolved!(pepo, false)

    pin = getstate(pepo)
    pev = getevolved(pepo)
    @tensoropt begin
        p_out[ok ob; e r s w l n] :=
            op_r[ok ob ik ib; l] * pin[ik ib; e s w n] * op_l[ook oob ok ob; r]
    end
    mergeinto!(pev, pin)
    return pepo
end

function applytrotter!(pepo::LocallyEvolvingPEPO, operator, bond)
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
