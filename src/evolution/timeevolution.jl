# This file defines the top level `timestep` function

newevolution(pepo, steps, method) = newevolution(eltype(steps), pepo, steps, method)
newevolution(::Type{<:AxialStepper}, pepo, steps, method) = AxialPEPO(pepo)
newevolution(::Type{<:AbstractTimeStepper}, pepo, steps, method) = pepo
function newevolution(::Type{<:LocalStepper}, pepo, steps, method::TEBD{<:Union{FET,NTU}})
    return AxialPEPO(pepo)
end

# Default
"""
    $(TYPEDSIGNATURES)

A single time step of the simulation.
"""
function timestep!(pepo, truncator, stepper, truncalg; kwargs...)
    # Its a bit hacky
    for (i, step) in enumerate(stepper)
        if eltype(truncator) <: Vector
            trunc = getindex.(truncator, i)
        else
            trunc = truncator
        end

        trotterstep!(pepo, truncator, step, truncalg)
    end
    return pepo
end

function _timestep!(
    pepo::PEPO, truncator, stepper::LocalStepper, truncalg::SimpleUpdate; kwargs...
)
    size(pepo) < (2, 2) && throw(ArgumentError(""))

    axis = stepper.axis

    for bond in bonds(pepo, axis)
        env = environment!(pepo, SimpleUpdateState(), bond)

        pair = evolve(pepo, stepper, bond)

        newpair, newweight = updatetensors(pair, env, truncalg, bond)

        pepo[bond] = newpair

        setweight!(pepo, newweight, bond; copy=false)

        # Normalizes each tensor using `norm`; not the same as normalizing the quantum
        # state. This is for algorithm stability reasons only
        quicknormalize!(pepo)
        hermitianize!(pepo)
    end

    return pepo
end

function trottertep!(
    pepo!::PEPO, truncator, stepper::FullStepper, truncalg::SimpleUpdate; kwargs...
)
    isoms = newisometries(pepo!, stepper)

    newpepo = evolve(pepo!, stepper)

    simple_update_gauge_parallel!(newpepo)

    truncate_bond_matrices!(pepo!, newpepo, isoms)

    quicknormalize!(pepo!)
    hermitianize!(pepo!)

    return pepo!
end

# function timestep!(
#     pepo!::AxialPEPO, truncator, stepper::AxialStepper, truncalg::SimpleUpdate; kwargs...
# )
#     axis = stepper.axis
#
#     println(axis)
#
#     # Evolve and then truncate each bond in sequence
#     if pepo!.currentaxis != axis
#         setaxis!(pepo!, stepper.axis)
#     end
#
#     isoms = newisometries(tensors(pepo!), stepper)
#
#     newpepo = evolve(pepo!, stepper)
#
#     simple_update_gauge_parallel!(newpepo)
#
#     truncate_bond_matrices!(pepo!, newpepo, isoms)
#
#     quicknormalize!(pepo!)
#     hermitianize!(pepo!)
#
#     return pepo!
# end

function trotterstep!(
    pepo::AxialPEPO, truncator, stepper::LocalStepper, truncalg::Union{NTU,FET}; kwargs...
)
    axis = stepper.axis

    # Evolve and then truncate each bond in sequence
    if pepo.currentaxis != axis
        setaxis!(pepo, axis)
    end

    infidelity = 0

    allbonds = bonds(pepo, 1)

    for bond in allbonds
        D = bondspace(pepo, bond)

        # Evolved the pair of tensors connected by `bond`, leaving the original pair
        # unmutated.
        pair = evolvemerge(pepo, stepper, bond)

        if false #truncalg.rankreduce
            init_weights = environment!(pepo.parent, SimpleUpdateState(), bond)
            if axis == 2
                init_weights = permutedims(init_weights)
            end

            pair, initiso = updatetensors(pair, init_weights, SimpleUpdate(), bond)

            env = environment!(pepo, truncator[axis], bond)
            (u, v), s, _ = metric_isometries_from_simple_update!(
                pair, initiso, env, bond; D=D, optalg=truncalg.optalg, zeta=truncalg.zeta
            )

        else
            env = environment!(pepo, truncator[axis], bond)
            (u, v), s, _ = isometries(pair, env, bond, truncalg; D=D)
        end

        oldA, oldB = pepo[bond]

        a1, a2, a3, a4 = domain(oldA)
        b1, b2, b3, b4 = domain(oldB)

        Du = domain(u)
        Dv = codomain(v)

        if bondaxis(bond) == 1
            domA = Du * a2 * a3 * a4
            domB = b1 * b2 * Dv' * b4
        elseif bondaxis(bond) == 2
            domA = a1 * Du * a3 * a4
            domB = b1 * b2 * b3 * Dv'
        end

        newA = similar(oldA, codomain(oldA), domA)
        newB = similar(oldB, codomain(oldB), domB)

        newpair = newA => newB

        truncateto!(newpair, pair, u => v; axis=bondaxis(bond))

        pepo[bond] = newpair

        setweight!(pepo, s, bond; copy=false)
    end


    quicknormalize!(pepo)
    hermitianize!(pepo)
    # if axis == 1
    #     simple_update_gauge!(pepo.parent; reset = true)
    # end

    return pepo, infidelity
end

function trotterstep!(
    pepo::AxialPEPO, truncator, stepper::AxialStepper, truncalg::SimpleUpdate; kwargs...
)
    axis = stepper.axis

    # Evolve and then truncate each bond in sequence
    if pepo.currentaxis != axis
        setaxis!(pepo, axis)
    end

    infidelity = similar(TRGroups.getdata(pepo))

    inweights = axis == 1 ? deepcopy(bondweights(pepo)) : permutedims(bondweights(pepo))

    evpepo = evolve(pepo, stepper)

    axial = false

    if !axial
        if 1 in size(evpepo)
            simple_update_gauge_parallel!(evpepo; maxiter=500, tol=1e-10)
        else
            simple_update_gauge!(evpepo)
        end
        inweights = bondweights(evpepo)
    else
        inweights = bondweights(pepo)
    end

    isoms = map(bonds(pepo, 1)) do bond
        D = bondspace(pepo, bond)

        # Evolved the pair of tensors connected by `bond`, leaving the original pair
        # unmutated.

        pair = evpepo[bond]

        iso, s, _ = isometries(pair, inweights, bond, truncalg; D=D, axial=axial)

        setweight!(pepo, s, bond; copy=false)

        return iso
    end

    isoe = broadcast(first, isoms)
    isow = broadcast(last, circshift(isoms, (1, 0)))

    broadcast(pepo, evpepo, isoe, isow) do dst, src, e, w
        truncateto!(dst, src, e, w)
    end

    quicknormalize!(pepo)
    hermitianize!(pepo)

    return pepo, infidelity
end

function trotterstep!(
    pepo::AxialPEPO, truncator, stepper::AxialStepper, truncalg::Union{FET,NTU}; kwargs...
)
    axis = stepper.axis

    if pepo.currentaxis != axis
        setaxis!(pepo, stepper.axis)
    end

    infidelity = similar(pepo)

    evpepo = evolve(pepo, stepper)

    T = tensormaptype(spacetype(pepo), 1, 1, scalartype(pepo))

    isomu = similar(getaxis(pepo, axis), T)
    isomv = similar(getaxis(pepo, axis), T)

    # This can happen in parallel
    Threads.@threads for bond in bonds(pepo, 1)
        # isoms = map(bonds(pepo, 1)) do bond
        D = bondspace(pepo, bond)

        env = environment!(evpepo, truncator[axis], bond)

        pair = evpepo[bond]

        (u, v), s, _ = isometries(pair, env, bond, truncalg; D=D, parallel=true)

        println(s)

        setweight!(pepo, s, bond; copy=false)

        isomu[first(bond)] = u
        isomv[last(bond)] = v

        # return u => v
    end

    # isomu = broadcast(first, isoms)
    # isomv = broadcast(last, circshift(isoms,(1,0)))

    broadcast(pepo, evpepo, isomu, isomv) do dst, src, u, v
        truncateto!(dst, src, u, v)
    end

    quicknormalize!(pepo)
    hermitianize!(pepo)

    return pepo, infidelity
end
function trotterstep!(pepo!::PEPO, truncator!::Isometries, stepper, truncalg::IterativeSU)
    iterative_simple_update_prototype!(
        pepo!,
        truncator!,
        stepper;
        maxiter=truncalg.maxiter,
        tol=truncalg.tol,
        recycle=truncalg.recycle,
        sequential=truncalg.sequential,
        gauge=truncalg.gauge,
        zeta=truncalg.zeta,
    )

    return pepo!
end
function trotterstep!(
    pepo!::PEPO, truncator!::Isometries, stepper, truncalg::IterativeSUTest
)
    iterative_simple_update_prototype!(
        pepo!,
        truncator!,
        stepper;
        maxiter=truncalg.maxiter,
        tol=truncalg.tol,
        recycle=truncalg.recycle,
        sequential=truncalg.sequential,
        gauge=truncalg.gauge,
        zeta=truncalg.zeta,
        check_fidelity=true,
    )

    return pepo!
end
