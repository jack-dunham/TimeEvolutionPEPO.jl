# function timestep!(pepo, truncator::SimpleUpdateState, stepper, ::Sequential, ::Sequential; kwargs...)
#     sequential_update_step!(pepo, truncator, stepper; kwargs...)
# end

newtruncators(::AxialPEPO, _, ::SU) = (SimpleUpdateState(), SimpleUpdateState())

function axial_trotter_step!(pepo, truncator, stepper; truncalg, kwargs...)
    axis = bondaxis(stepper)

    axistrunc = truncator[axis]

    axisbuffer = currentbuffer(pepo)

    if !ispostenv(pepo)
        renormalize!(truncator, currentbuffer(pepo))
    end

    sequential_update_step!(pepo, axistrunc, stepper; truncalg=truncalg, kwargs...)

    # This does nothing if `stepstyle isa Sequential` as truncation has already occured 
    # for all bonds
    truncate!(pepo)

    return pepo
end

function timestep!(pepo::PEPO, truncator, stepper::LocalStepper, truncalg::SU; kwargs...)

    # Evolve and then truncate each bond in sequence
    axis = stepper.axis

    buffer = currentbuffer(pepo)

    size(buffer) < (2,2) && throw(ArgumentError(""))

    infidelity = similar(buffer)

    axisbonds = bonds(pepo, stepper.axis)

    isreversed(stepper) && (axisbonds = Iterators.Reverse(axisbonds))

    checker = true

    if checker 
        altbonds = circshift(axisbonds[:, 2], 1)
        axisbonds = axisbonds[:, 1]
    end

    for (i, bond) in enumerate(axisbonds)


        # println(bond)
        # println("alt:", altbonds[i])

        # Evolved the pair of tensors connected by `bond`, leaving the original pair
        # unmutated.
        pair = evolve(pepo, stepper, bond)

        # (u, v), s, _ = isometries!(pair, pepo, bond, truncalg)
        absorb_adjacent_weights!(pair, weights(pepo), bond;inverse=false)
        newpair, s = isometries(pair; axis = bondaxis(bond), xi = truncalg.ξ)
        absorb_adjacent_weights!(newpair, weights(pepo), bond; inverse=true)

        # newpair = mapreduce(copy, =>, pepo[bond])

        # oldA, oldB = pepo[bond]
        #
        # a1, a2, a3, a4 = domain(oldA)
        # b1, b2, b3, b4 = domain(oldB)
        #
        # Du = domain(u)
        # Dv = codomain(v)
        #
        # if bondaxis(bond) == 1
        #     domA = Du * a2 * a3 * a4
        #     domB = b1 * b2 * Dv' * b4
        # elseif bondaxis(bond) == 2
        #     domA = a1 * Du * a3 * a4
        #     domB = b1 * b2 * b3 * Dv'
        # end
        #
        # newA = similar(oldA, codomain(oldA), domA)
        # newB = similar(oldB, codomain(oldB), domB)
        #
        # newpair = newA => newB
        #
        # truncateto!(newpair, pair, u => v; axis=bondaxis(bond))

        pepo.state[bond] = newpair

        if checker

            altbond = altbonds[i]

            pepo.state[altbond] = mapreduce(copy, =>, newpair)

            setweight!(pepo, s, altbond; copy=false)
        end

        # copy!(oldA, newA)
        # copy!(oldB, newB)

        setweight!(pepo, s, bond; copy=false)

        quicknormalize!(pepo)
        hermitianize!(pepo)
    end

    # simple_update_gauge!(pepo; reset=true, verbose=false)
#
    return pepo, infidelity
end

function timestep!(
    pepo::AxialPEPO, truncator, stepper::LocalStepper, truncalg::NTU; kwargs...
)
    axis = stepper.axis

    # Evolve and then truncate each bond in sequence
    if pepo.currentaxis != axis
        setaxis!(pepo, stepper.axis)
    end

    buffer = tensors(pepo)

    infidelity = 0


    axisbonds = bonds(buffer, 1)

    checker = true

    if checker 
        altbonds = circshift(axisbonds[:, 2], 1)
        axisbonds = axisbonds[:, 1]
    end

    for (i, bond) in enumerate(axisbonds)

        D = bondspace(buffer, bond)

        # Evolved the pair of tensors connected by `bond`, leaving the original pair
        # unmutated.
        pair = evolvemerge(buffer, stepper, bond)
        trunc = pair -> ntu_total(pair, buffer, bond)

        (u, v), s, _ = isometries(pair, trunc, bond, truncalg; D = D)

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

        buffer[bond] = newpair

        if checker

            altbond = altbonds[i]

            buffer[altbond] = mapreduce(copy, =>, newpair)
            realaltbond = axis == 1 ? altbond : transpose(altbond)

            setweight!(pepo, s, realaltbond; copy=false)
        end

        realbond = axis == 1 ? bond : transpose(bond)

        setweight!(pepo, s, realbond; copy=false)

        normalize!.(buffer)
        hermitianize!.(buffer)
    end

    # if axis == 1
    #     simple_update_gauge!(pepo.parent; reset = true)
    # end

    return pepo, infidelity
end

function timestep!(
    pepo::AxialPEPO, truncator, stepper::LocalStepper, truncalg::FET; kwargs...
)
    axis = stepper.axis

    # Evolve and then truncate each bond in sequence
    if pepo.currentaxis != axis
        setaxis!(pepo, stepper.axis)
    end

    buffer = tensors(pepo)

    infidelity = 0

    truncalg = FET(; alg=VUMPS(bonddim=16, tol=1e-8), ξ=Inf, rankreduce=false, optalg=EVB())


    bonditr = bonds(buffer, 1)

    for bond in bonditr

        D = bondspace(buffer, bond)

        # Evolved the pair of tensors connected by `bond`, leaving the original pair
        # unmutated.
        pair = evolve(buffer, stepper, bond)

        trunc = newtruncators(buffer, truncalg)
        renormalize!(trunc)

        (u, v), s, _ = isometries(pair, trunc, bond, truncalg; D = D)

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

        buffer[bond] = newpair

        realbond = axis == 1 ? bond : transpose(bond)

        setweight!(pepo, s, realbond; copy=false)

        normalize!.(buffer)
        hermitianize!.(buffer)
    end

    return pepo, infidelity
end

function timestep!(
    pepo::AxialPEPO, truncator, stepper::AxialStepper, truncalg::SU; kwargs...
)

    # Evolve and then truncate each bond in sequence
    if pepo.currentaxis != stepper.axis
        setaxis!(pepo, stepper.axis)
    end
    axis = stepper.axis

    buffer = tensors(pepo)
    infidelity = similar(buffer)

    outweights = deepcopy(weights(pepo))

    inweights = axis == 1 ? weights(pepo) : permutedims(weights(pepo))

    evolvedtensors = evolve(pepo, stepper)

    isoms = map(bonds(buffer, 1)) do bond
        D = bondspace(buffer, bond)

        # Evolved the pair of tensors connected by `bond`, leaving the original pair
        # unmutated.

        pair = evolvedtensors[bond]

        iso, s, _ = isometries(pair, inweights, bond, truncalg; D=D, axial=true)

        realbond = axis == 1 ? bond : transpose(bond)
        outweights[realbond] = s

        return iso
    end

    tolerance!(weights(pepo), outweights, 1.0)

    isoe = broadcast(first, isoms)
    isow = broadcast(last, circshift(isoms, (1, 0)))

    broadcast(buffer, evolvedtensors, isoe, isow) do dst, src, e, w
        truncateto!(dst, src, e, w)
    end

    quicknormalize!(pepo.parent)
    hermitianize!(pepo.parent)

    return pepo, infidelity
end



# isometries for sequential simple update
function isometries!(pair::Pair, pepo::AbstractPEPO, bond::Bond, truncalg::SU)
    D = bondspace(pepo, bond)

    trunc = truncspace(D)

    xi = truncalg.ξ

    iso, s, infid = simpleupdate!(pair, weights(pepo), bond; trunc=trunc, xi=xi)

    return iso, s, infid
end

function isometries(pair::Pair, pepo::AbstractPEPO, bond::Bond, truncalg::SU)
    return isometries(pair, weights(pepo), bond, truncalg; D=bondspace(pepo, bond))
end
function isometries(
    pair::Pair, weights::EdgeLattice, bond::Bond, truncalg::SU; D, axial=false
)
    trunc = truncspace(D)

    xi = truncalg.ξ

    iso, s, infid = simpleupdate(pair, weights, bond; axial=axial, trunc=trunc, xi=xi)

    return iso, s, infid
end

# For updating in parellel
function parallelisometries(pepo::AbstractPEPO, bond::Bond, truncalg::SU; kwargs...)
    a, b = pepo[bond]

    pair = copy(a) => copy(b)

    return isometries!(pair, pepo, bond, truncalg)
end

function parallelisometries(pair::Pair, pepo::AbstractPEPO, bond::Bond, truncalg::SU)
    return isometries(pair, pepo, bond, truncalg)
end
# Sequential simple update
function simpleupdate!(pair, weights, bond; trunc, absorb=true, axial=false, kwargs...)
    axis = bondaxis(bond)

    absorb && absorb_adjacent_weights!(pair, weights, bond; inverse=false, axial=axial)

    iso, s, infid = simple_update_svd!(pair; trunc=trunc, absorb=true, axis=axis, kwargs...)

    absorb && absorb_adjacent_weights!(pair, weights, bond; inverse=true, axial=axial)

    return iso, s, infid
end

function simpleupdate(pair, weights, bond; trunc, axial=false, absorb=true, kwargs...)
    axis = bondaxis(bond)

    newpair = mapreduce(copy, =>, pair)

    absorb && absorb_adjacent_weights!(newpair, weights, bond; axial=axial, inverse=false)

    iso, s, infid = simple_update_svd(
        newpair; trunc=trunc, absorb=true, axis=axis, kwargs...
    )

    return iso, s, infid
end

function simple_update_svd(pair; axis, kwargs...)
    (P, Q), (u0, v0) = rankreduce(pair; axis=axis)

    (u, v), s, infid = isometries_from_svd(P, Q, u0, v0;  axis = axis, kwargs...)

    return pinv(u0) * u => v * pinv(v0), s, infid
end

function simple_update_svd!(pair; axis, kwargs...)
    (P, Q), (u0, v0) = rankreduce!(pair; axis=axis)

    iso, s, infid = isometries_from_svd(P, Q, u0, v0; axis = axis, kwargs...)

    return iso, s, infid
end

function isometries_from_svd(P, Q, u0, v0; xi=Inf, trunc=notrunc(), absorb=true, axis)
    u, s, v, err2 = safetsvd!(normalize(u0 * v0); trunc=trunc)

    maxnormalize!(s)

    if absorb
        u = u * sqrt(s)
        v = sqrt(s) * v
        f = localfidelity(u, v, P, Q, u0, v0; axis = axis)
    else
        # f = fidelity(u * sqrt(s), sqrt(s) * v, P, Q, u0, v0)
    end
    f = 0

    iso = u => v

    xi < Inf && rescale!(PEPOKit.TruncationIsometries(u,s,v), xi; verbose=false)

    return iso, s, 1 - f
end

function hori_weights(uten::T, vten::T, U, V; axial) where {T<:AbsTen{2, 4}}
    s1,w1,n1 = U
    e2,s2,n2 = V

    if axial
        @tensoropt begin
            unew[k b; e ss w nn] := uten[k b; e s w n] * s1[s; ss] * n1[nn; n]
            vnew[k b; e ss w nn] := vten[k b; e s w n] * s2[s; ss] * n2[nn; n]
        end
    else
        @tensoropt begin
            unew[k b; e ss ww nn] :=
                uten[k b; e s w n] * s1[s; ss] * w1[ww; w] * n1[nn; n]
            vnew[k b; ee ss w nn] :=
                vten[k b; e s w n] * s2[s; ss] * e2[e; ee] * n2[nn; n]
        end
    end
    return unew, vnew
end
function hori_weights(uten::T, vten::T, U, V; axial) where {T<:AbsTen{2, 5}}
    s1,w1,n1 = U
    e2,s2,n2 = V

    if axial
        @tensoropt begin
            unew[k b; e1 e2 ss w1 w2 nn] := uten[k b; e1 e2 s w1 w2 n] * s1[s; ss] * n1[nn; n]
            vnew[k b; e1 e2 ss w1 w2 nn] := vten[k b; e1 e2 s w1 w2 n] * s2[s; ss] * n2[nn; n]
        end
    else
        @tensoropt begin
            unew[k b; e1 e2 ss ww nn] :=
                uten[k b; e1 e2 s w n] * s1[s; ss] * w1[ww; w] * n1[nn; n]
            vnew[k b; ee ss w1 w2 nn] :=
                vten[k b; e s w1 w2 n] * s2[s; ss] * e2[e; ee] * n2[nn; n]
        end
    end
    return unew, vnew
end

function vert_weights(uten::T, vten::T, U, V; axial) where {T<:AbsTen{2, 4}}
    e1, w1, n1 = U
    e2, s2, w2 = V

    if axial
        @tensoropt begin
            unew[k b; ee s ww n] := uten[k b; e s w n] * e1[e; ee] * w1[ww; w]
            vnew[k b; ee s ww n] := vten[k b; e s w n] * e2[e; ee] * w2[ww; w]
        end
    else
        @tensoropt begin
            unew[k b; ee s ww nn] :=
                uten[k b; e s w n] * e1[e; ee] * w1[ww; w] * n1[nn; n]
            vnew[k b; ee ss ww n] :=
                vten[k b; e s w n] * s2[s; ss] * e2[e; ee] * w2[ww; w]
        end
    end

    return unew, vnew
end

function vert_weights(uten::T, vten::T, U, V; axial) where {T<:AbsTen{2, 5}}
    e1, w1, n1 = U
    e2, s2, w2 = V

    if axial
        @tensoropt begin
            unew[k b; ee s1 s2 ww n1 n2] := uten[k b; e s1 s2 w n1 n2] * e1[e; ee] * w1[ww; w]
            vnew[k b; ee s1 s2 ww n1 n2] := vten[k b; e s1 s2 w n1 n2] * e2[e; ee] * w2[ww; w]
        end
    else
        @tensoropt begin
            unew[k b; ee s1 s2 ww nn] :=
                uten[k b; e s1 s2 w n] * e1[e; ee] * w1[ww; w] * n1[nn; n]
            vnew[k b; ee ss ww n1 n2] :=
                vten[k b; e s w n1 n2] * s2[s; ss] * e2[e; ee] * w2[ww; w]
        end
    end
    return unew, vnew
end


function absorb_adjacent_weights!(
    pair::Pair{T}, weights::EdgeLattice, bond::Bond; inverse=false, axial=false
) where {T<:AbsTen{2}}
    uten, vten = pair

    if inverse
        func = w -> try
            inv(sqrt(w))
        catch
            pinv(sqrt(w))
        end
    else
        func = w -> sqrt(w)
    end

    U, V = mapadjacent(func, weights, bond)

    axis = bondaxis(bond)

    if axis == 1
        unew, vnew = hori_weights(uten, vten, U, V; axial = axial)
    elseif axis == 2
        unew, vnew = vert_weights(uten, vten, U, V; axial = axial)
    end

    copy!(uten, unew)
    copy!(vten, vnew)

    return uten => vten
end

function absorb_axial_weights!(
    site::AbsTen{2,4},
    weights::EdgeLattice{G},
    ind::CartesianIndex{2};
    inverse=false,
    axis=1,
) where {G}
    if inverse
        func = w -> try
            inv(sqrt(w))
        catch
            # @warn "inv failed"
            pinv(sqrt(w))
        end
    else
        func = w -> sqrt(w)
    end

    e, s, w, n = func.(bondswith(G, ind))

    if axis == 1
        @tensoropt begin
            site[k b; e ss w nn] = copy(site)[k b; e s w n] * s[s; ss] * n[nn; n]
        end

    elseif axis == 2
        @tensoropt begin
            site[k b; ee s ww n] = copy(site)[k b; e s w n] * e[e; ee] * w[ww; w]
        end
    end

    return site
end


