newtruncators(::AxialPEPO, _, ::SimpleUpdate) = (SimpleUpdateState(), SimpleUpdateState())
newtruncators(::AxialPEPO, _, ::SVDU) = (SimpleUpdateState(), SimpleUpdateState())

function timestep!(
    pepo::PEPO, truncator, stepper::LocalStepper, truncalg::Union{SimpleUpdate,SVDU}; kwargs...
)

    # Evolve and then truncate each bond in sequence
    size(pepo) < (2, 2) && throw(ArgumentError(""))

    infidelity = similar(ITC.getdata(pepo))

    allbonds = bonds(pepo, stepper.axis)

    for bond in allbonds
        env = environment!(pepo, SimpleUpdateState(), bond)

        # println("alt:", altbonds[i])

        # Evolved the pair of tensors connected by `bond`, leaving the original pair
        # unmutated.

        if truncalg.reduce === :post
            pair = evolve(pepo, stepper, bond)
            newpair, newweight = updatetensors(pair, env, truncalg, bond)
        elseif truncalg.reduce === :pre
            newpair, newweight = evolve_updatetensors(pepo, env, stepper, truncalg, bond)
        end

        pepo[bond] = mapreduce(copy, =>, newpair)
        setweight!(pepo, newweight, bond; copy=false)
    end

    # simple_update_gauge!(pepo; reset=false, verbose=true)
    # quicknormalize!(pepo)
    hermitianize!(pepo)
    #
    return pepo, infidelity
end

function _timestep!(
    pepo::PEPO{Square}, truncator, stepper::LocalStepper, truncalg::Union{SimpleUpdate,SVDU}; kwargs...
)

    # Evolve and then truncate each bond in sequence
    size(pepo) < (2, 2) && throw(ArgumentError(""))

    infidelity = similar(ITC.getdata(pepo))

    allbonds = bonds(pepo, stepper.axis)

    isreversed(stepper) && (allbonds = collect(Iterators.Reverse(allbonds)))

    checker = true

    shiftf = checker ? (v, i) -> circshift(v, i - 1) : (v, _) -> circshift(v, 0)

    bondcols = map(in -> shiftf(in[2], in[1]), enumerate(eachslice(allbonds; dims=2)))

    while !isempty(bondcols)
        bondcol = first(bondcols)

        while !isempty(bondcol)
            bond = first(bondcol)

            env = environment!(pepo, SimpleUpdateState(), bond)

            # println(bond)
            # println("alt:", altbonds[i])

            # Evolved the pair of tensors connected by `bond`, leaving the original pair
            # unmutated.

            # pair = evolve(pepo, stepper, bond)
            # newpair, newweight = updatetensors(pair, env, truncalg, bond)

            newpair, newweight = evolve_updatetensors(pepo, env, stepper, truncalg, bond)
            # Loop through all the columns 
            for col in bondcols
                thisbond = popfirst!(col)

                pepo[thisbond] = mapreduce(copy, =>, newpair)
                setweight!(pepo, newweight, thisbond; copy=false)

                # If not a checkerboard lattice, only update the first column
                checker || break
            end

            # quicknormalize!(pepo)
            # hermitianize!(pepo)
        end

        popfirst!(bondcols)
    end

    # simple_update_gauge!(pepo; reset=false, verbose=true)
    quicknormalize!(pepo)
    hermitianize!(pepo)
    #
    return pepo, infidelity
end

# isometries for sequential simple update
function isometries!(pair::Pair, pepo::AbstractPEPO, bond::Bond, truncalg::SimpleUpdate)
    D = bondspace(pepo, bond)

    trunc = truncspace(D)

    xi = truncalg.zeta

    iso, s, infid = simpleupdate!(pair, weights(pepo), bond; trunc=trunc, zeta=zeta)

    return iso, s, infid
end

function isometries(pair::Pair, pepo::AbstractPEPO, bond::Bond, truncalg::SimpleUpdate)
    return isometries(pair, weights(pepo), bond, truncalg; D=bondspace(pepo, bond))
end
function isometries(
    pair::Pair, weights::UnitCellBonds, bond::Bond, truncalg::SimpleUpdate; D, axial=false
)
    trunc = truncspace(D)

    xi = truncalg.zeta

    iso, s, infid = simpleupdate(pair, weights, bond; axial=axial, trunc=trunc, zeta=zeta)

    return iso, s, infid
end

# For updating in parellel
function parallelisometries(pepo::AbstractPEPO, bond::Bond, truncalg::SimpleUpdate; kwargs...)
    a, b = pepo[bond]

    pair = copy(a) => copy(b)

    return isometries!(pair, pepo, bond, truncalg)
end

function parallelisometries(pair::Pair, pepo::AbstractPEPO, bond::Bond, truncalg::SimpleUpdate)
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

    (u, v), s, infid = isometries_from_svd(P, Q, u0, v0; axis=axis, kwargs...)

    return pinv(u0) * u => v * pinv(v0), s, infid
end

function simple_update_svd!(pair; axis, kwargs...)
    (P, Q), (u0, v0) = rankreduce!(pair; axis=axis)

    iso, s, infid = isometries_from_svd(P, Q, u0, v0; axis=axis, kwargs...)

    return iso, s, infid
end

function isometries_from_svd(P, Q, u0, v0; zeta=Inf, trunc=notrunc(), absorb=true, axis)
    # u, s, v, err2 = safetsvd!(normalize(u0 * v0); trunc=trunc)
    u, s, v, err2 = safetsvd!(u0 * v0; trunc=trunc)

    # rmul!(s, 1 / tr(s))
    maxnormalize!(s)

    uv = u => v

    if absorb
        uv, s = applyweight(uv, s, xi)
        f = localfidelity(u, v, P, Q, u0, v0; axis=axis)
    else
        xi < Inf && rescale!(s, copy(s), xi)
        # f = fidelity(u * sqrt(s), sqrt(s) * v, P, Q, u0, v0)
    end
    f = 0

    return uv, s, 1 - f
end

function hori_weights(uten::T, vten::T, U, V; axial) where {T<:AbsTen{2,4}}
    s1, w1, n1 = U
    e2, s2, n2 = V

    if axial
        @tensoropt begin
            unew[k b; e ss w nn] := uten[k b; e s w n] * s1[s; ss] * n1[nn; n]
            vnew[k b; e ss w nn] := vten[k b; e s w n] * s2[s; ss] * n2[nn; n]
        end
    else
        @tensoropt begin
            unew[k b; e ss ww nn] := uten[k b; e s w n] * s1[s; ss] * w1[ww; w] * n1[nn; n]
            vnew[k b; ee ss w nn] := vten[k b; e s w n] * s2[s; ss] * e2[e; ee] * n2[nn; n]
        end
    end
    return unew, vnew
end
function hori_weights(uten::T, vten::T, U, V; axial) where {T<:AbsTen{2,5}}
    s1, w1, n1 = U
    e2, s2, n2 = V

    if axial
        @tensoropt begin
            unew[k b; e1 e2 ss w1 w2 nn] :=
                uten[k b; e1 e2 s w1 w2 n] * s1[s; ss] * n1[nn; n]
            vnew[k b; e1 e2 ss w1 w2 nn] :=
                vten[k b; e1 e2 s w1 w2 n] * s2[s; ss] * n2[nn; n]
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

function vert_weights(uten::T, vten::T, U, V; axial) where {T<:AbsTen{2,4}}
    e1, w1, n1 = U
    e2, s2, w2 = V

    if axial
        @tensoropt begin
            unew[k b; ee s ww n] := uten[k b; e s w n] * e1[e; ee] * w1[ww; w]
            vnew[k b; ee s ww n] := vten[k b; e s w n] * e2[e; ee] * w2[ww; w]
        end
    else
        @tensoropt begin
            unew[k b; ee s ww nn] := uten[k b; e s w n] * e1[e; ee] * w1[ww; w] * n1[nn; n]
            vnew[k b; ee ss ww n] := vten[k b; e s w n] * s2[s; ss] * e2[e; ee] * w2[ww; w]
        end
    end

    return unew, vnew
end

function vert_weights(uten::T, vten::T, U, V; axial) where {T<:AbsTen{2,5}}
    e1, w1, n1 = U
    e2, s2, w2 = V

    if axial
        @tensoropt begin
            unew[k b; ee s1 s2 ww n1 n2] :=
                uten[k b; e s1 s2 w n1 n2] * e1[e; ee] * w1[ww; w]
            vnew[k b; ee s1 s2 ww n1 n2] :=
                vten[k b; e s1 s2 w n1 n2] * e2[e; ee] * w2[ww; w]
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
    pair::Pair{T}, weights::UnitCellBonds, bond::Bond; inverse=false, axial=false
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
        unew, vnew = hori_weights(uten, vten, U, V; axial=axial)
    elseif axis == 2
        unew, vnew = vert_weights(uten, vten, U, V; axial=axial)
    end

    copy!(uten, unew)
    copy!(vten, vnew)

    return uten => vten
end

function absorb_axial_weights!(
    site::AbsTen{2,4},
    weights::UnitCellBonds{G},
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

function weightenvironment(pair, weights, bond)
    U, V = adjacent(weights, bond)

    QL, QR = pair
    # s/e Q <-u<-
    # n/w   <-v<- Q

    if bondaxis(bond) == 2
        weights = permutedims(weights)

        QL = permutedom(QL, (2, 1, 4, 3))
        QR = permutedom(QR, (2, 1, 4, 3))
    end

    s1, w1, n1 = U
    e2, s2, n2 = V

    @tensoropt M[D7 E1; D1 E7] :=
        QL[k1 b1; D1 D2 D3 D4] *
        (QL')[E1 E2 E3 E4; k1 b1] *
        QR[k2 b2; D5 D6 D7 D8] *
        (QR')[E5 E6 E7 E8; k2 b2] *
        s1[D2; E2] *
        w1[E3; D3] *
        n1[E4; D4] *
        e2[D5; E5] *
        s2[D6; E6] *
        n2[E8; D8]

    return M
end
