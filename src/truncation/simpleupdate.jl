function newtruncators(pepo, ::AbstractTimeStepper, ::AbstractSimpleTruncationScheme)
    return SimpleUpdateState()
end
# Simple update interface
environment(pepo, truncalg::AbstractSimpleTruncationScheme, bond) = bondweights(pepo)
function environment!(truncator, pepo, truncalg::AbstractSimpleTruncationScheme, bond)
    return environment(pepo, truncalg, bond)
end

function trotterstep!(
    pepo::PEPO,
    truncator,
    stepper::LocalStepper,
    truncalg::AbstractSimpleTruncationScheme,
    kwargs...,
)

    # Evolve and then truncate each bond in sequence
    size(pepo) < (2, 2) && throw(ArgumentError(""))

    infidelity = similar(TRGroups.getdata(pepo))

    allbonds = bonds(pepo, stepper.axis)

    for bond in allbonds
        env = environment!(truncator, pepo, truncalg, bond)

        # println("alt:", altbonds[i])

        # Evolved the pair of tensors connected by `bond`, leaving the original pair
        # unmutated.

        if hasfield(typeof(truncalg), :reduce)
            if truncalg.reduce === :post
                pair = evolve(pepo, stepper, bond)
                newpair, newweight = updatetensors(pair, env, truncalg, bond)
            elseif truncalg.reduce === :pre
                newpair, newweight = updatetensors(pepo, env, stepper, truncalg, bond)
            end
        else
            pair = evolve(pepo, stepper, bond)
            newpair, newweight = updatetensors(pair, env, truncalg, bond)
        end

        pepo[bond] = mapreduce(copy, =>, newpair)
        setweight!(pepo, newweight, bond; copy=false)
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

    zeta = truncalg.zeta

    iso, s, infid = simpleupdate!(pair, bondweights(pepo), bond; trunc=trunc, zeta=zeta)

    return iso, s, infid
end

function isometries(pair::Pair, pepo::AbstractPEPO, bond::Bond, truncalg::SimpleUpdate)
    return isometries(pair, bondweights(pepo), bond, truncalg; D=bondspace(pepo, bond))
end
function isometries(
    pair::Pair, weights::UnitCellBonds, bond::Bond, truncalg::SimpleUpdate; D, axial=false
)
    trunc = truncspace(D)

    zeta = truncalg.zeta

    iso, s, infid = simpleupdate(pair, weights, bond; axial=axial, trunc=trunc, zeta=zeta)

    return iso, s, infid
end

# For updating in parellel
function parallelisometries(
    pepo::AbstractPEPO, bond::Bond, truncalg::SimpleUpdate; kwargs...
)
    a, b = pepo[bond]

    pair = copy(a) => copy(b)

    return isometries!(pair, pepo, bond, truncalg)
end

function parallelisometries(
    pair::Pair, pepo::AbstractPEPO, bond::Bond, truncalg::SimpleUpdate
)
    return isometries(pair, pepo, bond, truncalg)
end
# Sequential simple update
function simpleupdate!(pair, weights, bond; trunc, absorb=true, axial=false, kwargs...)
    axis = bondaxis(bond)

    if absorb
        pair = absorb_adjacent_weights(pair, weights, bond; inverse=false, axial=axial)
    end

    iso, s, infid = simple_update_svd!(pair; trunc=trunc, absorb=true, axis=axis, kwargs...)

    if absorb
        pair = absorb_adjacent_weights(pair, weights, bond; inverse=true, axial=axial)
    end

    return iso, s, infid
end

function simpleupdate(pair, weights, bond; trunc, axial=false, absorb=true, kwargs...)
    axis = bondaxis(bond)

    newpair = mapreduce(copy, =>, pair)

    if absorb
        newpair = absorb_adjacent_weights(pair, weights, bond; axial=axial, inverse=false)
    end

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

function isometries_from_svd(P, Q, u0, v0; zeta=Inf, trunc, absorb=true, axis)
    u, s, v, err2 = safetsvd!(normalize(u0 * v0); trunc=trunc)

    # rmul!(s, 1 / tr(s))
    maxnormalize!(s)

    uv = u => v

    if absorb
        uv, s = applyweight(uv, s, zeta)
        f = localfidelity(u, v, P, Q, u0, v0; axis=axis)
    else
        zeta < Inf && rescale!(s, copy(s), zeta)
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

function absorb_adjacent_weights(
    pair::TensorPair{2}, weights::UnitCellBonds, bond::Bond; inverse=false, axial=false
)
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

    return unew => vnew
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
