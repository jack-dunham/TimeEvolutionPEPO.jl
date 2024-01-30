@kwdef struct TNO{TAlg<:Union{SU,VGU}} <: AbstractTimeEvolution
    timestep::Float64 = 0.01
    numsteps::Int = 100
    breaktol::Float64 = 0.0
    order::Int = 1
    truncalg::TAlg = SU()
    verbose::Bool = true
end

function initialize_evolution(
    pepo::AbstractPEPO, map::DynamicalMap{<:WIIPEPO}, evoalg::Union{TNO}
)
    xpepo = _initialize_evolution(pepo, map.x, evoalg.truncalg)
    return TimeDependantPEPO(xpepo)
end
function _initialize_evolution(pepo::AbstractPEPO, ::WIIPEPO, ::VGU)
    return GlobalUpdatablePEPO(pepo, deepcopy(getstate(pepo)), Ref(false))
end
function _initialize_evolution(pepo::AbstractPEPO, map::WIIPEPO, ::SU)
    function useful_f(x, l)
        D = domain(x)
        η = domain(l)

        dom = map(x -> fuse(x[1] ⊗ x[2]), zip(D, η))

        rv = TensorMap(undef, promote_type(eltype(x), eltype(l)), codomain(x), dom)
        return rv
    end

    evolved = useful_f.(getstate(map), map.data)

    return GlobalUpdatablePEPO(pepo, evolved, Ref(false))
end

function timestep!(
    pepo::TimeDependantPEPO,
    truncators::Union{SimpleUpdateState,VidalTruncationState},
    map::DynamicalMap{<:WIIPEPO}, #{<:TrotterGates},
    order::Val{N};
    kwargs...,
) where {N}
    tnostep!(pepo.x, truncators, map.x, order; kwargs...)

    return pepo
end

function tnostep!(pepo, truncators, map, ::Val{1}; kwargs...)
    updatestep!(pepo, truncators, map; kwargs...)

    return pepo
end

function updatestep!(
    pepo::GlobalUpdatablePEPO, truncators::VidalTruncationState, ::WIIPEPO; truncalg::VGU
)
    mts = truncators.messages
    vidals = truncators.isometries
    weights = truncators.weights
    map = truncators.map

    beliefpropagation!(mts, pepo, map)

    # @info "" mts[1, 1].east.data

    bonds = [getbonds(pepo, 1)..., getbonds(pepo, 2)...]

    for bond in bonds
        u, s, v = vidalisometries(pepo, mts, bond; trunc=truncspace(pepo, bond))
        # u, s, v = vidalisometries(pepo, mts, bond; trunc=notrunc())
        normalize!(s)
        @debug s
        setvidal!(truncators, u, s, v, bond)
        # setweights!(pepo, s, bond)
    end

    # @info "" (
    #     normalize!(vidalgauge_test(getstate(pepo)[1, 1], map.data[1, 1], wh, wv)).data
    # )

    truncate!(pepo, truncators)

    quicknormalize!(pepo)

    return pepo
end

struct IntermediateVidalPEPO{G,P}
    state::Lattice{G,P,Matrix{P}}
    isevolved::Lattice{G,Bool,Matrix{Bool}}
end

function simpleupdate_truncate(pepo, truncators::VidalTruncationState, bond)
    # First, apply the message tensors, and absorb sqrt of weights
    A, B = apply_message_tensors(pepo, truncators, bond)

    # Perform a rank reduction for efficieny reasons 
    QL, QR = _rankreduce!(A, B)

    # Take the SVD of the resulting merged tensor
    u, s, v = tsvd!(QL * QR; trunc=truncspace(pepo))
    _partialtruncate(A, u * sqrt(s))
    _partialtruncate(sqrt(s) * v, B)

    Γ_A, Γ_B = insert_identity_resolution(pepo, truncators, bond)

    setisevolved!(pepo, true, bond)
    setstate!(pepo, Γ_A, Γ_B, bond)

    return A, B
end
# apply_message_tensors_horizontal(pepo, truncators, bond) = _apply_message_tensors(pepo, truncators, bond, 1, 3)
# apply_message_tensors_vertical(pepo, truncators, bond) =  _apply_message_tensors(pepo, truncators, bond, 2, 4)

function insert_identity_resolution(pepo, truncators, bond)
    ax = getaxis(bond)
    if ax == 1
        left_rotation = identity
        right_rotation = x -> permutedom(x, (3, 4, 1, 2))
    elseif ax == 2
        left_rotation = x -> permutedom(x, (2, 3, 4, 1))
        right_rotation = x -> permutedom(x, (4, 1, 2, 3))
    end

    w_left, w_right = truncators.weights[bond]

    l_w = get_matrices_except_dir(w_left, ax)
    r_w = get_matrices_except_dir(w_right, ax + 2)

    left_w = @. sqrt(l_w) * pinv(l_w)
    right_w = @. sqrt(r_w) * pinv(r_w)

    A = apply_identity_resolution(left_rotation(left0), left_w...)
    B = apply_identity_resolution(right_rotation(right0), right_w...)
    return A, B
end

function apply_message_tensors(pepo, truncators, bond)
    ax = getaxis(bond)
    if ax == 1
        left_rotation = identity
        right_rotation = x -> permutedom(x, (3, 4, 1, 2))
    elseif ax == 2
        left_rotation = x -> permutedom(x, (2, 3, 4, 1))
        right_rotation = x -> permutedom(x, (4, 1, 2, 3))
    end

    left0, right0 = pepo[bond]
    map_left, map_right = truncators.map.data[bond]

    isom_left, isom_right = truncators.isometries[bond]
    w_left, w_right = truncators.weights[bond]

    left_u = get_matrices_except_dir(isom_left, ax)
    left_w = sqrt_weights(w_left, ax)

    right_u = get_matrices_except_dir(isom_right, ax + 2)
    right_w = sqrt_weights(w_right, ax + 2)

    A = apply_message_tensors_for_truncation(
        left_rotation(left0), left_rotation(map_left), left_u..., left_w...
    )
    B = apply_message_tensors_for_truncation(
        right_rotation(right0), right_rotation(map_right), right_u..., right_w...
    )
    return A, B
end

function sqrt_weights(weights::VidalWeights, dir::Int)
    u1, u2, u3 = sqrt.(get_matrices_except_dir(weights, dir))
    return u1, u2, u3
end
function get_matrices_except_dir(matrices::Union{VidalWeights,VidalIsometries}, dir::Int)
    mats = unpack(matrices)
    if dir == 1 # ignore east
        _, u1, u2, u3 = mats
    elseif dir == 2 # south
        u1, _, u2, u3 = mats
    elseif dir == 3 # west
        u1, u2, _, u3 = mats
    elseif dir == 4 # north
        u1, u2, u3, _ = mats
    else
        throw(ArgumentError(""))
    end
    return u1, u2, u3
end

# east. c matrices should be square root of bond matrix (symmetric gauge)
function apply_message_tensors_for_truncation(T, P, us, vw, vn, cs, cw, cn)
    @tensoropt (
        k => 2,
        b => 2,
        kk => 2,
        bb => 2,
        e1 => D,
        e2 => D,
        w1 => D,
        w2 => D,
        s1 => D,
        s2 => D,
        n1 => D,
        n2 => D,
        ee1 => D,
        ee2 => D,
        ww1 => D,
        ww2 => D,
        ss1 => D,
        ss2 => D,
        nn1 => D,
        nn2 => D,
    ) out[kk bb; e1 e2 s w n] :=
        T[k b; e1 s1 w1 n1] *
        P[kk bb k b; e2 s2 w2 n2] *
        us[s1 s2; ss] *
        cs[ss; s] *
        vw[w1 w2; ww] *
        cw[ww; w] *
        vn[n1 n2; nn] *
        cn[nn; n]
    return out
end
function apply_identity_resolution(T, cs, cw, cn)
    @tensoropt out[kk bb; e s w n] := T[k b; e ss ww nn]
    cs[ss; s] * cw[ww; w] * cn[nn; n]
    return out
end
