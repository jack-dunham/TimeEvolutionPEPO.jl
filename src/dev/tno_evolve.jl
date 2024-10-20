function initevolution(
    pepo::AbstractPEPO, map::DynamicalMap{<:WIIPEPO}, evoalg::Union{TNO}
)
    pepo = _initevolution(pepo, map.x, evoalg.truncalg)
    return pepo
end
function _initevolution(pepo::AbstractPEPO, ::WIIPEPO, ::VGU)
    return GlobalUpdatablePEPO(pepo, deepcopy(getstate(pepo)), false)
end
function _initevolution(pepo::AbstractPEPO, map::WIIPEPO, ::SU)
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
    pepo::AbstractPEPO,
    truncators::Union{SimpleUpdateState,VidalTruncationState},
    map::DynamicalMap{<:WIIPEPO}, #{<:TrotterGates},
    order;
    kwargs...,
)
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
