struct PEPO{G,PType<:AbsTen{2,4},CType} <: AbstractPEPO
    state::Lattice{G,PType,Matrix{PType}}
    weights::Weights{G,CType}
end

latticetensors(pepo::PEPO) = pepo.state

getweights(pepo::PEPO) = pepo.weights
getweights(pepo::PEPO, bond) = getweights(pepo)[bond]

Base.getindex(p::PEPO, bond) = latticetensors(p)[bond]

bloch(a) = 1 / 2 * (I + sum(a * σ for (a, σ) in zip(a, PAULI)))

function pepositedata(a, Dw::Int, Ds::Int, De::Int, Dn::Int)
    m = bloch(a)

    tw = zeros(ComplexF64, Dw, 2, 2) # out, in
    ts = zeros(ComplexF64, Ds, 2, 2)
    te = zeros(ComplexF64, De, 2, 2)
    tn = zeros(ComplexF64, Dn, 2, 2)

    tw[1, :, :] = ts[1, :, :] = te[1, :, :] = tn[1, :, :] = m

    @tensoropt p[bot, top, e, s, w, n] :=
        tw[w, s1, top] * ts[s, s2, s1] * te[e, s3, s2] * tn[n, bot, s3]

    return p
end

struct ThermalState end

function PEPO(
    a,
    physbond::S,
    eastbond::S,
    southbond::S=eastbond;
    cellsize=Tuple{Int,Int},
    issymmetric=true,
) where {S}
    i1, i2 = cellsize
    return PEPO(
        a,
        fill(physbond, (i1, i2)),
        fill(eastbond, (i1, i2)),
        fill(southbond, (i1, i1));
        issymmetric=issymmetric,
    )
end

function pepodata(a::Vector, east, south, west, north)
    an = normalize(a)
    data = @. pepositedata($Ref(an), dim(east), dim(south), dim(west), dim(north))
    return data
end
function pepodata(a::Matrix{<:Vector}, east, south, west, north)
    an = normalize(a)
    data = @. pepositedata(an, dim(east), dim(south), dim(west), dim(north))
    return data
end

function pepodata(::ThermalState, east, south, west, north)
    data = @. pepositedata($Ref([0, 0, 0]), dim(east), dim(south), dim(west), dim(north))
    return 2 * data
end

dimtospace(d::Int) = ℂ^d
dimtospace(s::IndexSpace) = s

function PEPO(
    a,
    physbond::AbstractMatrix,
    eastbond::AbstractMatrix,
    southbond::AbstractMatrix=eastbond;
    issymmetric=true,
    geometry::G=Square(),
) where {G}
    physlat = dimtospace.(Lattice{G}(physbond, issymmetric))
    return PEPO(a, physlat, eastbond, southbond)
end
function PEPO(
    a,
    physbond::AbstractLattice,
    eastbond::AbstractMatrix,
    southbond::AbstractMatrix=eastbond;
)
    west = dimtospace.(circshift(eastbond, (1, 0)))
    east = dimtospace.(eastbond)
    south = dimtospace.(southbond)
    north = dimtospace.(circshift(southbond, (0, 1)))

    data = pepodata(a, east, south, west, north)

    tensors = @. TensorMap(
        data, physbond * adjoint(physbond), east * south * adjoint(west) * adjoint(north)
    )

    return PEPO(tensors)
end

function PEPO(tensors::AbstractLattice{G,<:AbsTen{2,4}}) where {G}
    horizontal_weights = @. id(getindex(domain(tensors), 1))
    vertical_weights = @. id(getindex(domain(tensors), 2))

    weights = Weights(horizontal_weights, vertical_weights)

    return PEPO(tensors, weights)
end

merge_pepobond(p::AbsTen{2,5}, dir::Symbol) = merge_pepobond(p, _DIR_TO_DOMIND[dir])
function merge_pepobond(p::AbstractTensorMap{S,2,5}, ind::Integer) where {S}
    dom = domain(p)

    new_eastbond = fuse(dom[ind] * dom[ind + 1])

    vnew_domain = deleteat!([dom...], ind + 1)
    vnew_domain[ind] = new_eastbond

    new_domain = *(vnew_domain)::ProductSpace{S,4}

    p_merged = TensorMap(p.data, codomain(p), new_domain)

    return p_merged
end

function mergeinto!(p_merged, p)
    @inbounds for c in blocksectors(p)
        copy!(TensorKit.block(p_merged, c), TensorKit.block(p, c))
    end
    return p_merged
end
