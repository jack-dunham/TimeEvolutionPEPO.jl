struct PEPO{G,PType<:TenAbs{4},CType} <: AbstractPEPO{G,PType}
    state::UnitCell{G,PType,Matrix{PType}}
    weights::UnitCellBonds{G,CType,Matrix{NTuple{2,CType}}}
end

ITC.getdata(pepo::PEPO) = pepo.state

latticetype(::PEPO{G,P}) where {G,P} = UnitCell{G,P,Matrix{P}}

weights(pepo::PEPO) = pepo.weights
getweight(pepo::PEPO, bond) = weights(pepo)[bond]

# Base.getindex(p::PEPO, bond::Bond) = getaxis(p)[bond]

bloch(a) = 1 / 2 * (I + sum(a * σ for (a, σ) in zip(a, PAULI)))

function pepositedata(m::AbstractMatrix, De::Int, Ds::Int, Dw::Int, Dn::Int)
    out, in = size(m)

    te = zeros(ComplexF64, De, out, in)
    ts = zeros(ComplexF64, Ds, out, in)
    tw = zeros(ComplexF64, Dw, out, in)
    tn = zeros(ComplexF64, Dn, out, in)

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

# function pepodata(
#     a::Union{AbstractVecOrMat{<:Number},NTuple{3,Number}}, east, south, west, north
# )
#     data = @. pepositedata($Ref(a), dim(east), dim(south), dim(west), dim(north))
#     return data
# end
# function pepodata(
#     a::AbstractMatrix{T}, east, south, west, north
# ) where {T<:Union{AbstractVecOrMat,NTuple{3,Number}}}
#     data = @. pepositedata(a, dim(east), dim(south), dim(west), dim(north))
#     return data
# end
#
# function pepodata(::ThermalState, east, south, west, north)
#     data = @. pepositedata($Ref([0, 0, 0]), dim(east), dim(south), dim(west), dim(north))
#     return 2 * data
# end
function pepodata(::ThermalState, phys, east, south, west, north)
    return pepositedata(I(dim(phys)), dim(east), dim(south), dim(west), dim(north))
end
function pepodata(rho::Tuple, phys, east, south, west, north)
    return pepositedata(bloch(rho), dim(east), dim(south), dim(west), dim(north))
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
    physlat = dimtospace.(UnitCell{G}(physbond, issymmetric))
    return PEPO(a, physlat, eastbond, southbond)
end
function PEPO(
    a::AbstractMatrix,
    physbond::AbstractMatrix,
    eastbond::AbstractMatrix,
    southbond::AbstractMatrix=eastbond;
)
    east = dimtospace.(eastbond)
    south = dimtospace.(southbond)

    west = dimtospace.(circshift(eastbond, (1, 0)))
    north = dimtospace.(circshift(southbond, (0, 1)))

    # Generate data 
    data = pepodata.(a, physbond, east, south, west, north)

    tensors = broadcast(data, physbond, east, south, west, north) do arr, p, e, s, w, n
        ten = TensorMap(arr, p * p', e * s * w' * n')
        return ten
    end

    return PEPO(tensors)
end

function PEPO(tensors::AbstractUnitCell{G,<:AbsTen{2,4}}) where {G}
    horizontal_weights = @. id(getindex(domain(tensors), 1))
    vertical_weights = @. id(getindex(domain(tensors), 2))

    weights = UnitCellBonds(horizontal_weights, vertical_weights)

    return PEPO(tensors, weights)
end

merge_pepobond(p::AbsTen{2,5}, dir::Symbol) = merge_pepobond(p, _DIR_TO_DOMIND[dir])
function merge_pepobond(p::AbsTen{2,5,S}, ind::Integer) where {S}
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
        bdst = TensorKit.block(p_merged, c)
        bsrc = TensorKit.block(p, c)

        copy!(bdst, bsrc)
    end
    return p_merged
end

function expandspace(p::TensorMap{T,S}, s::S) where {T,S<:IndexSpace}
    return expandspace(p, s * s * s' * s')
end
function expandspace(p::TensorMap{T,S}, s::ProductSpace{S,4}) where {T,S<:IndexSpace}
    u = mapreduce((a, b) -> isometry(a', b'), ⊗, s, domain(p))

    return p * _transpose(u)
end

function expandspace!(pepo, space::IndexSpace)
    pten = getaxis(pepo)

    broadcast!(pten, pten, Ref(space)) do p, s
        return expandspace(p, s)
    end

    wten = weights(pepo)

    wtennew = broadcast(wten) do w
        c = codomain(w)
        d = domain(w)

        println(typeof(w))

        T = scalartype(w)

        u = TensorMap(delta(T, dim(space), dim(c)), space, c)
        v = TensorMap(delta(T, dim(d), dim(space)), d, space)

        return u * w * v
    end

    for bond in eachbond(pepo)
        setweight!(pepo, wtennew[bond], bond; copy=false)
    end

    return pepo
end

function subpepo(pepo, i1, i2; weights=false)
    t = UnitCell(axisbuffer(pepo)[i1:i1, i2:i2])

    if weights
        w = UnitCellBonds(weights(pepo)[i1:i1, i2:i2])
        return PEPO(t, w)
    else
        return PEPO(t)
    end
end

function trivialise(pepo::PEPO)
    newstate = broadcast(pepo.state) do symten
        arr = convert(Array, symten)

        cod = mapreduce(ComplexSpace, *, dims(codomain(symten)))
        dom = map(ComplexSpace, dims(domain(symten)))

        return TensorMap(arr, cod[1] * cod[2]', dom[1] * dom[2] * dom[3]' * dom[4]')
    end

    neww = broadcast(pepo.weights) do symten
        arr = convert(Array, symten)

        cod = mapreduce(ComplexSpace, *, dims(codomain(symten)))
        dom = mapreduce(ComplexSpace, *, dims(domain(symten)))

        return TensorMap(arr, cod, dom)
    end

    return PEPO(newstate, neww)
end
