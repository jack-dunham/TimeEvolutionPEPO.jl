struct PEPO{G,PType<:TenAbs{4},CType} <: AbstractPEPO
    state::Lattice{G,PType,Matrix{PType}}
    weights::EdgeLattice{G,CType,Matrix{NTuple{2,CType}}}
end

tensors(pepo::PEPO) = pepo.state

latticetype(::PEPO{G,P}) where {G,P} = Lattice{G,P,Matrix{P}}

weights(pepo::PEPO) = pepo.weights
getweight(pepo::PEPO, bond) = weights(pepo)[bond]

Base.getindex(p::PEPO, bond::Bond) = tensors(p)[bond]

bloch(a) = 1 / 2 * (I + sum(a * σ for (a, σ) in zip(a, PAULI)))

function pepositedata(
    a::Union{AbstractVector,NTuple{3,Number}}, Dw::Int, Ds::Int, De::Int, Dn::Int
)
    return pepositedata(bloch(a), Dw, Ds, De, Dn)
end

function pepositedata(m::AbstractMatrix, Dw::Int, Ds::Int, De::Int, Dn::Int)
    out, in = size(m)

    tw = zeros(ComplexF64, Dw, out, in)
    ts = zeros(ComplexF64, Ds, out, in)
    te = zeros(ComplexF64, De, out, in)
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
function pepodata(::ThermalState, east, south, west, north)
    return 2 * pepositedata((0, 0, 0), dim(east), dim(south), dim(west), dim(north))
end
function pepodata(rho::Tuple, east, south, west, north)
    return pepositedata(rho, dim(east), dim(south), dim(west), dim(north))
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
    a::AbstractMatrix,
    physbond::AbstractMatrix,
    eastbond::AbstractMatrix,
    southbond::AbstractMatrix=eastbond;
)
    west = dimtospace.(circshift(eastbond, (1, 0)))
    east = dimtospace.(eastbond)
    south = dimtospace.(southbond)
    north = dimtospace.(circshift(southbond, (0, 1)))

    # Generate data 
    data = pepodata.(a, east, south, west, north)

    tensors = @. TensorMap(
        data, physbond * adjoint(physbond), east * south * adjoint(west) * adjoint(north)
    )

    return PEPO(tensors)
end

function PEPO(tensors::AbstractLattice{G,<:AbsTen{2,4}}) where {G}
    horizontal_weights = @. id(getindex(domain(tensors), 1))
    vertical_weights = @. id(getindex(domain(tensors), 2))

    weights = EdgeLattice(horizontal_weights, vertical_weights)

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
        bdst = TensorKit.block(p_merged, c)
        bsrc = TensorKit.block(p, c)
        copy!(bdst, bsrc)
    end
    return p_merged
end

mutable struct PEPOWithBuffer{G,PType,CType,StyleS,StyleT} <: AbstractPEPO
    const parent::PEPO{G,PType,CType}
    const buffer::Vector{Lattice{G,PType,Matrix{PType}}}
    const stepstyle::StyleS
    const truncstyle::StyleT
    state::Int
end

stepstyle(pepo) = pepo.stepstyle
truncstyle(pepo) = pepo.truncstyle

weights(pepo::PEPOWithBuffer) = weights(pepo.parent)
tensors(pepo::PEPOWithBuffer) = tensors(pepo.parent)

currentbuffer(pepo::PEPOWithBuffer) = _current_buffer(pepo.parent, pepo.buffer, pepo.state)
currentbuffer(pepo::PEPO) = tensors(pepo)
currentbuffer(cell::AbstractUnitCell) = cell

function truncate!(pepo::PEPOWithBuffer, isometries, weights)
    ustyle = stepstyle(pepo)
    tstyle = truncstyle(pepo)

    truncate!(ustyle, tstyle, pepo, isometries, weights)
    return pepo
end

function setaxis!(pepo::PEPOWithBuffer, axis)
    pepo.state = axis

    otheraxis = mod(axis + 1, 1:2)

    permutetranspose!(getbuffer(pepo, axis - 1), getbuffer(pepo, otheraxis - 1))

    return pepo
end
