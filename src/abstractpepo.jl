struct Weights{G,CType<:AbsTen{1,1}}
    horizontal::Lattice{G,CType,Matrix{CType}}
    vertical::Lattice{G,CType,Matrix{CType}}
end

function Base.getindex(weights::Weights, bond::Bond)
    axis = getaxis(bond)
    if axis == 1
        return weights.horizontal[left(bond)]
    elseif axis == 2
        return weights.vertical[left(bond)]
    else
        throw(
            ErrorException(
                "Impossible `else` reached! Value of axis = $axis returned from `getaxis` neither 1 or 2",
            ),
        )
    end
end

function TensorKit.truncspace(w::Weights, bond::Bond)
    return TensorKit.TruncationSpace(codomain(w[bond])[1])
end

function changeaxis(weights::Weights)
    return Weights(permutedims(weights.vertical), permutedims(weights.horizontal))
end

function tolerance!(weights::W, new_weights::W, tau::Float64) where {W<:Weights}
    h = _tolerance!.(weights.horizontal, new_weights.horizontal, tau)
    v = _tolerance!.(weights.vertical, new_weights.vertical, tau)
    return max(h..., v...)
end
function _tolerance!(ow, nw, tau::Float64)
    rv = norm(ow - nw) / (tau * norm(nw) * dim(codomain(nw)))
    copy!(ow, nw)
    return rv
end

abstract type AbstractPEPO end

issymmetric(pepo::AbstractPEPO) = issymmetric(latticetensors(pepo))
physicalspace(pepo::AbstractPEPO) = getindex.(codomain.(latticetensors(pepo)), 1)
uniqueindices(pepo::AbstractPEPO) = uniqueindices(latticetensors(pepo))

TensorKit.truncspace(p::AbstractPEPO, bond::Bond) = truncspace(getweights(p), bond)

quicknormalize!(pepo::AbstractPEPO) = normalize!.(latticetensors(pepo))

TensorKit.scalartype(pepo::AbstractPEPO) = scalartype(latticetensors(pepo))

function setweights!(pepo::AbstractPEPO, s, bond::Bond)
    copy!(getweights(pepo, bond), s)
    return pepo
end

getbonds(pepo::AbstractPEPO, args...) = getbonds(latticetensors(pepo), args...)

function changeaxis(pepo::AbstractPEPO)
    pepo_permuted = broadcast(latticetensors(pepo)) do tensor
        permute(tensor, (1, 2), (4, 3, 6, 5))
    end
    return PEPO(permutedims(pepo_permuted), changeaxis(getweights(pepo)))
end

function changeaxis!(pepo_permuted::AbstractPEPO, pepo::AbstractPEPO)
    broadcast(
        latticetensors(pepo_permuted), permutedims(latticetensors(pepo))
    ) do pdst, psrc
        permute!(pdst, psrc, ((1, 2), (4, 3, 6, 5)))
    end
    return pepo_permuted
end

function bondspace(pepo::AbstractPEPO, bond)
    return domain.(latticetensors(pepo), getaxis(bond))[left(bond)]
end

hermitianize!(pepo::AbstractPEPO) = hermitianize!.(latticetensors(pepo))

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

### Updatable PEPO's, used to pre-allocate the arrays that will store the intermediate stage
# where the pepo is evolved

abstract type AbstractUpdatablePEPO <: AbstractPEPO end

# Get the underlying PEPO.
getpepo(p::AbstractUpdatablePEPO) = p.parent

# Get the tensors of the underlying parent PEPO structure
latticetensors(p::AbstractUpdatablePEPO) = latticetensors(getpepo(p))

# Get the evolved state of the PEPO. May or may not be up to date.
evolvedtensors(p::AbstractUpdatablePEPO) = p.evolved_tensors

# Check if PEPO is in an evolved state (i.e. if `getevolved` outputs the current state.)
isevolved(p::AbstractUpdatablePEPO) = p.isevolved[]

# Set evolved state.
setisevolved!(p::AbstractUpdatablePEPO, bool::Bool) = p.isevolved.x = bool

function _getindex(p::AbstractUpdatablePEPO, bond::Bond, ::Val{false})
    return getindex(latticetensors(p), bond)
end

Base.getindex(p::AbstractUpdatablePEPO, bond::Bond) = _getindex(p, bond, Val(isevolved(p)))

getweights(pepo::AbstractUpdatablePEPO, args...) = getweights(getpepo(pepo), args...)

struct LocalUpdatablePEPO{G,PType,CType} <: AbstractUpdatablePEPO
    parent::PEPO{G,PType,CType}
    evolved_tensors::Lattice{G,Tuple{PType,PType},Matrix{Tuple{PType,PType}}}
    bond::Ref{Bond}
    isevolved::Ref{Bool}
end

currentbond(p::LocalUpdatablePEPO) = p.bond.x::Bond
function setcurrentbond!(p::LocalUpdatablePEPO, bond::Bond)
    p.bond.x = bond
    return p
end

function evolvedpair(p::LocalUpdatablePEPO{G,PType}, bond=currentbond(p)) where {G,PType}
    leftpair, rightpair = evolvedtensors(p)[bond]
    return leftpair[1]::PType, rightpair[2]::PType
end

function setevolvedpair!(p, val1, val2, bond)
    evolvedbuffer = evolvedtensors(p)

    leftpair, rightpair = evolvedbuffer[bond]
    newleftpair = (val1, leftpair[2])
    newrightpair = (rightpair[1], val2)

    evolvedbuffer[bond] = newleftpair, newrightpair
    return p
end

function parentpair(p::LocalUpdatablePEPO{G,PType}, bond=currentbond(p)) where {G,PType}
    leftpair, rightpair = latticetensors(p)[bond]
    return leftpair::PType, rightpair::PType
end

function _getindex(pe::LocalUpdatablePEPO, bond::Bond, ::Val{true})
    updated_bond = currentbond(pe)

    new_left, new_right = evolvedpair(pe)

    rv_left, rv_right = latticetensors(pe)[bond]

    if left(bond) == left(updated_bond)
        rv_left, rv_right = new_left, new_right
    elseif left(bond) == right(updated_bond)
        rv_left = new_right
    elseif right(bond) == left(updated_bond)
        rv_right = new_left
    end

    return rv_left, rv_right
end

struct GlobalUpdatablePEPO{G,PType,CType} <: AbstractUpdatablePEPO
    parent::PEPO{G,PType,CType}
    evolved_tensors::Lattice{G,PType,Matrix{PType}}
    isevolved::Ref{Bool}
end

#? intermediate tensors

# GloballyEvolvingPEPO(pepo::PEPO) = GloballyEvolvingPEPO(pepo, deepcopy(getstate(pepo)))

getcurrentstate(p::GlobalUpdatablePEPO) = _getcurrentstate(p, Val(isevolved(p)))
_getcurrentstate(p, ::Val{true}) = evolvedtensors(p)
_getcurrentstate(p, ::Val{false}) = latticetensors(p)

function _getindex(p::GlobalUpdatablePEPO, bond::Bond, ::Val{true})
    return getindex(evolvedtensors(p), bond)
end

function copyevolved(pepo::G) where {G<:GlobalUpdatablePEPO}
    newpepo = G(pepo.parent, copy(pepo.evolved_tensors), pepo.isevolved)
    return newpepo
end

function truncateinto!(dst::G, src::G) where {G<:GlobalUpdatablePEPO}
    copy!.(latticetensors(dst), evolvedtensors(src))
    setisevolved!(dst, false)
    return dst
end

###

function minimalcanonicalform!(tensor::AbsTen{2,4}; tol=1e-12, maxiter=10000)
    hor = id(domain(tensor)[1])
    ver = id(domain(tensor)[2])
    T = deepcopy(tensor)
    err = Inf
    numiter = 0

    elp = @elapsed begin
        for t in 1:maxiter
            numiter = t

            mcfgauge!(tensor, T, hor, ver)

            @tensoropt (x => 2, y => 2) p_h1[e1; e2] :=
                tensor'[e1 s w n; x y] * tensor[x y; e2 s w n]
            @tensoropt (x => 2, y => 2) p_h2[w1; w2] :=
                tensor'[e s w1 n; x y] * tensor[x y; e s w2 n]
            @tensoropt (x => 2, y => 2) p_v1[s1; s2] :=
                tensor'[e s1 w n; x y] * tensor[x y; e s2 w n]
            @tensoropt (x => 2, y => 2) p_v2[n1; n2] :=
                tensor'[e s w n1; x y] * tensor[x y; e s w n2]

            h_diff = p_h1 - transpose(p_h2)
            v_diff = p_v1 - transpose(p_v2)

            @tensor trketbra = tr(p_h1)
            err = 1 / real(trketbra) * (norm(h_diff)^2 + norm(v_diff)^2)

            # if err ≤ tol^2
            #     break
            # end

            if err ≤ tol
                break
            end
            hor = exp(-1 / 8 * 1 / trketbra * h_diff) * hor
            ver = exp(-1 / 8 * 1 / trketbra * v_diff) * ver
        end
    end
    @info "Minimal canonical form gauging: $err after $numiter iterations taking $(elp)s"
    return tensor
end

function mcfgauge!(p_out, p_in, g1, g2) #hor, ver
    g1i = permute(inv(g1), (2,), (1,))
    g2i = permute(inv(g2), (2,), (1,))
    @tensor p_out[k b; e s w n] =
        p_in[k b; ee ss ww nn] * g1[ee; e] * g2[ss; s] * g1i[ww; w] * g2i[nn; n]
    return p_out
end
