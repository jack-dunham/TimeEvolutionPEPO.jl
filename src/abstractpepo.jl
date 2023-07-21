abstract type AbstractPEPO end

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

struct PEPO{G,PType<:AbsTen{2,4},CType} <: AbstractPEPO
    state::Lattice{G,PType,Matrix{PType}}
    weights::Weights{G,CType}
end

ITC.numbertype(pepo::AbstractPEPO) = ITC.numbertype(getstate(pepo))

getstate(pepo::PEPO) = pepo.state
getweights(pepo::PEPO) = pepo.weights

getweights(pepo::PEPO, bond) = getweights(pepo)[bond]
Base.getindex(p::PEPO, bond) = getstate(p)[bond]

physicalspace(pepo::AbstractPEPO) = getindex.(codomain.(getstate(pepo)), 1)

function changeaxis(pepo::AbstractPEPO)
    pepo_permuted = permute.(getstate(pepo), Ref((1, 2)), Ref((4, 3, 6, 5)))
    return PEPO(permutedims(pepo_permuted), changeaxis(getweights(pepo)))
end

function changeaxis!(pepo_permuted::AbstractPEPO, pepo::AbstractPEPO)
    permute!.(
        getstate(pepo_permuted), permutedims(getstate(pepo)), Ref((1, 2)), Ref((4, 3, 6, 5))
    )
    return pepo_permuted
end

function setweights!(pepo::AbstractPEPO, s, bond::Bond)
    @debug "$(typeof(getweights(pepo, bond)))"
    copy!(getweights(pepo, bond), s)
    return pepo
end

getbonds(pepo::AbstractPEPO) = getbonds(getstate(pepo))

function bloch(a::Base.AbstractVecOrTuple)
    σ = pauli
    mat = σ[4]
    for j in 1:3
        mat = mat + a[j] * σ[j]
    end
    return 1 / 2 * mat
end

function pepodata(a::Base.AbstractVecOrTuple, Dw::Int, Ds::Int, De::Int, Dn::Int)
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

struct IdentityState end

function PEPO(
    a::Union{Base.AbstractVecOrTuple,IdentityState},
    i1::Int,
    i2::Int,
    physbond::S,
    eastbond::S,
    southbond::S=eastbond,
) where {S<:IndexSpace}
    return PEPO(
        a, fill(physbond, (i1, i2)), fill(eastbond, (i1, i2)), fill(southbond, (i1, i1))
    )
end

function PEPO(
    a::Base.AbstractVecOrTuple,
    physbond::AbstractMatrix,
    eastbond::AbstractMatrix,
    southbond::AbstractMatrix=eastbond;
    issymmetric=false,
)
    a = normalize([a...])
    phys = Lattice(physbond, issymmetric)

    bond_w = circshift(eastbond, (1, 0))
    bond_e = eastbond
    bond_s = southbond
    bond_n = circshift(southbond, (0, 1))

    p_data = pepodata.(Ref(a), dim.(bond_e), dim.(bond_s), dim.(bond_w), dim.(bond_n))

    tensors = @. TensorMap(
        p_data, phys * adjoint(phys), bond_e * bond_s * adjoint(bond_w) * adjoint(bond_n)
    )

    return PEPO(tensors)
end
function PEPO(
    a::IdentityState,
    physbond::AbstractMatrix,
    eastbond::AbstractMatrix,
    southbond::AbstractMatrix=eastbond;
    issymmetric=false,
)
    phys = Lattice(physbond, issymmetric)

    bond_w = circshift(eastbond, (1, 0))
    bond_e = eastbond
    bond_s = southbond
    bond_n = circshift(southbond, (0, 1))

    p_data =
        2 *
        pepodata.(Ref([0, 0, 0]), dim.(bond_e), dim.(bond_s), dim.(bond_w), dim.(bond_n))

    tensors = @. TensorMap(
        p_data, phys * adjoint(phys), bond_e * bond_s * adjoint(bond_w) * adjoint(bond_n)
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
getparent(p::AbstractUpdatablePEPO) = p.parent

# Get the tensors of the underlying parent PEPO structure
getstate(p::AbstractUpdatablePEPO) = getstate(p.parent)

# Get the evolved state of the PEPO. May or may not be up to date.
getevolved(p::AbstractUpdatablePEPO) = p.evolved_tensors

# Check if PEPO is in an evolved state (i.e. if `getevolved` outputs the current state.)
isevolved(p::AbstractUpdatablePEPO) = p.isevolved[]

# Set evolved state.
setisevolved!(p::AbstractUpdatablePEPO, bool::Bool) = p.isevolved.x = bool

_getindex(p::AbstractUpdatablePEPO, bond::Bond, ::Val{false}) = getindex(getstate(p), bond)

Base.getindex(p::AbstractUpdatablePEPO, bond::Bond) = _getindex(p, bond, Val(isevolved(p)))

getweights(pepo::AbstractUpdatablePEPO, args...) = getweights(getparent(pepo), args...)

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

function evolvedpair(p::LocalUpdatablePEPO{G,PType}) where {G,PType}
    leftpair, rightpair = getevolved(p)[currentbond(p)]
    return leftpair[1]::PType, rightpair[2]::PType
end

function _getindex(pe::LocalUpdatablePEPO, bond::Bond, ::Val{true})
    updated_bond = currentbond(pe)

    new_left, new_right = evolvedpair(pe)

    rv_left, rv_right = getstate(pe)[bond]

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
_getcurrentstate(p, ::Val{true}) = getevolved(p)
_getcurrentstate(p, ::Val{false}) = getstate(p)

_getindex(p::GlobalUpdatablePEPO, bond::Bond, ::Val{true}) = getindex(getevolved(p), bond)

struct TimeDependantPEPO{P<:AbstractUpdatablePEPO}
    x::P
    y::P
end
#=

function minimalcanonicalform!(tensor::AbsTen{2,4}; tol=1e-5, maxiter=1000)
    hor = id(domain(tensor)[1])
    ver = id(domain(tensor)[2])
    T = deepcopy(tensor)
    for t in 1:maxiter
        mcfgauge!(tensor, T, hor, ver)
        ketbra = tensor' * tensor
        @tensor begin
            p_h1[e1; e2] := ketbra[e1 s w n; e2 s w n]
            p_h2[w1; w2] := ketbra[e s w1 n; e s w2 n]
            #
            p_v1[s1; s2] := ketbra[e s1 w n; e s2 w n]
            p_v2[n1; n2] := ketbra[e s w n1; e s w n2]
        end

        h_diff = p_h1 - transpose(p_h2)
        v_diff = p_v1 - transpose(p_v2)

        err = 1 / real(tr(ketbra)) * (norm(h_diff)^2 + norm(v_diff)^2)

        if err ≤ tol^2
            return tensor
        end
        hor = exp(-1 / 8 * 1 / tr(ketbra) * h_diff) * hor
        ver = exp(-1 / 8 * 1 / tr(ketbra) * v_diff) * ver
    end
    return tensor
end

function mcfgauge!(p_out, p_in, g1, g2) #hor, ver
    g1i = permute(inv(g1), (2,), (1,))
    g2i = permute(inv(g2), (2,), (1,))
    @tensor p_out[k b; e s w n] =
        p_in[k b; ee ss ww nn] * g1[ee; e] * g2[ss; s] * g1i[ww; w] * g2i[nn; n]
    return p_out
end
=#
