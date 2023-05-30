abstract type AbstractPEPO end

struct Weights{L,CType<:AbsTen{1,1}}
    horizontal::OnLattice{L,CType,Matrix{CType}}
    vertical::OnLattice{L,CType,Matrix{CType}}
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
    return Weights(transpose(weights.vertical), transpose(weights.horizontal))
end

# Is transposed is a quasi-lazy way of swapping primary axis that does not 
# construct a new PEPO object, or change dims of data. Instead we just transpose the virtual 
# pepo bonds and set `istransposed` equal to `true`. Note the distinction to actually transpose
# the lattice.
struct PEPO{L,PType<:AbsTen{2,4},CType} <: AbstractPEPO
    state::OnLattice{L,PType,Matrix{PType}}
    weights::Weights{L,CType}
    istransposed::Ref{Bool}
end

lattice(p::AbstractPEPO) = lattice(getstate(p))

const TransposedPEPO{L<:TransposedLattice} = PEPO{L}

Base.getindex(p::PEPO, bond) = getstate(p)[bond]

function changeaxis(pepo::AbstractPEPO)
    pepo_permuted = permute.(getstate(pepo), Ref((1, 2)), Ref((4, 3, 6, 5)))
    return PEPO(transpose(pepo_permuted), changeaxis(getweights(pepo)), Ref(true))
end
function changeaxis!(pepo_permuted::AbstractPEPO, pepo::AbstractPEPO)
    permute!.(
        getstate(pepo_permuted), transpose(getstate(pepo)), Ref((1, 2)), Ref((4, 3, 6, 5))
    )
    return pepo_permuted
end

istransposed(pepo::PEPO) = pepo.istransposed[]

_flipaxis(istransposed::Bool, axis::Int) = mod(istransposed + axis, 1:2)
_flipaxis(istransposed::Bool, axis::Symbol) = axis

getstate(pepo::PEPO) = pepo.state

getweights(pepo::PEPO) = pepo.weights
getweights(pepo::PEPO, bond) = getweights(pepo)[bond]

function setweights!(pepo::AbstractPEPO, s, bond::Bond)
    @debug "$(typeof(getweights(pepo, bond)))"
    copy!(getweights(pepo, bond), s)
    return pepo
end

getbonds(pepo::PEPO) = getbonds(getstate(P))

function Base.permutedims!(p::AbsTen{3,3}, perm::NTuple{6,Int})
    p.tensor = _permute(p.tensor, perm)
    return p
end

function bloch(a::Vector)
    σ = pauli
    mat = σ[4]
    for j in 1:3
        mat = mat + a[j] * σ[j]
    end
    return 1 / 2 * mat
end

function pepodata(a::Vector, Dw::Int, Ds::Int, De::Int, Dn::Int)
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

function PEPO(
    lattice, a::Base.AbstractVecOrTuple, eastbond::S, southbond::S=eastbond
) where {S<:IndexSpace}
    return PEPO(lattice, a, fill(eastbond, lattice), fill(southbond, lattice))
end

function PEPO(
    lattice::AbstractLattice,
    a::Base.AbstractVecOrTuple,
    eastbond::AbstractOnLattice,
    southbond::AbstractOnLattice,
)
    bond_w = circshift(eastbond, (1, 0))
    bond_e = eastbond
    bond_s = southbond
    bond_n = circshift(southbond, (0, 1))

    p_data = pepodata.(Ref(a), dim.(bond_e), dim.(bond_s), dim.(bond_w), dim.(bond_n))

    tensors = @. TensorMap(
        p_data,
        lattice * adjoint(lattice),
        bond_e * bond_s * adjoint(bond_w) * adjoint(bond_n),
    )

    return PEPO(tensors)
end

function PEPO(tensors::AbstractOnLattice{<:AbstractLattice,<:AbsTen{2,4}})
    horizontal_weights = @. id(getindex(domain(tensors), 1))
    vertical_weights = @. id(getindex(domain(tensors), 2))

    weights = Weights(horizontal_weights, vertical_weights)

    return PEPO(tensors, weights, Ref(false))
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

### Evolving PEPO's, used to pre-allocate the arrays that will store the intermediate stage
# where the pepo is evolved

abstract type AbstractEvolvingPEPO <: AbstractPEPO end

# Get the underlying PEPO.
getparent(p::AbstractEvolvingPEPO) = p.parent

# Get the tensors of the underlying PEPO structure
getstate(p::AbstractEvolvingPEPO) = getstate(p.parent)

# Get the evolved state of the PEPO. May or may not be up to date.
getevolved(p::AbstractEvolvingPEPO) = p.evolved_tensors

# Check if PEPO is in an evolved state (i.e. if `getevolved` outputs the current state.)
isevolved(p::AbstractEvolvingPEPO) = p.isevolved[]

# Set evolved state.
setisevolved!(p::AbstractEvolvingPEPO, bool::Bool) = p.isevolved.x = bool

_getindex(p::AbstractEvolvingPEPO, bond::Bond, ::Val{false}) = getindex(getstate(p), bond)

Base.getindex(p::AbstractEvolvingPEPO, bond::Bond) = _getindex(p, bond, Val(isevolved(p)))

getweights(pepo::AbstractEvolvingPEPO, args...) = getweights(getparent(pepo), args...)

struct LocallyEvolvingPEPO{L,PType,CType} <: AbstractEvolvingPEPO
    parent::PEPO{L,PType,CType}
    evolved_tensors::OnLattice{L,Tuple{PType,PType},Matrix{Tuple{PType,PType}}}
    bond::Ref{Bond}
    isevolved::Ref{Bool}
end

currentbond(p::LocallyEvolvingPEPO) = p.bond[]
setcurrentbond!(p::LocallyEvolvingPEPO, bond::Bond) = p.bond.x = bond
function evolvedpair(p::LocallyEvolvingPEPO)
    leftpair, rightpair = getevolved(p)[currentbond(p)]
    return leftpair[1], rightpair[2]
end

function _getindex(pe::LocallyEvolvingPEPO, bond::Bond, ::Val{true})
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

struct GloballyEvolvingPEPO{L,PType,CType} <: AbstractEvolvingPEPO
    parent::PEPO{L,PType,CType}
    evolved_tensors::OnLattice{L,PType,Matrix{PType}}
    isevolved::Ref{Bool}
end

# GloballyEvolvingPEPO(pepo::PEPO) = GloballyEvolvingPEPO(pepo, deepcopy(getstate(pepo)))

getcurrentstate(p::GloballyEvolvingPEPO) = _getcurrentstate(p, Val(isevolved(p)))
_getcurrentstate(p, ::Val{true}) = getevolved(p)
_getcurrentstate(p, ::Val{false}) = getstate(p)

_getindex(p::GloballyEvolvingPEPO, bond::Bond, ::Val{true}) = getindex(getevolved(p), bond)

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
