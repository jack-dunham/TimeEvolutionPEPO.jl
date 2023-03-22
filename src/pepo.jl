struct Weights{L,CType <: AbstractOnLattice{L,<:AbsTen{1,1}}}
    horizontal::CType
    vertical::CType
end
mutable struct IsAbsorbed
    horizontal::Bool
    vertical::Bool
end

isabsorbed(a::IsAbsorbed) = a.horizontal && b.vertical

struct PEPO{L,PType<:AbstractOnLattice{L,<:AbsTen{3,3}}, CType}
    tensors::PType
    weights::Weights{L,CType}
    istransposed::Ref{Bool}
    isabsorbed::IsAbsorbed
end

function transpose!(pepo::PEPO)
    tens = gettensors(pepo)
    permutedims!(tens, permute.(tens, Ref((2,1,3)),Ref((5,4,6))), (2,1))
    pepo.istransposed[] = !(istransposed(pepo))
    return pepo
end

istransposed(pepo::PEPO) = pepo.istransposed[]

_flipaxis(istransposed::Bool, axis::Int) = mod(istransposed + axis, 1:2)
_flipaxis(istransposed::Bool, axis::Symbol) = axis

gettensors(P::PEPO) = P.tensors
getweights(P::PEPO) = P.weights
getweights(pepo::PEPO, axis) = getfield(getweights(P), _flipaxis(istransposed(pepo), axis))



# Return the IsAbsorbed object
getisabsorbed(P::PEPO) = P.isabsorbed
@doc raw"""
    isabsorbed(pepo::PEPO)
    isabsorbed(pepo::PEPO, axis)

Returns `true` if all the weights of the state `pepo` have been absorbed into the tensors;
returning `false` otherwise. Alternatively, return the same but along a single axis only, where
`axis` is either `:horizontal` or `:vertical`. The integers `1` and `2` can be used in place of 
the respective axis symbols with the added functionality that the returned axis is swapped
is `istransposed == true`.
"""
isabsorbed(P::PEPO) = isabsorbed(getisabsorbed(P))
isabsorbed(pepo::PEPO, axis) = getfield(getisabsorbed(pepo), _flipaxis(istransposed(pepo), axis))

function setabsorbed!(pepo::PEPO, axis, value::Bool) 
    setfield!(getisabsorbed(pepo), _flipaxis(istransposed(pepo),axis), value)
    return pepo
end
function setabsorbed!(pepo::PEPO, v::Tuple{Bool,Bool}) 
    setabsorbed!(pepo,1,v[1])
    setabsorbed!(pepo,2,v[2])
    return pepo
end

normalize!(P::PEPO, axis) = normalize!.(getweights(P,axis))
function normalize!(pepo::PEPO)
    normalize!(P::PEPO,1)
    normalize!(P::PEPO,2)
end

function weights_to_identity!(P::PEPO)
    w = getweights(P)
    @. w = id(codomain(w))
    setabsorbed!(P, true)
    return P
end

setweights!(pepo::PEPO, id::UniformScaling) = weights_to_identity!(P)
function setweights!(pepo::PEPO{L, PType, CType}, weights::CType) where {L,PType, CType}
    w = getweights(pepo)
    w .= weights
    setabsorbed!(pepo,false)
    return pepo
end

function getlinks(pepo)
    Nx,Ny = size(pepo)
    links_x = Vector{Tuple{CartesianIndex, CartesianIndex}}()
    links_y = Vector{Tuple{CartesianIndex, CartesianIndex}}()
    rx = 1:Nx
    rx = 1:Ny
    for x in rx
        push!(links_x,( CartesianIndex(x,y), CartesianIndex(mod(x + 1,rx),y) ) )
    end
    for y in ry
        push!(links_y,( CartesianIndex(x,y), CartesianIndex(x,mod(y +1, ry)) ) )
    end
    return links_x, links_y
end


@doc raw"""
    absorbweights!(P::PEPO)

Absorb the weights of the PEPO `P` into the state tensors, thus setting all weights to
identities.
"""
function absorbweights!(P::PEPO, axis)
end

_absorbweights(t, w,s,e,n)  = _absorbweights!(similar(t), t,w,s,e,n)
function _absorbweights!(tdst, tsrc, w,s,e,n)
    @tensoropt tdst[wo so k; eo no b] = tsrc[w s k; e n b] * w[wo ;w] * s[so;s] * e[e;eo] * n[n;no]
    return tdst
end

function getstate!(pepo::PEPO)
    if isabsorbed(pepo)
        return pepo
    else
        gettensors(absorbweights!(P))
    end
end

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

    tw = zeros(ComplexF64, 2, Dw, 2)
    ts = zeros(ComplexF64, 2, Ds, 2)
    tw[:, 1, :] = ts[:, 1, :] = m

    te = zeros(ComplexF64, 2, 2, De)
    tn = zeros(ComplexF64, 2, 2, Dn)
    te[:, :, 1] = tn[:, :, 1] = m

    @tensor p[2 3 1; 5 6 4] := tw[s1 2; 4] * ts[s2 3; s1] * te[s3; s2 5] * tn[1; s3 6]

    return p
end

function PEPO(a::AbstractVector, lattice, eastbond, southbond=eastbond)
    bond_w = circshift(eastbond,(1,0))
    bond_e = eastbond
    bond_s = southbond
    bond_n = circshift(southbond,(0,1))

    p_data = @. pepodata(Ref(a), dim(bond_w), dim(bond_s), dim(bond_e), dim(bond_n))
    tensors = @. TensorMap(p_data, bond_w * bond_s * localspace, bond_e * bond_n * localspace)

    return PEPO(tensors)
end

function PEPO(tensors::AbstractOnLattice{<:AbstractLattice, <:AbsTen{3,3}})
    horizontal_weights = @. id(getindex(domain(tensors),1))
    vertical_weights = @. id(getindex(codomain(tensors),2))

    weights = Weights(horizontal_weights, vertical_weights)

    return PEPO(tensors, weights, IsAbsorbed(true,true))
end
