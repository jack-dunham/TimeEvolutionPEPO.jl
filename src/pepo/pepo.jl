"""
$(TYPEDEF)

Concrete type representing a projected-entangled pair operator in the symmetric gauge with
with associated bond weights accessible with the `weights` function.

See also: [`AbstractPEPO`](@ref), [`ThermalState`](@ref), and [`weights`](@ref)

## Fields

$(FIELDS)

## Constructors

The following constructors can also be called with `PEPO{G} where {G<:AbstractUnitCellGeometry}`
to dictate the unit-cell geometry of the output `PEPO` object. This will error if any of the 
input arguments specify a unit-cell geometry the conflicts with `G`.

    PEPO(
        init::Union{T, AbstractMatrix{T}}, 
        phys::Union{S, AbstractMatrix{S}}, 
        east::Union{S, AbstractMatrix{S}},  
        [south::Union{S, AbstractMatrix{S}} = east]
    ) where {T, S<:Union{Int, IndexSpace}}

The type `T` can be one of the following:

- `NTuple{3, Float64}`: a spin-1/2 Bloch-vector
- `ThermalState`: the infinite temperature thermal density matrix
- `Matrix{<:Number}`: a density matrix
- `Array{6, <:Number}`: a tensor defining the pepo tensor for each lattice site

Similarly, `phys`, `east` and `south` can be of type `Union{S, AbstractMatrix{S}` where
`S<:Union{Int, IndexSpace}`. The size of the unit-cell of the resulting `PEPO` object is 
determined via `promote_shape` automatically through an internal call to `broadcast`. 

    PEPO(data::Union{T,AbstractMatrix{T}}) where {T<:TensorMap}

Construct a `PEPO` with with each tensor given by `data`.

"""
struct PEPO{G,PType<:TenAbs{4},CType} <: AbstractPEPO{G,PType}
    """
    The unit-cell of tensors defining the projected entangled pair operator with bond weights
    absorbed. 
    """
    data::UnitCell{G,PType,Matrix{PType}}
    """
    Bond weights on each virtual bond. Note, the `sqrt` of each weight are assumed to be pre
    absorbed in to the tensors in `data`.
    """
    weights::UnitCellBonds{G,CType,Matrix{NTuple{2,CType}}}
end

ITC.getdata(pepo::PEPO) = pepo.data

latticetype(::PEPO{G,P}) where {G,P} = UnitCell{G,P,Matrix{P}}

bondweights(pepo::PEPO) = pepo.weights
getweight(pepo::PEPO, bond) = bondweights(pepo)[bond]

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

"""
$(TYPEDEF)

Singleton type used to initialise a `PEPO` to the infinite temperature thermal density.
matrix

See also: [`PEPO`](@ref).
"""
struct ThermalState end

function pepodata(::ThermalState, phys, east, south, west, north)
    return pepositedata(I(dim(phys)), dim(east), dim(south), dim(west), dim(north))
end
function pepodata(rho::Tuple, phys, east, south, west, north)
    return pepositedata(bloch(rho), dim(east), dim(south), dim(west), dim(north))
end
function pepodata(rho::Matrix{<:Number}, phys, east, south, west, north)
    return pepositedata(rho, dim(east), dim(south), dim(west), dim(north))
end

dim2space(d::Int) = ℂ^d
dim2space(s::IndexSpace) = s

# Construct using a Bloch vector or as a thermal state
function PEPO{G}(
    init::Union{Tuple,ThermalState,Matrix{<:Number},Array{6,<:Number}},
    phys,
    east,
    south=east,
) where {G}
    initmat = Matrix{typeof(init)}(undef, 1, 1)
    initmat[1, 1] = init

    return PEPO{G}(initmat, phys, east, south)
end

# Quick constructors (no phys space and same bond space on both axis)
PEPO{G}(init, bondspace) where {G} = PEPO{G}(init, nothing, bondspace, nothing)
function PEPO{G}(init::AbstractMatrix{<:Tuple}, ::Nothing, bondspace, ::Nothing) where {G}
    return PEPO{G}(init, broadcast(_ -> ComplexSpace(2), init), bondspace, bondspace)
end
## These two methods are indentical but are required to avoid method ambiguities
function PEPO{G}(
    init::AbstractMatrix{<:Matrix{<:Number}}, ::Nothing, bondspace, ::Nothing
) where {G}
    phys = broadcast(p -> ComplexSpace(size(p, 1)), init)
    return PEPO{G}(init, phys, bondspace, bondspace)
end
function PEPO{G}(
    init::AbstractMatrix{<:Array{6,<:Number}}, ::Nothing, bondspace, ::Nothing
) where {G}
    phys = broadcast(p -> ComplexSpace(size(p, 1)), init)
    return PEPO{G}(init, phys, bondspace, bondspace)
end
##
#

function PEPO{G}(
    init::AbstractMatrix{T}, phys, east, south=east
) where {G,T<:Union{Tuple,ThermalState,Matrix{<:Number}}}
    init = UnitCell{G}(init)

    tensors = broadcast(init, pepospace(G, phys, east, south)...) do a, p, e, s, w, n
        arr = pepodata(a, p, e, s, w, n)
        ten = TensorMap(arr, p * p', e * s * w' * n')
        return ten
    end

    return PEPO{G}(tensors)
end

# Construct using an array
function PEPO{G}(init::AbstractMatrix{Array{6,<:Number}}, phys, east, south=east) where {G}
    init = UnitCell{G}(init)

    tensors = broadcast(init, pepospace(G, phys, east, south)...) do a, p, e, s, w, n
        ten = TensorMap(a, p * p', e * s * w' * n')
        return ten
    end

    return PEPO{G}(tensors)
end

# Construct using a function passed to `TensorMap`
PEPO(args...) = PEPO{Square}(args...)
function PEPO{G}(f::Union{Function,UndefInitializer}, s, h, v=h) where {G}
    return PEPO{G}(f, ComplexF64, s, h, v)
end
function PEPO{G}(f::Function, T::Type{<:Number}, s, h, v=h) where {G}
    tensors = broadcast(pepospace(G, s, h, v)...) do p, e, s, w, n
        return TensorMap(f, T, p * p', e * s * w' * n')
    end
    return PEPO(UnitCell{G}(tensors))
end
function PEPO{G}(f::UndefInitializer, T::Type{<:Number}, s, h, v=h) where {G}
    tensors = broadcast(pepospace(G, s, h, v)...) do p, e, s, w, n
        return TensorMap{T}(f, p * p', e * s * w' * n')
    end
    return PEPO(UnitCell{G}(tensors))
end

function PEPO{G}(data::Union{AbsTen{2,4},AbstractMatrix{<:AbsTen{2,4}}}) where {G}
    return PEPO(UnitCell{G}([data;;]))
end
function PEPO(tensors::AbstractUnitCell{G,<:AbsTen{2,4}}) where {G}
    horizontal_weights = @. id(getindex(domain(tensors), 1))
    vertical_weights = @. id(getindex(domain(tensors), 2))

    weights = UnitCellBonds(horizontal_weights, vertical_weights)

    return PEPO(tensors, weights)
end

function pepospace(G, s, h, v)
    spaces = map(uc -> UnitCell{G}(dim2space.(uc)), _pepospace(s, h, v))
    if !allequal(eltype, spaces)
        @error "Cannot construct PEPO with different space types on different bonds"
    end
    return spaces
end
function _pepospace(s, h, v)
    phys = [s;;]

    east = [h;;]
    south = [v;;]
    west = circshift([h;;], (1, 0))
    north = circshift([v;;], (0, 1))

    return phys, east, south, west, north
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

"""
    $(FUNCTIONNAME)(pepo::AbstractPEPO, space::IndexSpace)

Promote the virtual bond spaces of `pepo` to `space`. This function can be used to increase
the bond dimension of `pepo`. 
"""
function expandspace(pepo::PEPO, space::IndexSpace)
    new_pepo = broadcast(pepo, Ref(space)) do p, s
        return expandspace(p, s)
    end

    new_bond_weights = broadcast(bondweights(pepo)) do w
        c = codomain(w)
        d = domain(w)

        T = scalartype(w)

        u = TensorMap(delta(T, dim(space), dim(c)), space, c)
        v = TensorMap(delta(T, dim(d), dim(space)), d, space)

        return u * w * v
    end

    for bond in eachbond(pepo)
        setweight!(pepo, new_bond_weights[bond], bond; copy=false)
    end

    return PEPO(new_pepo, new_bond_weights)
end

"""
    $(FUNCTIONNAME)(pepo::PEPO, x, y; weights=false) -> PEPO

Return a PEPO over the smaller unitcell specified by indices `x` and `y`. If `weights=true`,
the PEPO is constructed with the weights of the chosen lattice sites, otherwise they are
initilised to identities. Arguments `x` and `y` can be of type `Union{Int, UnitRange{Int}, Colon}`.
"""
function subpepo(pepo::PEPO, i1::Int, i2::Int; weights=false)
    t = UnitCell(axisbuffer(pepo)[i1:i1, i2:i2])

    if weights
        w = UnitCellBonds(weights(pepo)[i1:i1, i2:i2])
        return PEPO(t, w)
    else
        return PEPO(t)
    end
end
function subpepo(pepo::PEPO, i1, i2; kwargs...)
    # There is probably a smarter way...
    indc(i::Int) = i:i
    indc(i::Union{Colon,UnitRange}) = i

    return subpepo(pepo, indc(i1), indc(i2); kwargs...)
end

"""
    $(FUNCTIONNAME)(pepo)

Remove the symmetric structure of the underlying tensors in `pepo`. If `pepo` is already
trivial, that is `spacetype(pepo)<:ComplexSpace`, then this function acts as the identity.
"""
function unsymmetrise(pepo::PEPO)
    if spacetype(pepo) == ComplexSpace
        return pepo
    end
    newstate = broadcast(pepo.data) do symten
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

function Base.convert(::Type{PEPO{Square}}, pepo)
    news = UnitCell(ITC._remove_symmetry(pepo.data))
    neww = UnitCellBonds(UnitCell(ITC._remove_symmetry(pepo.weights.data)))
    return PEPO(news, neww)
end

# TODO: excise to ITC
dealias(data::UnitCell{Square}) = data
function dealias(data::UnitCell{SquareSymmetric})
    A = similar(ITC.getdata(data), size(data))

    foreach(eachindex(A)) do ind
        setindex!(A, deepcopy(data[ind]), ind)
    end

    return UnitCell(A)
end
