"""
$(TYPEDEF)

Concrete type representing a single trotter layer in a trotter decomposition.

## Fields

-`data::Matrix{T}`: Each element of `data` defines the operator(s) applied to the
    corresponding lattice site.
-`axis::Int=0`: If `axis=0`, then this layer generates two layers for each axis in the
    Trotter decomposition when `isaxial(T)==true` is axial, otherwise `axis` defines the
    axis which the operators should be applied across. If `isradial(T)==true` then `axis`
    has no meaning and should be set equal to `0`.
"""
struct TrotterLayer{T<:AbstractTerm}
    data::Matrix{T}
    axis::Int
    function TrotterLayer(data::Matrix{T}, axis) where {T}
        if any(!isaxial, data) && axis != 0 
            throw(ArgumentError("cannot put non-axial term on axis $axis"))
        else
            return new{layertype(T)}(data, axis)
        end
    end
end

function TrotterLayer(data, axis = 0)
    T = layertype(typeof(data))
    mat = Matrix{T}(undef, 1, 1)
    mat[1,1] = data
    return TrotterLayer(mat, axis)
end
TrotterLayer(data::TrotterLayer) = data

Base.convert(::TrotterLayer, term) = TrotterLayer(term)
function Base.convert(::Type{<:TrotterLayer{T}}, term) where {T}
    newdata = broadcast(t -> convert(T, t), term.data)
    return TrotterLayer(newdata, term.axis)
end

layertype(::Type{<:AbstractTerm}) = Hamiltonian
layertype(::Type{<:Liouvillian}) = Liouvillian
layertype(::Type{LocalOp{1}}) = LocalOp{1}
layertype(::Type{Dissipator}) = Dissipator
layertype(::Type{<:Ising}) = Ising


Base.:*(l::TrotterLayer, a) = l.data .= l.data .* a
Base.:*(a, l::TrotterLayer) = l.data .= a .* l.data

Base.:+(l::TrotterLayer, h::LocalOp{1}) = TrotterLayer(l.data .+ Ref(h), l.axis)
function Base.:+(l1::TrotterLayer, l2::TrotterLayer)

    if !(l1.axis == 0 && l2.axis == 0)
        if l1.axis != l2.axis
            throw(ArgumentError("cannot combine operators acting on different lattice axis into the same trotter layer"))
        end
    else
        rvdata = l1.data .+ l2.data
    end
    @assert l1.axis == l2.axis
    return TrotterLayer(rvdata, l1.axis)
end

Base.zero(t::TrotterLayer) = TrotterLayer(zero.(t.data), t.axis)

Base.:+(terms::AbstractSumOfTerms, l::TrotterLayer) = +(l, terms)
function Base.:+(l::TrotterLayer, terms::AbstractSumOfTerms)
    newdata = broadcast(i -> i + terms, l.data)
    return TrotterLayer(newdata, l.axis)
end
Base.:+(l::TrotterLayer, term::AbstractTerm) = +(l, TrotterLayer(term))

# linops(l::TrotterLayer{Hamiltonian}) = nothing
# terms(l::TrotterLayer) = TrotterLayer(sum.(terms.(l.data)), l.axis)
# oneterms(l::TrotterLayer) = TrotterLayer(sum.(oneterms.(l.data)), l.axis)

unwrap(model::TrotterLayer) = model.data


changeaxis(t::TrotterLayer, axis; a = 1.0) = TrotterLayer(scalelocal.(t.data, a), axis)

function broadcastmodel(
    G::Type{<:ITC.AbstractUnitCellGeometry}, model::TrotterLayer, physical, method; inreverse=false,kwargs...
)

    return broadcastmodel(
        G, model.data, physical, method; axis=model.axis, inreverse, kwargs...
    )
end

function applykwargs(S::Type, o::TrotterLayer)
    function rv(lat; kwargs...)
        return S(lat; axis=o.axis, kwargs...)
    end
    return rv
end

# TODO: move this
struct Combine
    model::NTuple{2,TrotterLayer}
end
Combine(a,b) = Combine((a,b))

function Base.:+(c::Combine, h::AbstractTerm) 
    c1,c2 = c.model
    return Combine((c1 + h), (c2 + h))
end

function addlinops(c::Combine, linops)
    c1,c2 = c.model
    rv1 = addlinops(c1, linops)
    rv2 = addlinops(c2, linops)
    return Combine(rv1,rv2)
end

function (model::Combine)(G, physical, method, dt; inreverse=false, kwargs...)
    @info "Combining..."

    @debug "" model.model

    # Swap the order of the axis
    termtuple = inreverse ? reverse(model.model) : model.model

    t1,t2 = termtuple
    @debug "" typeof(t1) typeof(t2)

    tlat1 = broadcastmodel(G, t1, physical, method;  kwargs...)(dt)
    tlat2 = broadcastmodel(G, t2, physical, method;  kwargs...)(dt)

    rv = broadcast(tlat1, tlat2) do ten1, ten2
        @tensoropt out[ok ob ik ib; e s w n] :=
            ten2[ok ob x y; e w] * ten1[x y ik ib; s n]
        return out
    end

    return rv
end

function broadcastmodel(
    G::Type{<:ITC.AbstractUnitCellGeometry}, model::Combine, physical, method; kwargs...
)
    return dt -> onlattice(G, model(G, physical, method, dt; kwargs...))
end
function applykwargs(S::Type, ::Combine)
    function rv(lat; kwargs...)
        return S(lat; kwargs...)
    end
    return rv
end


