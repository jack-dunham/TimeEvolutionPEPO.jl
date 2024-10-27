struct InReverse{M}
    model::M
end

struct OnAxis{M}
    model::M
    axis::Int
end

struct Combine{M}
    model::NTuple{2, Vector{M}}
end


function (model::Combine)(G, physical, method, dt; inreverse = false, kwargs...)

    # Swap the order of the axis
  termtuple = inreverse ? reverse(model.model) : model.model

    rv = mapreduce(vcat, termtuple...) do t1, t2

        tvec1 = broadcastmodel(G, t1, physical, method; kwargs...)(dt)
        tvec2 = broadcastmodel(G, t2, physical, method; kwargs...)(dt)

        rv = map(tvec1, tvec2) do tlat1, tlat2
          rv = broadcast(tlat1, tlat2) do ten1, ten2
                @tensoropt out[ok ob ik ib; e s w n] := ten2[ok ob x y; e w] * ten1[x y ik ib; s n]
                return out
          end
          return rv
        end

        return rv
    end

    return rv
end

unwrap(model) = model
unwrap(m::Union{InReverse,OnAxis}) = unwrap(m.model)

(o::OnAxis)(space, method) = o.model(space, method)

onlattice(::Type{G}, v::Vector) where {G} = map(Lattice{G}, v)
onlattice(::Type{G}, v) where {G} = Lattice{G}(v)

# Default
function broadcastmodel(::Type, model, physical, method; kwargs...)
    rv = dt -> broadcast(physical, unwrap(model)) do space, func
        return func(space, method; kwargs...)(dt)
    end
    return rv
end
# Entry point
function broadcastmodel(G::Type{<:ITC.AbstractUnitCellGeometry}, model, physical, method; kwargs...)
    @assert model isa AbstractMatrix
    rv = broadcastmodel(eltype(model), model, physical, method; kwargs...)
    return x -> onlattice(G, rv(x))
end
function broadcastmodel(G::Type{<:ITC.AbstractUnitCellGeometry}, model::Combine, physical, method; kwargs...)
    return dt -> onlattice(G, model(G, physical, method, dt; kwargs...))
end
function broadcastmodel(G::Type{<:ITC.AbstractUnitCellGeometry}, model::InReverse, physical, method; kwargs...)
    return broadcastmodel(G, model.model, physical, method; inreverse=true, kwargs...)
end
function broadcastmodel(G::Type{<:ITC.AbstractUnitCellGeometry}, model::OnAxis, physical, method; kwargs...)
    return broadcastmodel(G,
        model.model, physical, method; axis=model.axis, inreverse=true, kwargs...
    )
end

function stepper_type(o::OnAxis, method)
    function rv(lat; kwargs...)
        return stepper_type(o.model, method)(lat; axis=o.axis, kwargs...)
    end

    return rv
end
function stepper_type(r::InReverse, method)
    function rv(lat; kwargs...)
        return stepper_type(r.model, method)(lat; inreverse=true, kwargs...)
    end

    return rv
end


abstract type ModelStyle end

struct IsLocal <: ModelStyle end
struct IsAxial <: ModelStyle end
struct IsGlobal <: ModelStyle end

ModelStyle(a) = ModelStyle(typeof(a))
ModelStyle(::Type{<:AbstractMatrix{T}}) where {T} = ModelStyle(T)
ModelStyle(::Type) = IsAxial()

processmodel(model, method) = adapt_to_method(process_model(model), method)

function process_model(model)
    vec = typeof(model)[]
    push!(vec, model)
    return process_model(vec)
end
function process_model(invec::Vector)
    outvec = map(m -> [m;;], invec)
    println(outvec)

    T = eltype(outvec)
    MS = ModelStyle(T)

    return process_model(MS, outvec)
end

# By default, need to wrap non-wrapped axial and local equations.
function process_model(::Union{IsAxial,IsLocal}, invec::Vector)
    T = eltype(invec)

    outvec = OnAxis{T}[]

    for v in invec
        push!(outvec, OnAxis(v, 1))
        push!(outvec, OnAxis(v, 2))
    end

    return outvec
end

process_model(::ModelStyle, vec::Vector) = vec

adapt_to_method(model, method) = adapt_to_method(ModelStyle(method), model)

# Default (do nothing)
adapt_to_method(::ModelStyle, model) = model

function adapt_to_method(::IsGlobal, model::Vector{<:OnAxis{M}}) where {M}

    T = eltype(model)

    terms = (T[],T[])

    for op in model
        push!(terms[op.axis], op)
    end

    return [Combine(terms)]
end
