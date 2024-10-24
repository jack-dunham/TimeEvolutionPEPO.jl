struct InReverse end

struct OnAxis{M}
    model::M
    axis::Int
end

unwrap(model) = model
unwrap(m::OnAxis) = m.model

(o::OnAxis)(space, method) = o.model(space, method)
stepper_type(o::OnAxis, method) =  lat -> stepper_type(o.model, method)(lat, o.axis)

struct Combine{M}
    model::Vector{M}
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
        push!(outvec, OnAxis(v,1))
        push!(outvec, OnAxis(v,2))
    end

    return outvec
end

process_model(::ModelStyle, vec::Vector) = vec

adapt_to_method(model, method) = adapt_to_method(ModelStyle(method), model)

# Default (do nothing)
adapt_to_method(::ModelStyle, model) = model

adapt_to_method(::IsGlobal, model::Vector{<:OnAxis}) = [Combine(model)]
