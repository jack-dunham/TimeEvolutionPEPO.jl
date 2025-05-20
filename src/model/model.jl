abstract type AbstractModel end

"""
$(TYPEDEF)
"""
struct Model{HL<:Union{Hamiltonian,Liouvillian,Ising}} <: AbstractModel
    layers::Vector{TrotterLayer{HL}}
    function Model{T}(terms::Vector) where {T}
        layers = map(TrotterLayer,terms)
        return new{T}(layers)
    end
    Model{T}() where {T} = new{T}([])
end

Model(HL::Vararg) = Model(collect(HL))

Model(H::T) where {T} = Model{layertype(T)}([H])
Model(H::Vector{T}) where {T} = Model{layertype(T)}(H)

Base.eltype(::Model{HL}) where {HL} = HL

function Base.promote_rule(::Type{<:AbstractTerm}, ::Type{<:Union{Dissipator,Liouvillian}})
    return Liouvillian
end

# Model{T}(HL::Vararg) where {T} = Model{T}(collect(HL))

Base.iterate(model::Model, state = 1) = iterate(model.layers, state)
Base.length(model::Model) = length(model.layers)

function Base.show(io::IO, ::MIME"text/plain", model::Model)
    print(io, "$(typeof(model)) with $(length(model)) Trotter layer")
    length(model) > 1 && print(io, "s")
    return nothing
end

Base.sum(t::TrotterLayer) = t

function Base.push!(model::Model{T}, layer) where {T}
    push!(model.layers, TrotterLayer(sum(layer)))
    return model
end

function Base.push!(model::Model{T}, layer::TrotterLayer) where {T}
    try
        push!(model.layers, layer)
    catch e
        if e isa MethodError
            if e.f === convert
                type, value = e.args
                if all(S -> S<:TrotterLayer{Hamiltonian}, (type, TrotterLayer{T}))
                    if value isa Union{TrotterLayer{Liouvillian},TrotterLayer{Dissipator}}
                        @error "Failed to `push!` term of type `$(typeof(value))` to `$(typeof(model))`. You should instead construct an instance of the type `Model{Liouvillian}` to add terms of type `Dissipator` and `Liouvillian`."
                    end
                end
            end
        end
        rethrow()
    end
    return model
end
function Base.push!(model::Model, layer::TrotterLayer{<:Union{Dissipator, LocalOp{1}}})

    l = length(model)

    broadcast!(+, model.layers, model.layers, 1 / l * layer)
    return model
end

function Base.append!(m1::Model, m2::Model)
    for layer in m2
        push!(m1, m2)
    end
    return m1
end
function Base.push!(model::Model, term::AbstractTerm) 
    if isexpandable(term)
        for subterm in expandterm(term)
            push!(model, expandterm(subterm))
        end
    else
        push!(model, TrotterLayer(term))
    end
    return model
end
