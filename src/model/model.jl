# A model is an ordered list of Suzuki-Trotter layers, swept in order once per time step.

"""
$(TYPEDEF)

An ordered list of [`TrotterLayer`](@ref)s, exponentiated and swept in order once per time step.
Terms in the same layer are exponentiated together rather than Trotterised apart.

The type parameter is `Hamiltonian`, `Liouvillian` or `Ising`; a `Model{Hamiltonian}` rejects
dissipative terms.

# Constructors

    Model(layers...)
    Model(layers::AbstractVector)
    Model{T}()

See also: [`distribute!`](@ref), [`trotterise`](@ref)
"""
struct Model{T <: AbstractTerm}
    layers::Vector{TrotterLayer}
end

# A term's own type varies by concrete shape within a family; `layers` holds them unparametrized and
# `modelkind` recovers the family for the `push!`/`distribute!` checks.
modelkind(::Type{<:Hamiltonian}) = Hamiltonian
modelkind(::Type{<:Liouvillian}) = Liouvillian
modelkind(::Type{<:Ising}) = Ising

function Model(layers::AbstractVector)
    wrapped = map(TrotterLayer, layers)
    T = mapreduce(l -> modelkind(eltype(l.data)), promote_type, wrapped)
    return Model{T}(wrapped)
end

Model(layers...) = Model(collect(layers))
Model{T}() where {T} = Model{T}(TrotterLayer[])

Base.length(model::Model) = length(model.layers)
Base.iterate(model::Model, state = 1) = iterate(model.layers, state)
Base.getindex(model::Model, i) = model.layers[i]

function Base.show(io::IO, ::MIME"text/plain", model::Model{T}) where {T}
    n = length(model)
    return print(io, "Model{$T} with $n Trotter layer", n == 1 ? "" : "s")
end

function _checkfamily(model::Model{T}, l::TrotterLayer) where {T}
    modelkind(eltype(l.data)) <: T || throw(
        ArgumentError(
            "cannot add a term of type $(eltype(l.data)) to a $(typeof(model)); " *
                "construct a Model{$(promote_type(T, modelkind(eltype(l.data))))} instead"
        ),
    )
    return nothing
end

function Base.push!(model::Model{T}, layer) where {T}
    l = TrotterLayer(layer)
    _checkfamily(model, l)
    push!(model.layers, l)
    return model
end

"""
    $(TYPEDSIGNATURES)

Add `term` to every existing layer of `model`, scaled by `1 / length(model)`, so that a full time
step applies it once in total. Use `push!` instead to give the term a layer of its own.
"""
function distribute!(model::Model, term)
    n = length(model)
    n > 0 || throw(ArgumentError("cannot distribute a term over a model with no layers"))
    _checkfamily(model, TrotterLayer(term))
    share = (1 / n) * term
    # Build every replacement before assigning any, so a layer that rejects the axial merge
    # leaves `model` untouched rather than partially replaced.
    replaced = [TrotterLayer(map(t -> t + share, layer.data), layer.axis) for layer in model.layers]
    model.layers .= replaced
    return model
end
