# Suzuki-Trotter decomposition: a model becomes a sequence of (propagators, axis) pairs, one per layer
# per axis, with a propagator matrix per layer indexed as its term matrix is.

"""
    $(TYPEDSIGNATURES)

A term as the sum type a Trotter layer holds. A bare single term becomes a `Hamiltonian`, or a
`Liouvillian` when it is dissipative; `Hamiltonian`, `Liouvillian` and `Ising` pass through.
"""
layerterm(t::Union{Hamiltonian, Liouvillian, Ising}) = t
layerterm(t::Dissipator) = convert(Liouvillian, t)
layerterm(t::AbstractTerm) = convert(Hamiltonian, t)

"""
    $(TYPEDSIGNATURES)

Throw unless every term of `t` can be exponentiated into a two-site propagator.
"""
function _checkroute(t::AbstractTerm)
    unroutable = filter(x -> !(x isa LocalOp), collect(terms(t)))
    isempty(unroutable) || throw(
        ArgumentError(
            "no propagator route for $(typeof(first(unroutable))): only nearest-neighbour " *
                "terms can be exponentiated into a two-site propagator"
        ),
    )
    return nothing
end

# An Ising coupling is nearest-neighbour by construction, and `terms` is not defined for it.
_checkroute(::Ising) = nothing

"""
    $(TYPEDSIGNATURES)

Suzuki-Trotter decomposition of `model` on physical space `P` with time step `dt`, as a vector of
`(propagators, axis)` pairs to be swept in order. `propagators` is a matrix indexed as the layer's
term matrix is: the bond anchored at `(r, c)` takes its coupling and first-slot one-local part from
`(r, c)` and its second-slot one-local part from the neighbour in the `+axis` direction.

A layer without an axis is split across both, with its one-local part halved so that a full time step
applies it once per site: each of the two sweeps would otherwise apply it in full.
"""
function trotterise(model::Model, P::IndexSpace, dt::Number)
    decomposition = Tuple{Matrix{AbstractTensorMap}, Int}[]

    for layer in model
        if layer.axis == 0
            halved = scalelayer(layer, 0.5)
            push!(decomposition, (_layerpropagators(halved, P, dt, 1), 1))
            push!(decomposition, (_layerpropagators(halved, P, dt, 2), 2))
        else
            push!(decomposition, (_layerpropagators(layer, P, dt, layer.axis), layer.axis))
        end
    end

    return decomposition
end

function _bondpropagator(layer::TrotterLayer, P::IndexSpace, dt::Number, axis::Int, r::Int, c::Int)
    anchor = layer[r, c]
    neighbour = axis == 1 ? layer[r, c + 1] : layer[r + 1, c]
    _checkroute(anchor)
    return bondpropagator(anchor, neighbour, P, dt)
end

function _layerpropagators(layer::TrotterLayer, P::IndexSpace, dt::Number, axis::Int)
    Nr, Nc = size(layer)
    return AbstractTensorMap[_bondpropagator(layer, P, dt, axis, r, c) for r in 1:Nr, c in 1:Nc]
end

trotterise(layers::AbstractVector, P::IndexSpace, dt::Number) =
    trotterise(Model(map(_aslayer, layers)), P, dt)
trotterise(layer, P::IndexSpace, dt::Number) = trotterise(Model([_aslayer(layer)]), P, dt)

# `term => axis` restricts a bare term to one axis; `Model`'s own constructor does not unpack pairs.
_aslayer(p::Pair) = TrotterLayer(first(p), last(p))
_aslayer(x) = x
