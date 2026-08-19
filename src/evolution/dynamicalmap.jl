# Suzuki-Trotter decomposition: a model becomes a sequence of (propagator, axis) pairs, one per layer
# per axis. `Model` and `TrotterLayer` stay parked with the lattice-broadcasting code they carry, so a
# layer here is a bare term and a decomposition is a vector of them.

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

Suzuki-Trotter decomposition of `layers` on physical space `P` with time step `dt`, as a vector of
`(propagator, axis)` pairs to be swept in order.

Each layer is a term (`Hamiltonian`, `Liouvillian` or `Ising`), or a `term => axis` pair to restrict it
to one axis. A layer without an axis is split across both, with its one-local part halved so that a
full time step applies it once per site: each of the two sweeps would otherwise apply it in full.
"""
function trotterise(layers::AbstractVector, P::IndexSpace, dt::Number)
    decomposition = Tuple{AbstractTensorMap, Int}[]

    for layer in layers
        raw, axis = layer isa Pair ? (first(layer), last(layer)) : (layer, 0)
        term = layerterm(raw)
        _checkroute(term)
        if axis == 0
            halved = scalelocal(term, 0.5)
            push!(decomposition, (bondpropagator(halved, P, dt), 1))
            push!(decomposition, (bondpropagator(halved, P, dt), 2))
        else
            push!(decomposition, (bondpropagator(term, P, dt), axis))
        end
    end

    return decomposition
end

trotterise(layer, P::IndexSpace, dt::Number) = trotterise([layer], P, dt)
