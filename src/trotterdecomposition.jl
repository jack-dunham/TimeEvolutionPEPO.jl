# Suzuki-Trotter decomposition: a model becomes a sequence of (propagator, axis) pairs, one per layer
# per axis. `Model` and `TrotterLayer` stay parked with the lattice-broadcasting code they carry, so a
# layer here is a bare term and a decomposition is a vector of them.

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
        term, axis = layer isa Pair ? (first(layer), last(layer)) : (layer, 0)
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
