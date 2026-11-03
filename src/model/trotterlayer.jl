# One layer of a Suzuki-Trotter decomposition: a term per unit-cell site, and the axis it acts on.

"""
$(TYPEDEF)

One layer of a Suzuki-Trotter decomposition.

`data[r, c]` is anchored at site `(r, c)`: its two-local part is the coupling between `(r, c)` and its
neighbour in the `+axis` direction — east for axis 1, south for axis 2 — and its one-local part
belongs to `(r, c)` alone. Indexing wraps periodically, so a `1×1` layer is uniform and serves any
unit cell.

`axis` is 1 for the column direction, 2 for the row direction, or 0 to apply the layer on both.

# Constructors

    TrotterLayer(data::AbstractMatrix, axis::Int = 0)
    TrotterLayer(term, axis::Int = 0)
"""
struct TrotterLayer{T <: AbstractTerm}
    data::Matrix{T}
    axis::Int
    function TrotterLayer(data::AbstractMatrix, axis::Int = 0)
        axis in (0, 1, 2) || throw(ArgumentError("axis must be 0, 1 or 2, got $axis"))
        promoted = map(layerterm, data)
        if axis != 0 && any(!isaxial, promoted)
            throw(ArgumentError("a non-axial term cannot be pinned to axis $axis"))
        end
        T = mapreduce(typeof, promote_type, promoted)
        return new{T}(convert(Matrix{T}, promoted), axis)
    end
end

TrotterLayer(term, axis::Int = 0) = TrotterLayer(fill(term, 1, 1), axis)
TrotterLayer(layer::TrotterLayer) = layer

Base.size(l::TrotterLayer) = size(l.data)
Base.getindex(l::TrotterLayer, r::Int, c::Int) = l.data[mod1(r, size(l, 1)), mod1(c, size(l, 2))]
Base.size(l::TrotterLayer, d::Int) = size(l.data, d)

"""
    $(TYPEDSIGNATURES)

Scale the one-local part of every entry of a layer, leaving the two-local couplings alone.
"""
scalelayer(l::TrotterLayer, a) = TrotterLayer(map(t -> scalelocal(t, a), l.data), l.axis)
