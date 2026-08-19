# Two-site propagators. A Liouvillian acts jointly on the ket and bra of a site, so the propagator
# carries all four physical legs of a bond and the ket and bra stay separate throughout.

"""
    $(TYPEDSIGNATURES)

Wrap the matrix of a vectorised superoperator as a `TensorMap` on `cod ← dom`. The composite index of
such a matrix runs ket-slowest (row major) whereas `reshape` fills the fastest axis first, so the legs
of each group arrive reversed and are permuted back here.
"""
function vectorised_tensormap(mat::AbstractMatrix, cod::TensorSpace, dom::TensorSpace)
    nc, nd = length(dims(cod)), length(dims(dom))
    perm = (reverse(1:nc)..., reverse((nc + 1):(nc + nd))...)
    d = dims(cod * dom)
    arr = permutedims(reshape(mat, map(i -> d[i], perm)), perm)
    return TensorMap(arr, cod, dom)
end

"""
    $(TYPEDSIGNATURES)

Two-site generator of `anchor` and `neighbour` as a matrix, in the vectorisation the term algebra
uses: composite index `(ket₁, ket₂, bra₁, bra₂)`, ket slowest. A `Hamiltonian` gives the thermal
generator and a `Liouvillian` the Lindblad one.
"""
function generator(anchor::AbstractTerm, neighbour::AbstractTerm, P::IndexSpace)
    d = dim(P)
    data = zeros(ComplexF64, d^4, d^4)

    vecf = _vectorisation(anchor)
    vecf === _vectorisation(neighbour) || throw(
        ArgumentError(
            "a bond cannot mix a thermal and a Lindblad generator: " *
                "got $(typeof(anchor)) and $(typeof(neighbour))"
        ),
    )

    # Two-local terms belong to the bond and come from the anchor; one-local terms belong to a site.
    length(terms(anchor)) > 0 && (data += sum(vecf, terms(anchor)))

    for (slot, site) in ((1, anchor), (2, neighbour))
        for h in oneterms(site)
            data += onsite(vecf(LocalOp{2}), slot)(Matrix{ComplexF64}(h))
        end
        jumps = dissipator(site)
        isnothing(jumps) && continue
        for L in terms(jumps)
            data += onsite(_dissipatorvec, slot)(L)
        end
    end

    return data
end

_vectorisation(::Hamiltonian) = thermalvectorise
_vectorisation(::Liouvillian) = vectorise
_vectorisation(t) = throw(ArgumentError("no generator defined for a model of type $(typeof(t))"))

generator(model::AbstractTerm, P::IndexSpace) = generator(model, model, P)

generator(t::Ising, P::IndexSpace) = generator(_equation(t), P)
generator(a::Ising, b::Ising, P::IndexSpace) = generator(_equation(a), _equation(b), P)

"""
    $(TYPEDSIGNATURES)

Two-site propagator `exp(dt * generator(model, P))` as an `AbsTen{4,4}` on the physical legs of a bond,
ordered `(ket₁, bra₁, ket₂, bra₂)` in both the codomain and the domain.
"""
function bondpropagator(model::AbstractTerm, P::IndexSpace, dt::Number)
    matrix = exp(dt * generator(model, P))

    # Declared in the order the generator's composite index runs, then regrouped per site.
    spaces = P ⊗ P ⊗ P' ⊗ P'
    t = vectorised_tensormap(matrix, spaces, spaces)

    return permute(t, ((1, 3, 2, 4), (5, 7, 6, 8)))
end
