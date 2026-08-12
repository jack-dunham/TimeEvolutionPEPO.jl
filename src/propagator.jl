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

Two-site generator of `model` as a matrix, in the vectorisation the term algebra uses: composite index
`(ket₁, ket₂, bra₁, bra₂)`, ket slowest. A `Hamiltonian` gives the thermal generator and a
`Liouvillian` the Lindblad one.
"""
function generator(model::AbstractTerm, P::IndexSpace)
    d = dim(P)
    data = zeros(ComplexF64, d^4, d^4)

    if model isa Hamiltonian
        vecf = thermalvectorise
    elseif model isa Liouvillian
        vecf = vectorise
        jumps = dissipator(model)
        length(jumps) > 0 && (data += vectorise(jumps, 2))
    else
        throw(ArgumentError("no generator defined for a model of type $(typeof(model))"))
    end

    # One-local terms are shared between the two sites of the bond, hence the halving in `twolocal`.
    length(oneterms(model)) > 0 && (data += sum(vecf, oneterms(model)))
    length(terms(model)) > 0 && (data += sum(vecf, terms(model)))

    return data
end

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
