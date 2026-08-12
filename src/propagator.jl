# Two-site propagators for PEPSKit's simple-update pipeline. A Liouvillian acts jointly on the ket and
# bra of a site, which PEPSKit's own two-site operators cannot express, so each site's pair is fused
# into one leg and the propagator becomes an ordinary two-site operator on `fuse(P ⊗ P')`.

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

Isometry fusing a site's physical pair into one leg, `fuse(P ⊗ P') ← P ⊗ P'`.
"""
fusephysical(P::IndexSpace) = isomorphism(fuse(P ⊗ P'), P ⊗ P')

"""
    $(TYPEDSIGNATURES)

Two-site propagator `exp(dt * generator(model, P))` on the fused physical space of each site, giving an
`(F ⊗ F) ← (F ⊗ F)` map with `F = fuse(P ⊗ P')` — the `(2, 2)` shape PEPSKit's two-site operators
require.
"""
function bondpropagator(model::AbstractTerm, P::IndexSpace, dt::Number)
    matrix = exp(dt * generator(model, P))

    # Declared in the order the generator's composite index runs, then regrouped per site.
    spaces = P ⊗ P ⊗ P' ⊗ P'
    t = vectorised_tensormap(matrix, spaces, spaces)
    t = permute(t, ((1, 3, 2, 4), (5, 7, 6, 8)))

    u = fusephysical(P)
    ud = u'
    @tensor propagator[o1 o2; i1 i2] :=
        u[o1; k1 b1] * u[o2; k2 b2] * t[k1 b1 k2 b2; q1 c1 q2 c2] *
        ud[q1 c1; i1] * ud[q2 c2; i2]

    return propagator
end

"""
    $(TYPEDSIGNATURES)

Vectorised identity in the fused basis, as a plain vector: the trace functional, whose invariance
under a propagator is trace preservation.
"""
function vectorisedidentity(P::IndexSpace)
    arr = convert(Array, fusephysical(P))
    return [sum(arr[a, k, k] for k in 1:dim(P)) for a in 1:size(arr, 1)]
end
