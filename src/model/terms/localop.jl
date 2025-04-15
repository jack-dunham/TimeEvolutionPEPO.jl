"""
$(TYPEDEF)

Concrete type representing an ``l``-local axial term of the form:
```math
\\sum_{i = -\\infty}^{\\infty} a C_{i} A^{[1]}_{i + 1} A^{[2]}_{i + 2} \\cdots B_{i + l}
```
where ``i`` is a sum of lattice sites on a single axis, and ``l \\geq 0`` defines the distance
between the ``C`` and ``B`` operators. If ``l = 0``, then ``C \\cdot B = D``, and the term has the form
```math
\\sum_{i = -\\infty}^{\\infty} a D_{i}
```
The bulk operators ``A^{[\\alpha]}`` can be distinct in general, however a common scenario is to have
``A^{[\\alpha]} = \\mathbf{I}`` for all ``\\alpha`` (default behaviour).
"""
struct LocalOp{N} <: AbstractSingleTerm
    a::Float64
    CAB::NTuple{N,Matrix{ComplexF64}}
    function LocalOp(a, CAB::NTuple{N}) where {N}
        c = prod(op -> norm(op, Inf), CAB)
        return new{N}(a * c, map(op -> normalize(op, Inf), CAB))
    end
end

terms(::LocalOp{1}) = ()

oneterms(::AbstractTerm) = LocalOp{1}[]
oneterms(t::LocalOp{1}) = LocalOp{1}[t]

function Base.hash(term::LocalOp, h::UInt)
    for op in term.CAB
        h = hash(op, h)
    end
    h = hash(term.a, h)
    return hash(typeof(term), h)
end

Base.:(==)(l1::LocalOp, l2::LocalOp) = hash(l1) == hash(l2)
Base.:(≈)(l1::LocalOp{N}, l2::LocalOp{N}) where {N} = all(l1.CAB .≈ l2.CAB) && l1.a ≈ l2.a

"""
    $(FUNCTIONNAME)(D::AbstractMatrix) -> LocalOp{1}

Construct the onsite (one-local) term ``D = `` `D`
"""
LocalOp(D::AbstractMatrix) = LocalOp((D,))
"""
    $(FUNCTIONNAME)(C::AbstractMatrix, B::AbstractMatrix, [A=one(C)]; l::Int=0)

Construct a local axial term as a `LocalOp` with ``A^{[1]} = A^{[2]} =`` `A`
"""
function LocalOp(C, B, A=one(C); l::Int=1)
    
    if l < 1
        if l == 0
            throw(
                ArgumentError(
                "onsite (one-local) terms should be constructed using the " *
                    "single argument method `LocalOp(D::AbstractMatrix)`"
                        ))
        else
            throw(ArgumentError("distance `l` must be positive; got l = $l"))
        end
    end

    terms = Memory{Matrix{ComplexF64}}(undef, l + 1)

    terms[begin] = C

    for i in 1:(l - 1)
        terms[1 + i] = A
    end

    terms[end] = B

    return LocalOp(Tuple(terms))
end
"""
    $(FUNCTIONNAME)(CAB::Tuple)

Construct a general local axial term  with ``C = `` `first(CAB)`, ``B = `` `last(CAB)` and
``A^{[\\alpha]} = `` `CAB[2:end-1][α]`
"""
LocalOp(CAB::NTuple{N,AbstractMatrix}) where {N} = LocalOp(1.0, CAB)
LocalOp(D::NTuple{1,AbstractMatrix}) = LocalOp(1.0, D)

term_to_matrix(A::LocalOp) = A.a * foldl(kron, A.CAB)
term_to_matrix(A::LocalOp{1}) = A.a * only(A.CAB)

function Base.getproperty(A::LocalOp{N}, name::Symbol) where {N}
    if N == 1
        if name === :D
            return only(A.CAB)
        end
    else
        if name === :C
            return first(A.CAB)
        elseif name === :B
            return last(A.CAB)
        elseif name === :A
            return A.CAB[2:(N - 1)]
        end
    end
    return getfield(A, name)
end

thermalvectorise(term) = thermalvectorise(typeof(term))(Matrix(term))

vectorise(::Type{LocalOp{1}}) = twolocal(vectorise(LocalOp{2}))
thermalvectorise(::Type{LocalOp{1}}) = twolocal(thermalvectorise(LocalOp{2}))

# Vectorise the superoperators; we do this the non-julian way (row major) as the rest
# of the code assumes (ket, bra) ordering (whoops).
function vectorise(::Type{LocalOp{2}})
    return H -> -im * (kron(H, one(H)) - kron(one(H), transpose(H)))
end

function thermalvectorise(::Type{LocalOp{2}})
    return H -> -0.5 * (kron(H, one(H)) + kron(one(H), transpose(H)))
end

# We need both local terms that have support on two sites, and therefore need to divide
# by two to avoid double counting in the brickwork pattern
twolocal(f) = H -> 1 / 2 * (f(kron(one(H), H)) + f(kron(H, one(H))))

struct J1J2Op
    a::Float64
    C::Matrix{ComplexF64}
    B::Matrix{ComplexF64}
    k::Float64
end
