struct LocalOp{N} <: AbstractSingleTerm
    a::Float64
    CAB::NTuple{N,Matrix{ComplexF64}}
    function LocalOp(a, CAB::NTuple{N}) where {N}
        c = prod(op -> norm(op, Inf), CAB)
        new{N}(a * c, map(op -> normalize(op, Inf), CAB))
    end
end

terms(t::LocalOp{1}) = ()

oneterms(t::AbstractTerm) = LocalOp{1}[]
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
    $(FUNCTIONNAME)(D::AbstractMatrix)

Return a Hamiltonian containing the one-local term `D`.
"""
LocalOp(D::AbstractMatrix) = LocalOp((D,))
"""
    $(FUNCTIONNAME)(C::AbstractMatrix, B::AbstractMatrix, [A=one(C)]; l::Int=0)

Return a Hamiltonian containing the local axial term C⋅A⋅…⋅A⋅B where `l` defines the number
of A operators. The default `l=0` defines a nearest-neighbour term, for example.
"""
function LocalOp(C, B, A=one(C); l::Int=0)
    terms = Memory{Matrix{ComplexF64}}(undef, l + 2)

    terms[begin] = C

    for i in 1:l
        terms[1 + i] = A
    end

    terms[end] = B

    return LocalOp(Tuple(terms))
end
"""
    $(FUNCTIONNAME)(CAB::Tuple)

Return a Hamiltonian containing the local axial term C⋅A₁⋅A₂⋅…⋅B where `C = first(CAB)`,
`B = last(CAB)`, `A₁ = CAB[2]`, `A₂ = CAB[3]`, etc.
"""
LocalOp(CAB::NTuple{N,AbstractMatrix}) where {N} = LocalOp(1.0, CAB)
LocalOp(D::NTuple{1, AbstractMatrix}) = LocalOp(1.0, D)

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
    return H -> - 0.5 * (kron(H, one(H)) + kron(one(H), transpose(H)))
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

