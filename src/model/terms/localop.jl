"""
$(TYPEDEF)

Concrete type representing an ``N``-local axial terms.

# Constructors

    $(FUNCTIONNAME)(D::AbstractMatrix) -> LocalOp{1}

Construct a one-local (on-site) term `op::LocalOp{1}` such that `a * op` represents:
```math
\\hat{h}_{j} = a \\hat{D}_j
```
---
    $(FUNCTIONNAME)(C::AbstractMatrix, B::AbstractMatrix [,A=one(C), N = 2])

Construct a ``N``-local one-dimensional (axial) term  `op::LocalOp{N}` such that `a * op` represents:
```math
\\hat{h}_{j} = a \\hat{C}_{j} \\hat{A}_{j + 1} \\hat{A}_{j + 2} \\dots \\hat{B}_{j + N - 1}
```
Note, for type stabilty pass `Val(N)` instead of `N::Integer` as the final argument or use the generic `Tuple`
constructor.

---
    $(FUNCTIONNAME)(CAB::NTuple{N, <:AbstractMatrix})

Construct a ``N``-local one-dimensional (axial) term  `op::LocalOp{N}` such that `a * op` represents:
```math
\\hat{h}_{j} = a \\hat{C}_{j} \\hat{A}^{[1]}_{j + 1} \\hat{A}^{[2]}_{j + 2} \\dots \\hat{B}_{j + N - 1}
```
with ``C``, ``B``, and ``\\{A^{[\\alpha]}\\}`` given by `first(CAB)`, `last(CAB)` and `CAB[2:end-1]` respectively.
"""
struct LocalOp{N} <: AbstractSingleTerm
    a::Float64
    CAB::NTuple{N,Matrix{ComplexF64}}
    function LocalOp(a, CAB::NTuple{N}) where {N}
        c = prod(op -> norm(op, Inf), CAB)
        return new{N}(a * c, map(op -> normalize(op, Inf), CAB))
    end
end

function Base.show(io::IO, m::MIME"text/plain", op::LocalOp)
    iotemp = IOBuffer()
    strings = map(op.CAB) do mat
        show(iotemp, "text/plain", mat)
        io_temp_again = IOBuffer(String(take!(iotemp)))
        lines = readlines(io_temp_again)
        close(io_temp_again)
        return lines
    end
    close(iotemp)

    colwidth = map(str -> maximum(length, str), strings)
    coeff = " $(op.a) × "

    itr = zip(strings...)

    println(io, typeof(op), ":")
    for (row, tuple_of_lines) in enumerate(itr)
        if row == 1
            print(io, coeff)
        else
            print(io, "\n")
            print(io, " "^(length(coeff)))
        end

        print(io, "| ")
        for (col, (line, pad)) in enumerate(zip(tuple_of_lines, colwidth))
            print(io, line)
            print(io, " "^(pad - length(line)))
            if col != length(colwidth)
                if row == 1
                    print(io, " | ⊗ | ")
                else
                    print(io, " |   | ")
                end
            end
        end
        print(io, " |")
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

LocalOp(D::AbstractMatrix) = LocalOp((D,))

function LocalOp(C, B, A=one(C); l::Int=1)
    if l < 1
        if l == 0
            throw(
                ArgumentError(
                    "onsite (one-local) terms should be constructed using the " *
                    "single argument method `LocalOp(D::AbstractMatrix)`",
                ),
            )
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
onsite(f, slot::Int) = H -> 1 / 2 * f(slot == 1 ? kron(H, one(H)) : kron(one(H), H))
twolocal(f) = H -> onsite(f, 1)(H) + onsite(f, 2)(H)

struct J1J2Op
    a::Float64
    C::Matrix{ComplexF64}
    B::Matrix{ComplexF64}
    k::Float64
end

struct J1J2SymmetricOp <: AbstractSingleTerm
    a::Float64
    k::Float64
    CB::Matrix{ComplexF64}
end
