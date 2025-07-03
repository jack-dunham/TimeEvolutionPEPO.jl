"""
$(TYPEDEF)

Concrete type representing one-dimensional (axial) non-local between two *collinear* sites 
with exponetially decaying interaction strength with distance between the lattice sites.

# Constructors

    $(FUNCTIONNAME)(C::AbstractMatrix, B::AbstractMatrix, λ::Float64, [A = one(C)]

Constructs `op::ExpOp` such that `a * op` represents terms of the form:
```math
\\hat{h}_{j} = a \\sum_{k > j}^{\\infty} 
\\lambda^{k - j} \\hat{C}_j \\hat{A}_{j + 1}  \\dots \\hat{A}_{k - 1} \\hat{B}_{k}
```
Note, this function checks that `abs(λ)< 1` as otherwise the interaction strength diverges in 
the thermodynamic limit.
"""
struct ExpOp <: AbstractAxialTerm
    a::ComplexF64
    C::Matrix{ComplexF64}
    B::Matrix{ComplexF64}
    λ::ComplexF64
    A::Matrix{ComplexF64}
    function ExpOp(a, C, B, λ, A)
        if !(abs(λ) < 1)
            throw(
                ArgumentError(
                    "ill-defined exponential term: decay coefficient λ must satisfy |λ| < 1, got |λ| = $(abs(λ))",
                ),
            )
        end
        return new(a, copy(C), copy(B), λ, copy(A))
    end
end
ExpOp(C, B, λ, A=one(C)) = ExpOp(1.0, C, B, λ, A)

"""
$(TYPEDEF)

Concrete type representing two-dimensional non-local between two sites interactions strength
decaying as a function of the Manhatten distance.

# Constructors

    $(FUNCTIONNAME)(C::AbstractMatrix, B::AbstractMatrix, λ::Float64, [A = one(C)]

Constructs `op::ExpOp2d` such that `a * op` represents terms of the form:
```math
\\hat{h}_{j} = a \\sum_{x = -\\infty, y \\geq y_j}^{\\infty}
\\lambda^{| x | +| y|}
\\hat{C}_{\\mathbf{r}_j} 
\\hat{A}_{\\mathbf{r}_j + \\mathbf{y} } \\dots \\hat{A}_{\\mathbf{r}_j + y \\mathbf{y} } 
\\hat{A}_{\\mathbf{r}_j +  \\mathbf{x}  + y \\mathbf{y} } \\dots
\\hat{A}_{\\mathbf{r}_{j} + (x - 1) \\mathbf{x} + y\\mathbf{y} }
\\hat{B}_{\\mathbf{r}_{j} + x \\mathbf{x} + y\\mathbf{y} }
```
where ``\\mathbf{r}_j = x_j \\mathbf{x} + y_j \\mathbf{y}``.
Note, this constructor checks that `abs(λ)< 1` as otherwise the interaction strength diverges in 
the thermodynamic limit. 
"""
struct ExpOp2d <: AbstractManhattanTerm
    a::Float64
    C::Matrix{ComplexF64}
    B::Matrix{ComplexF64}
    λ::Float64
    A::Matrix{ComplexF64}
    function ExpOp(a, C, B, λ, A)
        if !(abs(λ) < 1)
            throw(
                ArgumentError(
                    "ill-defined exponential term: decay coefficient λ must satisfy |λ| < 1, got |λ| = $(abs(λ))",
                ),
            )
        end
        return new(a, copy(C), copy(B), λ, copy(A))
    end
end
ExpOp2d(C, B, λ, A=one(C)) = ExpOp2d(1.0, C, B, λ, A)
