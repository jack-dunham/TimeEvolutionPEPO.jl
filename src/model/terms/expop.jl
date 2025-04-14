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

"""
    $(FUNCTIONNAME)(C::AbstractMatrix, B::AbstractMatrix, λ::Float64, [A = one(C)]

Return a Hamiltonian containing the exponentially-decaying non-local axial terms 
λʳ⋅C⋅A⋅...⋅A⋅B for all `r in 1,2,...`. Note, this function checks that `|λ|< 1` as otherwise 
the interaction strength diverges in the thermodynamic limit.
"""
ExpOp(C, B, λ, A=one(C)) = ExpOp(1.0, C, B, λ, A)

"""
    $(FUNCTIONNAME)(C::AbstractMatrix, B::AbstractMatrix, λ::Float64, [A = one(C)]

Return a Hamiltonian containing the exponentially-decaying non-local radial terms 
λˣλʸ⋅C⋅A⋅...⋅A⋅B for all `x,y in 1,2,...`. Note, this function checks that `|λ|< 1` as 
otherwise the interaction strength diverges in the thermodynamic limit. 
"""
ExpOp2d(C, B, λ, A=one(C)) = ExpOp2d(1.0, C, B, λ, A)
