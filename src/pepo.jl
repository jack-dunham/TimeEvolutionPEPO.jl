"""
    abstract type AbstractPEPO{S<:IndexSpace} <: AbstractArray{S,2} end 

Abstract supertype of all PEPO-like objects, i.e. networks of `PEPOTensor` types.
"""
abstract type AbstractPEPO{S<:IndexSpace} <: AbstractMatrix{S} end

# TODO: Change requirement on PEPOTensor to trace = 1, check for Hermiticity. 
"""
    struct PEPOTensor{S<:IndexSpace} <: AbstractTensorMap{S,3,3} 

Concrete type containing an `AbstractTensorMap{S,3,3}` with unit norm. By convention,
the codomain is ordered (1,2,out) and the domain is ordered (3,4,in).
"""
struct PEPOTensor{
    S<:IndexSpace,
    I<:Sector,
    A<:Union{<:DenseMatrix,TensorKit.SectorDict{I,<:DenseMatrix}},
    F₁,
    F₂,
} <: AbstractPhysicalTensor{S,3,3, I, A, F₁, F₂}
    tensor::TensorMap{S,3,3,I,A,F₁,F₂}
    function PEPOTensor(
        tensor::TensorMap{S,3,3,I,A,F₁,F₂}
    ) where {
        S<:IndexSpace,
        I<:Sector,
        A<:Union{<:DenseMatrix,TensorKit.SectorDict{I,<:DenseMatrix}},
        F₁,
        F₂,
    }
        if norm(tensor) ≈ one(ComplexF64)
            return new{S,I,A,F₁,F₂}(tensor)
        else
            @warn "TensorMap does not have norm equal to 1"
            return new{S,I,A,F₁,F₂}(tensor)
        end
    end
end

"""
    PEPOTensor(D::Int, x::Real, y::Real, z::Real)
    PEPOTensor(D::Int, a::Vector{<:Real})

Construct a non-symmetric `PEPOTensor` with bond dimension `D` corresponding to a given 
Bloch vector. Each element of the Bloch vector can be provided as a seperate arguments, 
or a vector of real numbers can be provided. 
"""
function PEPOTensor(D::Int, x::Real, y::Real, z::Real)
    return PEPOTensor(bloch_pepo(D, x, y, z))
end
PEPOTensor(D::Int, a::Vector{<:Real}) = PEPOTensor(D, a...)

function PEPOTensor(bond::S, phys::S, x::Real, y::Real, z::Real) where {S<:IndexSpace}
    return PEPOTensor(bloch_pepo(bond, phys, x, y, z))
end

function PEPOTensor(bond::S, phys::S, a::Vector{<:Real}) where {S<:IndexSpace}
    return PEPOTensor(bloch_pepo(bond, phys, a...))
end

"""
    struct FinitePEPO{S<:IndexSpace} <: AbstractPEPO{S}

Concrete subtype of `AbstractPEPO` for representing finite sized square lattices of 
single-site PEPO tensors.
"""
struct FinitePEPO{S<:IndexSpace} <: AbstractPEPO{S}
    sites::Array{PEPOTensor{S},2}
    function FinitePEPO(sites::Array{PEPOTensor{S},2}) where {S<:IndexSpace}
        _check_bond_axis(sites)
        _check_bond_axis(sites; permute=true)
        return new{S}(sites)
    end
end

# FIXME: Refactor this function such that the array of tensor does not need to be copied.
# This will also allow the index of sites in the error message to be displayed more 
# easily. Solution would be to transpose the indexing rather than transposing the array.
function _check_bond_axis(sites; permute=false)
    if permute
        permutedims(sites)
    end
    N = Base.size(sites)[1] - 1  # Number of rows in lattice
    M = Base.size(sites)[2] - 1  # Number of columns in lattice
    for i in N  # Loop down the rows (from top to bottom)
        for j in M  # Loop along the columns (from left to right)
            if !(_check_bond(sites[i, j], sites[i, j + 1]))
                throw(
                    SpaceMismatch(
                        "Domain of site ($i,$j) and codomain" *
                        " of site ($i,$(j+1)) do not match",
                    ),
                )
            end
        end
    end

    return nothing
end

function _check_bond(site1, site2)
    if domain(site1.tensor)[2] == codomain(site2.tensor)[1]
        return true
    else
        return false
    end
end

"""
    struct InfinitePEPO{S<:IndexSpace} <: AbstractPEPO{S}

Concrete subtype of `AbstractPEPO` for representing infinite sized square lattices 
constructed by tessellating a single unit-cell.
"""
struct InfinitePEPO{S<:IndexSpace} <: AbstractPEPO{S}
    sites::Array{PEPOTensor{S},2}
end

function Base.getindex(A::AbstractPEPO, i1::Int, i2::Int)
    return getfield(A, :sites)[i1, i2]
end

function Base.size(A::AbstractPEPO)
    return Base.size(getfield(A, :sites))
end

function bloch_pepo_data(D::Int, a1::Real, a2::Real, a3::Real)
    if norm([a1, a2, a3]) > one(a1)
        throw(
            ArgumentError(
                "Magnitude of Bloch vector is $(norm([a1,a2,a3])). A valid Bloch vector " *
                "must have magnitude ≤ 1.",
            ),
        )
    end

    if D < 1
        throw(
            ArgumentError(
                "Bond dimension D is $(D). The bond dimension must be a positive integer " *
                " greater than or equal to 1",
            ),
        )
    end
    I = complex([1 0; 0 1])
    X = complex([0 1; 1 0])
    Y = complex([1 -im; im 0])
    Z = complex([1 0; 0 -1])
    tens = complex(zeros(D, 2, 2))
    tens[1, :, :] .= 1//2 * (I + a1 * X + a2 * Y + a3 * Z)
    @tensor begin
        A[i j d; k l dp] :=
            deepcopy(tens)[i, d, d1] *
            deepcopy(tens)[j, d1, d2] *
            deepcopy(tens)[k, d2, d3] *
            deepcopy(tens)[l, d3, dp]
    end
    A ./= maximum(abs.(A))
    return A
end

function bloch_pepo(D::Int, a1::Real, a2::Real, a3::Real)
    A = bloch_pepo_data(D, a1, a2, a3)
    return TensorMap(A, ℂ^D * ℂ^D * ℂ^2, ℂ^D * ℂ^D * ℂ^2)
end

bloch_pepo(D::Int, a::Vector{<:Real}) = bloch_pepo(D, a...)

function bloch_pepo(
    bond::S, phys::S, a1::Real, a2::Real, a3::Real
) where {S<:GradedSpace}
    A = bloch_pepo_data(dim(bond), a1, a2, a3)
    return TensorMap(A, bond * bond * phys, bond * bond * phys)
end

# TODO: structure source file.
