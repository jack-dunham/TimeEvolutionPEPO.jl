const UngaugedInfiniteMPS{Nx, D<:AbsTen{2,1}} = Lattice{MPS, Infinite{Nx,1}, D}

abstract type AbstractMatrixProductState{L<:RowLatticeStyle,M<:AbsTen{2,1},C<:AbsTen{1,1}} end

const AbstractMPS = AbstractMatrixProductState
const AbstractInfiniteMPS{Nx,M,C} = AbstractMatrixProductState{UnitCell{Nx,1},M,C}

struct MatrixProductState{L,M<:AbsTen{2,1},CT<:AbsTen{1,1}} <: AbstractMPS{L,M,CT}
    AL::Lattice{MPS,L,M}
    C::Lattice{BondMatrix,L,CT}
    AR::Lattice{MPS,L,M}
    AC::Lattice{MPS,L,M}
    function MatrixProductState(
        AL::Lattice{MPS,L,M}, C::Lattice{BondMatrix,L,CT}, AR::Lattice{MPS,L,M}
    ) where {L<:RowLatticeStyle,S,M<:AbsTen{2,1}{S},CT<:AbsTen{1,1}{S}}
        Nx, _ = size(L) #(Nx,1)
        r = 1:Nx
        for i in r
            if !(westbond(AL[i]) == eastbond(AL[i + 1]))
                throw(SpaceMismatch(""))
            end
        end
        AC = AL * C
        return new{L,M,CT}(AL, C, AR, AC)
    end
end

function MatrixProductState(l::LatticeStyle, AL::Vector, C::Vector, AR::Vector)
    return MatrixProductState(
        Lattice(MPS(), l, AL), Lattice(MPS(), l, C), Lattice(MPS(), l, AR)
    )
end

function Base.:*(A::Lattice{MPS},C::Lattice{BondMatrix})
    return convertstyle(MPS, _AC.(A,C))
end
function Base.:*(C::Lattice{BondMatrix},A::Lattice{MPS})
    return convertstyle(MPS, _CA.(circshift(C,(1,0)),A))
end

bondtensor(A::AbstractMPS) = A.C
centraltensor(A::AbstractMPS) = A.AC
lefttensor(A::AbstractMPS) = A.AL
righttensor(A::AbstractMPS) = A.AR


westbond(A::MatrixProductState) = westbond.(lefttensor(A))
eastbond(A::MatrixProductState) = eastbond.(lefttensor(A))

numbertype(A::Type{<:MatrixProductState{L,M}}) where {L,M} = numbertype(M)

const InfiniteMPS{Nx,M,C} = MatrixProductState{Infinite{Nx,1},M,C}

@doc raw"""
    InfiniteMPS(V::Vector{<:MPSTensor})

Constructs an `InfMPS` from a vector `V` of `MPSTensor`'s representing a matrix product
state. The output `InfMPS` is the represention of the matrix product state `V` in the mixed
gauge. 
"""
function InfiniteMPS{Nx,Ny}(AL::Vector, C::Vector, AR::Vector) where {Nx,Ny}
    return MatrixProductState(Infinite{Nx,Ny}(), AL, C, AR)
end
function InfiniteMPS(A::UnitCell{MPS, Nx,1,<:AbsTen{2,1}}) where {Nx}
    AL, C, AR = mixedgauge(A)
    return MatrixProductState(AL, C, AR)
end

function InfiniteMPS{Nx}(v::AbstractVector{<:AbsTen{2,1}}) where {Nx}
    AL, C, AR = mixedgauge(Lattice(MPS(),Infinite{Nx,1}(), v))
    return MatrixProductState(AL, C, AR)
end

@doc raw"""
    InfiniteMPS(f, T::Type{<:Number}, d::TensorSpace{S}, χ::S, Nx::Int=1) where {S<:IndexSpace}

Construct a `Nx` unit cell `InfMPS` with `MPSTensor`s and `BondTensor` generated from the 
function `f`. The space of the physical index or indices is provided by `d` and the space
of the bond index is provided by `χ`. If `Nx` is not provided, then a trivial unit cell
(type `InfMPS{1}`) is assumed. 

Possibilites for `f` are the same as the `TensorMap` struct from TensorKit.jl. That is, 
`randn` and `rand` from Julia Base (alternatively one can use aliases `randnormal` and 
`randuniform` respectively, as provided by TensorKit.jl.), as well as `randisometry` 
(alternatively called `randharr`) from TensorKit.jl.
"""
function InfiniteMPS{Nx}(f, T::Type{<:Number}, χ₁::S, d::S, χ₂::S) where {Nx,S<:IndexSpace}
    tensors = (TensorMap(f, T, χ₁ * d, χ₂) for _ in 1:Nx)
    return InfiniteMPS{Nx}(tensors...)
end

numsites(::AbstractMPS{<:LatticeStyle{Nx}}) where {Nx} = Nx
Base.length(a::AbstractMPS) = numsites(a)

# The type of the bond tensor
function bondtensortype(
    ::Type{<:AbstractMPS{<:LatticeStyle,<:AbsTen{2,1},C}}
) where {C<:AbsTen{1,1}}
    return C
end
bondtensortype(m::AbstractMPS) = bondtensortype(typeof(m))

# The type of the matrix product state tensors
function mpstensortype(::Type{<:AbstractMPS{<:LatticeStyle,M}}) where {M<:AbsTen{2,1}}
    return M
end
mpstensortype(m::AbstractMPS) = mpstensortype(typeof(m))

# The type of the underlying spaces (ComplexSpace, GradedSpace{...} etc)
function TensorKit.spacetype(
    ::Type{<:AbstractMPS{<:LatticeStyle,<:AbsTen{2,1}{S}}}
) where {S<:IndexSpace}
    return S
end
TensorKit.spacetype(m::AbstractMPS) = spacetype(typeof(m))

# MultilineInfiniteMPS
const AbstractMultilineInfiniteMPS{Nx,Ny,M,C} = AbstractUnitCell{NotTensor,1,Ny,InfiniteMPS{Nx,M,C}}
const MultilineInfiniteMPS{Nx,Ny,M,C} = UnitCell{NotTensor,1,Ny,InfiniteMPS{Nx,M,C}}

function MultilineInfiniteMPS(v::Vararg{<:InfiniteMPS,Ny}) where {Ny}
    return Lattice(NotTensor(), Infinite{1,Ny}(), hcat(v...))
end

function MultilineInfiniteMPS{Nx,Ny}(
    f, T::Type{<:Number}, χ₁::S, d::S, χ₂::S
) where {S<:IndexSpace,Nx,Ny}
    mps = (InfiniteMPS{Nx}(f, T, χ₁, d, χ₂) for _ in Ny)
    return MultilineInfiniteMPS(mps...)
end


lefttensor(A::MultilineInfiniteMPS) = hcat((lefttensor(a) for a in A)...)
righttensor(A::MultilineInfiniteMPS) = hcat((righttensor(a) for a in A)...)
bondtensor(A::MultilineInfiniteMPS) = hcat((bondtensor(a) for a in A)...)
centraltensor(A::MultilineInfiniteMPS) = hcat((centraltensor(a) for a in A)...)

unpack(A::Union{AbstractMPS, MultilineInfiniteMPS}) = (lefttensor(A), bondtensor(A), righttensor(A), centraltensor(A))

westbond(A::MultilineInfiniteMPS) = hcat(westbond.(A))
