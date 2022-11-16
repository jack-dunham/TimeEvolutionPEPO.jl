@doc raw"""
struct MPSTensor{S<:IndexSpace, N₁, N₂, ...} <: AbstractPhysicalTensor

Concrete subtype of `AbstractPhysicalTensor` used to represent a single tensor in a matrix
product state. It is assumed that the left and right bond indices are positioned at `N₁` and
`N₂` respectively. Has a single field `tensor` which is of type `TensorMap{S,N₁, N₂}`.
"""
struct MPSTensor{S,N₁,N₂,T} <: AbstractPhysicalTensor{S,N₁,N₂,T}
    tensor::T
end

@doc raw"""
struct BondTensor{S<:IndexSpace, ...} <: AbstractPhysicalTensor

Concrete subtype of `AbstractPhysicalTensor` used to represent a single bond tensor in a 
matrix product state. Has a single field `tensor` which is of type `TensorMap{S,1,1}`.
"""
struct BondTensor{S,T} <: AbstractPhysicalTensor{S,1,1,T}
    tensor::T
end

function MPSTensor(
    f, T::Type{<:Number}, physcod::TensorSpace{S}, physdom::TensorSpace{S}, χ::S
) where {S<:IndexSpace}
    return MPSTensor(TensorMap(f, T, physcod * χ, physdom * χ))
end
function BondTensor(f, T::Type{<:Number}, χ::S) where {S<:IndexSpace}
    return BondTensor(TensorMap(f, T, χ, χ))
end

leftbondspace(M::MPSTensor{<:IndexSpace,N₁}) where {N₁} = codomain(M)[N₁]
rightbondspace(M::MPSTensor{<:IndexSpace,N₁,N₂}) where {N₁,N₂} = domain(M)[N₂]
bspace(M::BondTensor) = codomain(M)

@doc raw"""

    ×(C::BondTensor{S}, A::MPSTensor{S}) where {S<:IndexSpace}

Apply a matrix `C` to the left bond of a MPS tensor `A`. The tensor `A` has $N₁$ indices in
the codomain, and $N₂$ in the domain with $N = N₁ + N₂$ total indices. The function assumes
the left and right bond indices are at positions N₁ and N₂ respectively. The symbol `×` can
be typed using `\times + TAB`.
                                          
                                ┌─┐   ┌───────┐ 
                           N₁◄──│C│◄──│   A   │◄── N₂
                                └─┘   └───────┘ 
                                        │ ⋯  ▲       
                                        ▼    │     
                                        1   N₁+1     

"""
function TensorKit.:×(C::BondTensor{S}, A::MPSTensor{S}) where {S<:IndexSpace}
    return MPSTensor(_lmul(C.tensor, A.tensor))
end

@doc raw"""

    ×(A::MPSTensor{S}, C::BondTensor{S}) where {S<:IndexSpace}

Apply a matrix `C` to the right bond of a MPS tensor `A`. This function assumes the same as
the method `×(C::BondTensor{S}, A::MPSTensor{S}) where {S<:IndexSpace}`.

                                  ┌───────┐   ┌─┐   
                             N₁◄──│   A   │◄──│C│◄── N₂
                                  └───────┘   └─┘   
                                    │ ⋯  ▲       
                                    ▼    │     
                                    1   N₁+1     

"""
function TensorKit.:×(A::MPSTensor{S}, C::BondTensor{S}) where {S<:IndexSpace}
    return MPSTensor(_rmul(A.tensor, C.tensor))
end
# struct AdjointMPSTensor{T<:TensorKit.AdjointTensorMap} <: AbstractPhysicalTensor
#     tensor::T
# end

function bondind(::MPSTensor{<:IndexSpace,N₁,N₂}) where {N₁,N₂}
    return N₁, N₂
end

#TODO: This needs some more though. MPS work differently to MPOs etc due to gauging

abstract type AbstractMatrixProductState{L<:RowLatticeSize,M<:MPSTensor,C<:BondTensor} end

const AbstractMPS = AbstractMatrixProductState

struct MatrixProductState{L,Mt<:MPSTensor,Ct<:BondTensor} <: AbstractMPS{L,Mt,Ct}
    AL::TensorLattice{L,Mt}
    C::TensorLattice{L,Ct}
    AR::TensorLattice{L,Mt}
    AC::TensorLattice{L,Mt}
    function MatrixProductState(
        AL::TensorLattice{L,Mt}, C::TensorLattice{L,Ct}, AR::TensorLattice{L,Mt}
    ) where {L,S,Mt<:MPSTensor{S},Ct<:BondTensor{S}}
        Nx, _ = size(L)
        r = 1:Nx
        for i in r
            if !(leftbondspace(AL[i]) == rightbondspace(AL[i + 1]))
                throw(SpaceMismatch(""))
            end
        end
        AC = similar(AL)
        AC .= AL .× C
        return new{L,Mt,Ct}(AL, C, AR, AC)
    end
end

const InfMPS{L<:Infinite} = AbstractMatrixProductState{L}

function MatrixProductState{L}(data::Tuple) where {L}
    return MatrixProductState{L}((TensorLattice{L}.(hcat.(data)))...)
end

function MatrixProductState{L}(AL::Vector, C::Vector, AR::Vector) where {L}
    return MatrixProductState((AL, C, AR))
end

@doc raw"""
    InfMPS(V::Vector{<:MPSTensor})

Constructs an `InfMPS` from a vector `V` of `MPSTensor`'s representing a matrix product
state. The output `InfMPS` is the represention of the matrix product state `V` in the mixed
gauge. 
"""
function InfMPS{L}(v::AbstractVector{<:MPSTensor}) where {L}
    AL, C, AR = mixedgauge(v)
    return InfMPS{L}(AL, C, AR)
end

@doc raw"""
    InfMPS(f, T::Type{<:Number}, d::TensorSpace{S}, χ::S, Nx::Int=1) where {S<:IndexSpace}

Construct a `Nx` unit cell `InfMPS` with `MPSTensor`s and `BondTensor` generated from the 
function `f`. The space of the physical index or indices is provided by `d` and the space
of the bond index is provided by `χ`. If `Nx` is not provided, then a trivial unit cell
(type `InfMPS{1}`) is assumed. 

Possibilites for `f` are the same as the `TensorMap` struct from TensorKit.jl. That is, 
`randn` and `rand` from Julia Base (alternatively one can use aliases `randnormal` and 
`randuniform` respectively, as provided by TensorKit.jl.), as well as `randisometry` 
(alternatively called `randharr`) from TensorKit.jl.
"""
function InfMPS{L}(
    f, T::Type{<:Number}, physcod::ProductSpace{S,Nc}, physdom::ProductSpace{S,Nd}, χ::S
) where {L,Nc,Nd,S<:IndexSpace}
    N₁ = 1 + Nc
    N₂ = 1 + Nd
    Nx, _ = size(L)
    A = SizedVector{Nx}((
        MPSTensor(f, T, physcod, physdom, χ)::MPSTensor{S,N₁,N₂} for _ in 1:Nx
    ))
    return InfMPS{L}(A)
end
function InfMPS{L}(
    f, T::Type{<:Number}, physcod::TensorSpace{S}, physdom::TensorSpace{S}, χ::S
) where {L,S<:IndexSpace}
    return InfMPS{L}(f, T, convert(ProductSpace, physcod), convert(ProducSpace, physdom), χ)
end
# function InfMPS(f, T::Type{<:Number}, d::TensorSpace{S}, χ::S) where {S<:IndexSpace}
#     return InfMPS(f, T, 1, d, χ)
# end

numsites(::AbstractMPS{<:LatticeSize{Nx}}) where {Nx} = Nx
Base.length(a::AbstractMPS) = numsites(a)

function bondtensortype(
    ::Type{<:AbstractMPS{<:LatticeSize,<:MPSTensor,C}}
) where {C<:BondTensor}
    return C
end
bondtensortype(m::AbstractMPS) = bondtensortype(typeof(m))

function mpstensortype(::Type{<:AbstractMPS{<:LatticeSize,M}}) where {M<:MPSTensor}
    return M
end
mpstensortype(m::InfMPS) = mpstensortype(typeof(m))

function TensorKit.spacetype(
    ::Type{<:AbstractMPS{<:LatticeSize,<:MPSTensor{S}}}
) where {S<:IndexSpace}
    return S
end
TensorKit.spacetype(m::AbstractMPS) = spacetype(typeof(m))

bondtensor(A::InfMPS) = A.C
centraltensor(A::InfMPS) = A.AC
lefttensor(A::InfMPS) = A.AL
righttensor(A::InfMPS) = A.AR

function leftbondspace(A::InfMPS{<:LatticeSize{Nx}}) where {Nx}
    return (codomain(bondtensor(A)[x]) for x in 1:Nx)
end
function rightbondspace(A::InfMPS{<:LatticeSize{Nx}}) where {Nx}
    return (domain(bondtensor(A)[x]) for x in 1:Nx)
end
# function InfMPS(
#     f, ::Type{T}, physcodom::S, physdom::S, bond::S
# ) where {S<:IndexSpace,T<:Number}
#     return InfMPS(mpsdata(f, T, physcodom, physdom, bond))
# end

# function mpsdata(
#     f, ::Type{T}, physcodom::S, physdom::S, bond::S
# ) where {S<:IndexSpace,T<:Number}
#     A = TensorMap(f, T, physcodom * bond, physdom * bond)
#     AL, C, _ = leftgauge(A)
#     AR, C, _ = rightgauge(AL, C)
#     return AL, C, AR
# end

# bondmatrix(M::AbstractInfMPS) = M.C

# centraltensor(M::AbstractInfMPS) = _AC(M.AL, M.C)

# lefttensor(M::AbstractInfMPS) = M.AL
# righttensor(M::AbstractInfMPS) = M.AR

abstract type AbstractMultilineInfMPS{Ly,Lx,M<:InfMPS{Lx}} end

struct MultilineInfMPS{Ly,Lx,M<:InfMPS{Lx}} <: AbstractMultilineInfMPS{Ly,Lx,M}
    data::CompositeLattice{Ly,M}
end
mpstype(::Type{<:MultilineInfMPS{Ly,Lx,M}}) where {Ly,Lx,M} = M
mpstype(m::MultilineInfMPS) = mpstype(typeof(m))

function MultilineInfMPS(v::Vararg{<:InfMPS,Ny}) where {Ny}
    return MultilineInfMPS(CompositeLattice{Infinite{1,Ny}}(hcat(v...)))
end
MultilineInfMPS{Ly}(v::Vector{M}) where {Ly,M<:InfMPS} = MultilineInfMPS(v...)

function MultilineInfMPS{L}(
    f, T::Type{<:Number}, physcod::ProductSpace{S,Nc}, physdom::ProductSpace{S,Nd}, χ::S
) where {Nc,Nd,S<:IndexSpace,L}
    N₁ = 1 + Nc
    N₂ = 1 + Nd
    t = (InfMPS{L}(f, T, physcod, physdom, χ) for _ in 1:Ny)
    return MultilineInfMPS(t...)
end
function MultilineInfMPS{L}(
    f, T::Type{<:Number}, physcod::TensorSpace{S}, physdom::TensorSpace{S}, χ::S
) where {S<:IndexSpace,L}
    return MultilineInfMPS{Nx,Ny}(
        f, T, convert(ProductSpace, physcod), convert(ProductSpace, physdom), χ
    )
end

Base.firstindex(A::MultilineInfMPS) = (A.mps)[begin]
Base.lastindex(A::MultilineInfMPS) = (A.mps)[end]
Base.getindex(A::MultilineInfMPS, y::Int) = (A.mps)[y]

Base.size(::MultilineInfMPS{L}) where {L} = size(L)
Base.length(A::MultilineInfMPS) = size(A)[2]

function Base.iterate(A::MultilineInfMPS, state=1)
    _, Ny = size(A)
    if state > Ny
        return nothing
    else
        return A[state], state + 1
    end
end

function Base.setindex!(
    A::MultilineInfMPS{L,M}, v::M, y::Int
) where {L,M<:InfMPS}
    vec = A.mps
    vec[y] = v
    return A
end

mpstensortype(A::MultilineInfMPS) = mpstensortype(mpstype(A))
bondtensortype(A::MultilineInfMPS) = bondtensortype(mpstype(A))
#return a matrix of tensors (this is a bit verbose)
function bondtensor(A::MultilineInfMPS{Ly,Lx}) where {Ly,Lx}
    Nx = size(Lx)[1]
    Ny = size(Ly)[2]
    tm = Matrix{bondtensortype(A)}(undef, Nx, Ny)
    tup = (bondtensor(y) for y in A)
    tm .= hcat(tup...)
    return TensorLattice(tm, unitcell(A))
end
function centraltensor(A::MultilineInfMPS{Ly,Lx}) where {Ly,Lx}
    Nx = size(Lx)[1]
    Ny = size(Ly)[2]
    tm = Matrix{mpstensortype(A)}(undef, Nx, Ny)
    tup = (centraltensor(y) for y in A)
    tm .= hcat(tup...)
    return TensorLattice(tm, unitcell(A))
end
function lefttensor(A::MultilineInfMPS{Ly,Lx}) where {Ly,Lx}
    Nx = size(Lx)[1]
    Ny = size(Ly)[2]
    tm = Matrix{mpstensortype(A)}(undef, Nx, Ny)
    tup = (lefttensor(y) for y in A)
    tm .= hcat(tup...)
    return TensorLattice(tm, unitcell(A))
end
function righttensor(A::MultilineInfMPS{Ly,Lx}) where {Ly,Lx}
    Nx = size(Lx)[1]
    Ny = size(Ly)[2]
    tm = Matrix{mpstensortype(A)}(undef, Nx, Ny)
    tup = (righttensor(y) for y in A)
    tm .= hcat(tup...)
    return TensorLattice(tm, unitcell(A))
end
