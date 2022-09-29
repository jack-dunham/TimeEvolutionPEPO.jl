abstract type AbstractInfMPS end

@doc raw"""
struct MPSTensor{S<:IndexSpace, N₁, N₂, ...} <: AbstractPhysicalTensor

Concrete subtype of `AbstractPhysicalTensor` used to represent a single tensor in a matrix
product state. It is assumed that the left and right bond indices are positioned at `N₁` and
`N₂` respectively. Has a single field `tensor` which is of type `TensorMap{S,N₁, N₂}`.
"""
struct MPSTensor{S<:IndexSpace,N₁,N₂,T<:TensorMap{S,N₁,N₂}} <: AbstractPhysicalTensor{S,T}
    tensor::T
end

@doc raw"""
struct BondTensor{S<:IndexSpace, ...} <: AbstractPhysicalTensor

Concrete subtype of `AbstractPhysicalTensor` used to represent a single bond tensor in a 
matrix product state. Has a single field `tensor` which is of type `TensorMap{S,1,1}`.
"""
struct BondTensor{S<:IndexSpace,T<:TensorMap{S,1,1}} <: AbstractPhysicalTensor{S,T}
    tensor::T
end

function MPSTensor(f, T::Type{<:Number}, d::TensorSpace{S}, χ::S) where {S<:IndexSpace}
    return MPSTensor(TensorMap(f, T, d * χ, χ))
end
function BondTensor(f, T::Type{<:Number}, χ::S) where {S<:IndexSpace}
    return BondTensor(TensorMap(f, T, χ, χ))
end

lbspace(M::MPSTensor{<:IndexSpace,N₁}) where {N₁} = codomain(M)[N₁]
rbspace(M::MPSTensor{<:IndexSpace,N₁,N₂}) where {N₁,N₂} = domain(M)[N₂]
bspace(M::BondTensor) = codomain(M)

@doc raw"""

    *(C::BondTensor{S}, A::MPSTensor{S}) where {S<:IndexSpace}

Apply a matrix `C` to the left bond of a MPS tensor `A`. The tensor `A` has $N₁$ indices in
the codomain, and $N₂$ in the domain with $N = N₁ + N₂$ total indices. The function assumes
the left and right bond indices are at positions N₁ and N₂ respectively. 
                                          
                                ┌─┐   ┌───────┐ 
                           N₁◄──│C│◄──│   A   │◄── N₂
                                └─┘   └───────┘ 
                                        │ ⋯  ▲       
                                        ▼    │     
                                        1   N₁+1     

"""
function Base.:*(C::BondTensor{S}, A::MPSTensor{S}) where {S<:IndexSpace}
    return MPSTensor(_lmul(C.tensor, A.tensor))
end

@doc raw"""

    *(A::MPSTensor{S}, C::BondTensor{S}) where {S<:IndexSpace}

Apply a matrix `C` to the right bond of a MPS tensor `A`. This function assumes the same as
the method `*(C::BondTensor{S}, A::MPSTensor{S}) where {S<:IndexSpace}`.

                                  ┌───────┐   ┌─┐   
                             N₁◄──│   A   │◄──│C│◄── N₂
                                  └───────┘   └─┘   
                                    │ ⋯  ▲       
                                    ▼    │     
                                    1   N₁+1     

"""
function Base.:*(A::MPSTensor{S}, C::BondTensor{S}) where {S<:IndexSpace}
    return MPSTensor(_rmul(A.tensor, C.tensor))
end
# struct AdjointMPSTensor{T<:TensorKit.AdjointTensorMap} <: AbstractPhysicalTensor
#     tensor::T
# end

function bondind(::MPSTensor{<:IndexSpace,N₁,N₂}) where {N₁,N₂}
    return N₁, N₂
end

struct InfMPS{Nx,S<:IndexSpace,N₁,N₂,Mt<:MPSTensor{S,N₁,N₂},Ct<:BondTensor{S}} <:
       AbstractInfMPS
    AL::SizedVector{Nx,Mt}
    C::SizedVector{Nx,Ct}
    AR::SizedVector{Nx,Mt}
    AC::SizedVector{Nx,Mt}
    function InfMPS(
        AL::SizedVector{Nx,Mt}, C::SizedVector{Nx,Ct}, AR::SizedVector{Nx,Mt}
    ) where {Nx,S<:IndexSpace,N₁,N₂,Mt<:MPSTensor{S,N₁,N₂},Ct<:BondTensor{S}}
        r = 1:Nx
        for i in r
            if !(lbspace(AL[i]) == rbspace(AL[mod(i + 1, r)]))
                throw(SpaceMismatch(""))
            end
        end
        AC = similar(AL)
        AC .= AL .* C
        return new{Nx,S,N₁,N₂,Mt,Ct}(AL, C, AR, AC)
    end
end

@doc raw"""
    InfMPS(V::Vector{<:MPSTensor})

Constructs an `InfMPS` from a vector `V` of `MPSTensor`'s representing a matrix product
state. The output `InfMPS` is the represention of the matrix product state `V` in the mixed
gauge. 

To construct an *ungauged* matrix product state object from `V`, see constructers associated
with the type `UngaugedInfMPS`.
"""
function InfMPS(v::SizedVector{Nx,<:MPSTensor}) where {Nx}
    AL, C, AR = mixedgauge(v)
    return InfMPS(AL, C, AR)
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
function InfMPS{Nx}(
    f, T::Type{<:Number}, d::ProductSpace{S,N}, χ::S
) where {Nx,N,S<:IndexSpace}
    N₁ = 1 + N
    N₂ = 1
    A = SizedVector{Nx}((MPSTensor(f, T, d, χ)::MPSTensor{S,N₁,N₂} for _ in 1:Nx))
    return InfMPS(A)
end
function InfMPS{Nx}(f, T::Type{<:Number}, d::S, χ::S) where {Nx,S<:IndexSpace}
    return InfMPS{Nx}(f, T, convert(ProductSpace, d), χ)
end
# function InfMPS(f, T::Type{<:Number}, d::TensorSpace{S}, χ::S) where {S<:IndexSpace}
#     return InfMPS(f, T, 1, d, χ)
# end

numsites(::InfMPS{Nx}) where {Nx} = Nx
Base.length(a::InfMPS) = numsites(a)

function bondtensortype(
    ::Type{<:InfMPS{Nx,<:IndexSpace,N₁,N₂,<:MPSTensor,Ct}}
) where {Nx,N₁,N₂,Ct<:BondTensor}
    return Ct
end
bondtensortype(m::InfMPS) = bondtensortype(typeof(m))

function mpstensortype(
    ::Type{<:InfMPS{Nx,<:IndexSpace,N₁,N₂,Mt}}
) where {Nx,N₁,N₂,Mt<:MPSTensor}
    return Mt
end
mpstensortype(m::InfMPS) = mpstensortype(typeof(m))

TensorKit.spacetype(::Type{<:InfMPS{Nx,S}}) where {Nx,S<:IndexSpace} = S
TensorKit.spacetype(m::InfMPS) = spacetype(typeof(m))

bondtensor(A::InfMPS) = A.C
centraltensor(A::InfMPS) = A.AC
lefttensor(A::InfMPS) = A.AL
righttensor(A::InfMPS) = A.AR

lbspace(A::InfMPS{Nx}) where {Nx} = (codomain(bondtensor(A)[x]) for x in 1:Nx)
rbspace(A::InfMPS{Nx}) where {Nx} = (domain(bondtensor(A)[x]) for x in 1:Nx)
# function InfMPS(
#     f, ::Type{T}, physcodom::S, physdom::S, bond::S
# ) where {S<:IndexSpace,T<:Number}
#     return InfMPS(mpsdata(f, T, physcodom, physdom, bond))
# end

function mpsdata(
    f, ::Type{T}, physcodom::S, physdom::S, bond::S
) where {S<:IndexSpace,T<:Number}
    A = TensorMap(f, T, physcodom * bond, physdom * bond)
    AL, C, _ = leftgauge(A)
    AR, C, _ = rightgauge(AL, C)
    return AL, C, AR
end

bondmatrix(M::AbstractInfMPS) = M.C

centraltensor(M::AbstractInfMPS) = _AC(M.AL, M.C)

lefttensor(M::AbstractInfMPS) = M.AL
righttensor(M::AbstractInfMPS) = M.AR

mutable struct MultiSiteInfMPS{Sites,S<:IndexSpace,N₁,N₂} <: AbstractInfMPS
    AL::Vector{<:TensorMap{S,N₁,N₂}}
    C::Vector{<:TensorMap{S,1,1}}
    AR::Vector{<:TensorMap{S,N₁,N₂}}
    function MultiSiteInfMPS(
        AL::Vector{<:TensorMap{S,N₁,N₂}},
        C::Vector{<:TensorMap{S,1,1}},
        AR::Vector{<:TensorMap{S,N₁,N₂}},
    ) where {S<:IndexSpace,N₁,N₂}
        if !(length(AL) == length(C) == length(AR))
            throw()
        end
        Sites = length(AL)
        return new{Sites,S,N₁,N₂}(AL, C, AR)
    end
end

function _multisitemps(
    f,
    ::Type{T},
    Sites::Integer,
    physcod::TensorSpace{S},
    physdom::TensorSpace{S},
    bond::TensorSpace{S},
) where {S<:IndexSpace,T<:Number}
    # Ai = TensorMap(f, T, physcod * bond, physdom * bond)

    # C = fill(Ci, Sites)
    # AL = fill(Ai, Sites)
    # AR = fill(Ai, Sites)

    r = 1:Sites

    A = [TensorMap(f, T, physcod * bond, physdom * bond) for i in r]
    AL = similar(A)
    C = [TensorMap(f, T, bond, bond) for i in r]
    AR = similar(A)

    λL = zeros(T, Sites)
    λR = zeros(T, Sites)

    # for i in r
    #     AL[i], C[i], λL[i] = leftgauge(A[i], C[i])
    # end
    # for i in r
    #     AR[i], C[mod(i - 1, r)], λR[i] = rightgauge(AL[i], C[mod(i - 1, r)])
    # end

    AL, C, AR = mixedgauge(A, C; diagonal=true)

    for i in r
        @assert _AC(AL[i], C[i]) ≈ _CA(C[mod(i - 1, r)], AR[i])
    end

    return AL, C, AR
end

function MultiSiteInfMPS{Sites}(
    f, ::Type{T}, physcod::S, physdom::S, bond::S
) where {S<:IndexSpace,T<:Number,Sites}
    mps = _multisitemps(f, T, Sites, physcod, physdom, bond)
    return MultiSiteInfMPS(mps...)
end
function MultiSiteInfMPS{Sites}(
    f, ::Type{T}, phys::S, bond::S
) where {S<:IndexSpace,T<:Number,Sites}
    mps = _multisitemps(f, T, Sites, phys, one(S), bond)
    return MultiSiteInfMPS(mps...)
end

numsites(::MultiSiteInfMPS{Sites}) where {Sites} = Sites

struct MultilineInfMPS{Nx,Ny,M<:InfMPS{Nx}}
    mps::SizedVector{Ny,M}
    cell::UnitCell{Nx,Ny}
    function MultilineInfMPS(mps::SizedVector{Ny,M}) where {Nx,Ny,M<:InfMPS{Nx}}
        S = spacetype(M)
        return new{Nx,Ny,M}(mps, UnitCell{Nx,Ny,S}())
    end
end

mpstype(::Type{<:MultilineInfMPS{Nx,Ny,M}}) where {Nx,Ny,M} = M
mpstype(m::MultilineInfMPS) = mpstype(typeof(m))

function unitcell(a::MultilineInfMPS{Nx,Ny,<:InfMPS{Nx,S}}) where {Nx,Ny,S<:IndexSpace}
    return (a.cell)::UnitCell{Nx,Ny,S}
end

function MultilineInfMPS(v::Vararg{<:InfMPS{Nx},Ny}) where {Nx,Ny}
    return MultilineInfMPS(SizedVector{Nx}(v))
end
MultilineInfMPS{Ny}(v::Vector{<:InfMPS{Nx}}) where {Nx,Ny} = SizedVector{Ny}(v)

function MultilineInfMPS{Nx,Ny}(
    f, T::Type{<:Number}, d::ProductSpace{S,N}, χ::S
) where {N,S<:IndexSpace,Nx,Ny}
    N₁ = 1 + N
    N₂ = 1
    M = InfMPS{Nx,S,N₁,N₂}
    t = (InfMPS{Nx}(f, T, d, χ)::M for _ in 1:Ny)
    return MultilineInfMPS(t...)
end
function MultilineInfMPS{Nx,Ny}(
    f, T::Type{<:Number}, d::S, χ::S
) where {S<:IndexSpace,Nx,Ny}
    return MultilineInfMPS{Nx,Ny}(f, T, convert(ProductSpace, d), χ)
end

Base.firstindex(A::MultilineInfMPS) = (A.mps)[begin]
Base.lastindex(A::MultilineInfMPS) = (A.mps)[end]
Base.getindex(A::MultilineInfMPS, y::Int) = (A.mps)[y]
Base.length(::MultilineInfMPS{Nx,Ny}) where {Nx,Ny} = Ny

Base.size(::MultilineInfMPS{Nx,Ny}) where {Nx,Ny} = (Nx,Ny)

function Base.iterate(A::MultilineInfMPS{Nx,Ny}, state=1) where {Nx,Ny}
    if state > Ny
        return nothing
    else
        return A[state], state + 1
    end
end

function Base.setindex!(
    A::MultilineInfMPS{Nx,Ny,M}, v::M, y::Int
) where {Nx,Ny,M<:InfMPS{Nx}}
    vec = A.mps
    vec[y] = v
    return A
end


mpstensortype(A::MultilineInfMPS) = mpstensortype(mpstype(A))
bondtensortype(A::MultilineInfMPS) = bondtensortype(mpstype(A))
#return a matrix of tensors (this is a bit verbose)
function bondtensor(A::MultilineInfMPS{Nx,Ny}) where {Nx,Ny}
    tm = Matrix{bondtensortype(A)}(undef, Nx, Ny)
    tup = (bondtensor(y) for y in A)
    tm .= hcat(tup...)
    return TensorMatrix(tm, unitcell(A))
end
function centraltensor(A::MultilineInfMPS{Nx,Ny}) where {Nx,Ny}
    tm = Matrix{mpstensortype(A)}(undef, Nx, Ny)
    tup = (centraltensor(y) for y in A)
    tm .= hcat(tup...)
    return TensorMatrix(tm, unitcell(A))
end
function lefttensor(A::MultilineInfMPS{Nx,Ny}) where {Nx,Ny}
    tm = Matrix{mpstensortype(A)}(undef, Nx, Ny)
    tup = (lefttensor(y) for y in A)
    tm .= hcat(tup...)
    return TensorMatrix(tm, unitcell(A))
end
function righttensor(A::MultilineInfMPS{Nx,Ny}) where {Nx,Ny}
    tm = Matrix{mpstensortype(A)}(undef, Nx, Ny)
    tup = (righttensor(y) for y in A)
    tm .= hcat(tup...)
    return TensorMatrix(tm, unitcell(A))
end


