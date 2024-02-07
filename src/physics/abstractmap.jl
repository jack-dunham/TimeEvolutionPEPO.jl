struct LiouvillianData{L,ElType,N}
    data::Lattice{L,Array{ElType,N},Matrix{Array{ElType,N}}}
end

abstract type AbstractTimeStepper end

struct DynamicalMap{T<:AbstractTimeStepper}
    x::T
    y::T
    DynamicalMap(x::T, y::T) where {T} = new{T}(x, y)
    DynamicalMap(x::T) where {T} = new{T}(x)
end

struct TrotterGates{L,MType} <: AbstractTimeStepper
    east::Lattice{L,MType,Matrix{MType}} #left
    west::Lattice{L,MType,Matrix{MType}} #right
end
struct SimpleGates{L,MType,LType} <: AbstractTimeStepper
    east::Lattice{L,MType,Matrix{MType}}
    west::Lattice{L,MType,Matrix{MType}}
    loc::Lattice{L,LType,Matrix{LType}}
end

struct WIIMPO{G,MType} <: AbstractTimeStepper
    data::Lattice{G,MType,Matrix{MType}}
end
struct WIIPEPO{G,MType} <: AbstractTimeStepper
    data::Lattice{G,MType,Matrix{MType}}
end

function Base.getindex(gates::TrotterGates, bond::Bond)
    return (gates.east)[bond.left], (gates.west)[bond.right]
end

struct Unidirectional{T}
    horizontal::T
    vertical::T
end

struct GlobalStepper{G,MType} <: AbstractTimeStepper
    t::Lattice{G,MType,Matrix{MType}}
end
struct LocalStepper{G,MType} <: AbstractTimeStepper
    u::Lattice{G,MType,Matrix{MType}} #left
    v::Lattice{G,MType,Matrix{MType}} #right
end

function Base.getindex(stepper::LocalStepper, bond::Bond)
    return (stepper.u)[bond.left], (stepper.v)[bond.right]
end
