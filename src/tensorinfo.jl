abstract type AbstractTensorInfo{Cod<:NTuple{6,Any},Dom<:NTuple{6,Any}} end

abstract type AbstractIndexInfo{Cod<:NTuple{6,Any},Dom<:NTuple{6,Any}} end
struct IndexInfo{Cod,Dom} <: AbstractIndexInfo{Cod,Dom} end

abstract type AbstractMPSInfo{Cod<:Tuple{Any,Any,0,Any,0,0},Dom<:Tuple{Any,0,0,Any,Any,0}} <: AbstractIndexInfo{Cod,Dom} end
struct MPSInfo{Cod,Dom} <: AbstactMPSInfo{Cod,Dom} end

abstract type AbstractMPOInfo{Cod<:Tuple{Any,Any,0,Any,Any,0},Dom<:Tuple{Any,Any,0,Any,Any,0}} <: AbstractIndexInfo{Cod,Dom} end
struct MPOInfo{Cod,Dom} <: AbstactMPOInfo{Cod,Dom} end

abstract type AbstractPEPSInfo{Cod<:Tuple{Any,Any,Any,Any,Any,0},Dom<:Tuple{Any,Any,0,Any,Any,Any}} <: AbstractIndexInfo{Cod,Dom} end
struct PEPSInfo{Cod,Dom} <: AbstactPEPSInfo{Cod,Dom} end

abstract type AbstractPEPOInfo{Cod,Dom} <: AbstractIndexInfo{Cod,Dom} end
struct PEPOInfo{Cod,Dom} <: AbstactPEPOInfo{Cod,Dom} end

const TrivialMPS = MPSInfo{Tuple{1,1,0,0,0,0}, Tuple{1,0,0,0,0,0}}
const TrivialMPO = MPOInfo{Tuple{1,1,0,0,0,0}, Tuple{1,1,0,0,0,0}}
const TrivialPEPS = PEPSInfo{Tuple{1,1,1,0,0,0}, Tuple{1,1,0,0,0,0}}
const TrivialPEPO = PEPOInfo{Tuple{1,1,1,0,0,0}, Tuple{1,1,1,0,0,0}}

# User defined


abstract type AbstractTensorSpace{S} end

struct TensorSpace{Info, S<:IndexSpace} <: AbstractTensorSpace{S}
    codomain::NTuple{6,ProductSpace{S}}
    domain::NTuple{6,ProductSpace{S}}
    function TensorSpace(
        cod::NTuple{6,ProductSpace{S}}, dom::NTuple{6,ProductSpace{S}}
    ) where {S}
        cod_ranks = ntuple(i -> length(cod[i]), Val(6))
        dom_ranks = ntuple(i -> length(dom[i]), Val(6))
        Info = IndexInfo{Tuple{cod_ranks...}, Tuple{dom_ranks...}}
        return new{Info, S}(cod, dom)
    end
end

const MPSSpace{Info<:AbstractMPSInfo, S} <: TensorSpace{Info,S}
const MPOSpace{Info<:AbstractMPOInfo, S} <: TensorSpace{Info,S}
const PEPSSpace{Info<:AbstractPEPSInfo, S} <: TensorSpace{Info,S}
const PEPOSpace{Info<:AbstractPEPOInfo, S} <: TensorSpace{Info,S}

abstract type AbstractPhysicalTensor{S<:AbstractTensorSpace,T<:AbstractTensorMap} end

convert(T::Type{<:AbstractTensorMap}, p::AbstractPhysicalTensor) = convert(T, p.data)

struct PhysicalTensor{S,T<:TensorMap} <: AbstractPhysicalTensor{S,T}
    data::T
    space::S
end

const MPSTensor{S<:MPSSpace, T} <: AbstractPhysicalTensor{S,T}
const MPOTensor{S<:MPOSpace, T} <: AbstractPhysicalTensor{S,T}
const PEPSTensor{S<:PEPSSpace, T} <: AbstractPhysicalTensor{S,T}
const PEPOTensor{S<:PEPOSpace, T} <: AbstractPhysicalTensor{S,T}

struct AdjointPhysicalTensor{S,T<:AdjointTensorMap{S}} <: AbstractPhysicalTensor{S,T}
    data::T
    space::S
end

# function parse_einsumsize(cod::Expr, dom::Expr)
#     N₁ = sum(cod.args[2:7])
#     N₂ = sum(dom.args[2:7])
#     ex_cod = Expr(:row, Array(undef,N₁)...)
#     ex_dom = Expr(:row, Array(undef,N₂)...)
#     return Expr(:vcat, ex_cod, ex_cod)
# end

# function size_to_tuple(ex)
#     return tuple(ex.args[2:7]...)
# end

# function parse_matrix(ex)
#     for face in size
#     end
# end

# function order_check(a::Int64, b::Float64)
#     @hello a
# end

# function applytoT()
#     @tensor Fn[d; ]:=F[ui,mi,di]*a[ui,4,u] * m[1,2,3,4]
# end


# const TrivialMPS = TensorInfo{Tuple{1,1,0,0,0,0}, Tuple{1,0,0,0,0,0}}
# TrivialMPO


# @contract function applyhac(A::MPS, L::LEnv, M::MPO, R::REnv)
#     return [δ A δ ; L M R ; δ 0 δ]
# end


# macro newtensor(name_ex,cod_ex,dom_ex,super_ex=:Index)
#     info_name = string(name_ex, "Info")
#     super_name = string(super_ex, "Info")
# end


# function testfunc()
#     a = quote
#         function testfuncout()
#             println("hello!")
#         end
#     end
#     eval(a)
# end
#
