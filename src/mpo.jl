#FIX: sort out this useless crap 
struct MPOSignature{Kind,Nw,Mw,Ns,Ms,Ne,Me,Nn,Mn}
    codind::NamedTuple{
        (:w, :s, :e, :n),
        <:Tuple{NTuple{Nw,Int},NTuple{Ns,Int},NTuple{Ne,Int},NTuple{Nn,Int}},
    }
    domind::NamedTuple{
        (:w, :s, :e, :n),
        <:Tuple{NTuple{Mw,Int},NTuple{Ms,Int},NTuple{Me,Int},NTuple{Mn,Int}},
    }
    function MPOSignature{Kind}(
        codind::NamedTuple{
            (:w, :s, :e, :n),
            <:Tuple{NTuple{Nw,Int},NTuple{Ns,Int},NTuple{Ne,Int},NTuple{Nn,Int}},
        },
        domind::NamedTuple{
            (:w, :s, :e, :n),
            <:Tuple{NTuple{Mw,Int},NTuple{Ms,Int},NTuple{Me,Int},NTuple{Mn,Int}},
        },
    ) where {Kind,Nw,Mw,Ns,Ms,Ne,Me,Nn,Mn}
        return new{Kind,Nw,Mw,Ns,Ms,Ne,Me,Nn,Mn}(codind, domind)
    end
end

const TR_SIG = MPOSignature{:TR_SIG}(
    (w=(1,), s=(2,), e=(), n=()), (w=(), s=(), e=(3,), n=(4,))
)

const POST_HS_SIG = MPOSignature{:POST_HS_SIG}(
    (w=(1, 2), s=(3,), e=(4, 5), n=(6,)), (w=(11, 12), s=(13,), e=(8, 9), n=(10,))
)

codind(s::MPOSignature) = s.codind
domind(s::MPOSignature) = s.domind

westind(s::MPOSignature) = codind(s)[:w], domind(s)[:w]
southind(s::MPOSignature) = codind(s)[:s], domind(s)[:s]
eastind(s::MPOSignature) = codind(s)[:e], domind(s)[:e]
northind(s::MPOSignature) = codind(s)[:n], domind(s)[:n]
ind(s::MPOSignature, d::Symbol) = codind(s)[d], domind(s)[d]

struct MPOTensor{Kind,S,N₁,N₂,T,M<:MPOSignature{Kind}} <: AbstractPhysicalTensor{S,N₁,N₂,T}
    tensor::T
    sig::M
end

signature(m::MPOTensor) = m.sig

westind(m::MPOTensor) = westind(signature(m))
southind(m::MPOTensor) = southind(signature(m))
eastind(m::MPOTensor) = eastind(signature(m))
northind(m::MPOTensor) = northind(signature(m))
ind(m::MPOTensor, d::Symbol) = ind(signature(m), d)

function TensorKit.space(m::MPOTensor, d::Symbol)
    N = numout(m)
    codi, domi = ind(m, d)
    cod = slice(codi, codomain(m))
    dom = slice(domi .- N, domain(m))
    return cod, dom
end

MPOTensor(t::TensorMap{<:IndexSpace,2,2}) = MPOTensor(t, TR_SIG)
MPOTensor(t::TensorMap{<:IndexSpace,6,6}) = MPOTensor(t, POST_HS_SIG)

abstract type AbstractMatrixProductOperator{L,M<:MPOTensor} <: AbstractLattice{L,M} end
const AbstractMPO = AbstractMatrixProductOperator

struct MatrixProductOperator{L,M} <: AbstractMatrixProductOperator{L,M}
    data::Matrix{M}
end

const MPO = MatrixProductOperator
const InfMPO{L<:Infinite} = MatrixProductOperator{L}

function MatrixProductOperator{L}(m::MPOTensor) where {L}
    return MatrixProductOperator{L}(fill(m, size(L)...))
end
function MatrixProductOperator(::L, m::MPOTensor) where {L<:LatticeSize}
    return MatrixProductOperator{L}(m)
end


signature(m::InfMPO) = signature(to_matrix(m))
