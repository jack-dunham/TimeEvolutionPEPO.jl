abstract type AbstractUpdateStyle end

struct LocalUpdate <: AbstractUpdateStyle end
struct GlobalUpdate <: AbstractUpdateStyle end

abstract type AbstractTimeStepper end

latticetuple(s::AbstractTimeStepper) = s.data

abstract type AbstractTrotterizedTimeStepper <: AbstractTimeStepper end

bondaxis(s::AbstractTrotterizedTimeStepper) = s.axis
isreversed(s::AbstractTrotterizedTimeStepper) = s.isreversed

function Base.reverse(s::S) where {S<:AbstractTrotterizedTimeStepper}
    return S(latticetuple(s), bondaxis(s), !isreversed(s))
end

# # Updates along a single bond on an axis. 
# struct LocalStepper{G,MType} <: AbstractTrotterizedTimeStepper
#     data::NTuple{2,Lattice{G,MType,Matrix{MType}}} #left (u, e bond), right (v, w bond)
#     axis::Int
#     isreversed::Bool
# end

struct LocalStepper{G,MType} <: AbstractTrotterizedTimeStepper
    data::Lattice{G,NTuple{2,MType},Matrix{NTuple{2, MType}}} #left (u, e bond), right (v, w bond)
    axis::Int
    isreversed::Bool
end

function LocalStepper(lat; axis=1, isreversed=false)
    return LocalStepper(lat, axis, isreversed)
end

Base.size(ls::LocalStepper) = size(ls.data)

tensors(s::LocalStepper) = latticetuple(s)

function Base.getindex(stepper::LocalStepper, bond::Bond)
    u, v = tensors(stepper)

    u1, _ = u[bond]
    _, v2 = v[bond]

    return u1, v2
end
# Updates along both bonds on an axis
struct TwoBondStepper{G,MType<:AbsTen{4,2}} <: AbstractTrotterizedTimeStepper
    data::NTuple{2,Lattice{G,MType,Matrix{MType}}}
    axis::Int
    isreversed::Bool
end

struct AxialStepper{G,MType<:AbsTen{4,2}} <: AbstractTrotterizedTimeStepper
    data::Lattice{G,MType,Matrix{MType}}
    axis::Int
end


AxialStepper(lat; axis, kwargs...) = AxialStepper(lat, axis)

Base.size(s::AxialStepper) = size(tensors(s))

tensors(s::AxialStepper) = s.data

function tensors(s::TwoBondStepper)
    (t1, t2) = latticetuple(s)
    return isreversed(s) ? t2 : t1
end

function twobondstepper(u::L, v::L) where {G,L<:AbstractLattice{G,<:AbsTen{4,1}}}
    t1 = broadcast(u, v) do ui, vi
        @tensoropt begin
            op[ook oob ik ib; e w] := vi[ok ob ik ib; w] * ui[ook oob ok ob; e]
        end
        return op
    end
    t2 = broadcast(u, v) do ui, vi
        @tensoropt begin
            op[ook oob ik ib; e w] := vi[ook oob ok ob; w] * ui[ok ob ik ib; e]
        end
        return op
    end
    return t1, t2
end

struct FullStepper{G,MType} <: AbstractTimeStepper
    t::Lattice{G,MType,Matrix{MType}}
end

FullStepper(t; kwargs...) = FullStepper(t)

stepper_type(::Type, ::Type) = GenericStepper
stepper_type(m1, m2) = stepper_type(totype(m1), totype(m2))
stepper_type(m1::Type{<:AbstractMatrix}, m2::Type)  = stepper_type(eltype(m1), m2)

totype(t::Type) = t
totype(t) = typeof(t)

Base.size(fs::FullStepper) = size(fs.t)

tensors(s::FullStepper) = s.t
Base.reverse(s::FullStepper) = s

stepstyle(::Type{<:FullStepper}) = Global()

struct SplitStepper{G,MType} <: AbstractTimeStepper
    t::Lattice{G,MType,Matrix{MType}}
end

tensors(s::SplitStepper) = s.t
Base.reverse(s::SplitStepper) = s

struct TrotterizedHeisenbergStepper{G,MType}
    t::Vector{Lattice{G,MType,Matrix{MType}}}
end

struct GenericStepper end

GenericStepper(l::AbstractMatrix{L}) where {L<:AbsTen{4,2}} = AxialStepper(l)
GenericStepper(l::AbstractMatrix{L}) where {L<:AbsTen{4,4}} = FullStepper(l)

function applystepper!(pepo::LocalUpdatablePEPO, operator::LocalStepper, bond::Bond)
    # Get the current left and right pepo tensors
    setisevolved!(pepo, false)

    p1, p2 = pepo[bond]
    u1, v2 = operator[bond]

    # Get the evolved pepo tensors such that we can write into them
    pe1, pe2 = evolvedpair(pepo)

    @tensoropt begin
        q1[ok ob; e x s w n] := u1[ok ob ik ib; x] * p1[ik ib; e s w n]
        q2[ok ob; e s w x n] := v2[ok ob ik ib; x] * p2[ik ib; e s w n]
    end

    mergeinto!(pe1, q1)
    mergeinto!(pe2, q2)

    return pepo
end

function evolve(pepo::AxialPEPO, stepper::AxialStepper)

    @assert pepo.currentaxis == stepper.axis
    
    rv = broadcast(tensors(pepo), tensors(stepper)) do pin, mpo
        d1, d2, d3, d4 = domain(pin)
        D1, D3 = domain(mpo)

        dst = similar(pin, codomain(pin), fuse(d1 * D1) * d2 * fuse(d3 * D3)' * d4)
        # mpo = currentstepper(op)
        @tensoropt begin
            pout[ok ob; e ee s w ww n] := mpo[ok ob ik ib; ee ww] * pin[ik ib; e s w n]
        end

        mergeinto!(dst, pout)
    end

    return rv
end

function apply_four_bond_stepper!(dst, src, stepper::FullStepper)
    broadcast(dst, src, tensors(stepper)) do pev, pin, mpo
        @tensoropt begin
            pout[ok ob; e ee s ss w ww n nn] :=
                mpo[ok ob ik ib; ee ss ww nn] * pin[ik ib; e s w n]
        end

        mergeinto!(pev, pout)
    end

    return pepo
end
function apply_four_bond_stepper!(pev, pin, mpo::AbstractTensorMap)
    @tensoropt begin
        pout[ok ob; e ee s ss w ww n nn] :=
            mpo[ok ob ik ib; ee ss ww nn] * pin[ik ib; e s w n]
    end

    mergeinto!(pev, pout)
    return pev
end
function apply_four_bond_stepper_locally_hori(src, op, isoms)
    (_, us1, vw1, vn1), (ue2, us2, _, vn2) = isoms

    src1, src2 = map(sptensor, src)

    op1, op2 = op

    T = promote_type(scalartype(src1), scalartype(op1))
    ST = storagetype(first(src))

    D1, D2, D3, D4 = domain(src1)
    η1, η2, η3, η4 = domain(op1)

    ust = sptensor(us1)

    vwt = sptensor(vw1)
    vnt = sptensor(vn1)

    us = similar(ust, D2 * η2, D2)

    vw = similar(vwt, D3', D3' * η3')
    vn = similar(vnt, D4', D4' * η4')

    mergeinto!(us, ust)

    mergeinto!(vw, vwt)
    mergeinto!(vn, vnt)

    ok, ob, _, _ = codomain(op1)

    dst1_cod = ok * ob
    dst1_dom = D1 * η1 * D2 * D3 * D4

    dst1 = SparseTensorMap(undef, T, dst1_cod, dst1_dom)
    ten1 = TensorMap(undef, T, codomain(dst1), fuse(D1 * η1) * D2 * D3 * D4)

    @tensoropt begin
        dst1[ok ob; e1 e2 s w n] =
            op1[ok ob ik ib; e2 s2 w2 n2] *
            src1[ik ib; e1 s1 w1 n1] *
            us[s1 s2; s] *
            vw[w; w1 w2] *
            vn[n; n1 n2]
    end

    D1, D2, D3, D4 = domain(src2)
    η1, η2, η3, η4 = domain(op2)

    uet = sptensor(ue2)
    ust = sptensor(us2)

    vnt = sptensor(vn2)

    ue = similar(uet, D1 * η1, D1)
    us = similar(ust, D2 * η2, D2)

    vn = similar(vnt, D4', D4' * η4')

    mergeinto!(ue, uet)
    mergeinto!(us, ust)

    mergeinto!(vn, vnt)

    ok, ob, _, _ = codomain(op2)

    dst2_cod = ok * ob
    dst2_dom = D1 * D2 * D3 * η3 * D4

    dst2 = SparseTensorMap(undef, T, dst2_cod, dst2_dom)
    ten2 = TensorMap(undef, T, codomain(dst2), D1 * D2 * fuse(D3 * η3)' * D4)

    @tensoropt begin
        dst2[ok ob; e s w1 w2 n] =
            op2[ok ob ik ib; e2 s2 w2 n2] *
            src2[ik ib; e1 s1 w1 n1] *
            us[s1 s2; s] *
            ue[e1 e2; e] *
            vn[n; n1 n2]
    end

    mergeinto!(ten1, dst1)
    mergeinto!(ten2, dst2)

    return ten1 => ten2
end

function apply_pepso_left!(dst, src, op, isoms)
    (e, s, w, n) = isoms

    temp = _apply_pepso_left(src, op, s, w, n)
    mergeinto!(dst, temp)

    return dst
end
function apply_pepso_right!(dst, src, op, isoms)
    (e, s, w, n) = isoms

    temp = _apply_pepso_right(src, op, e, s, n)
    mergeinto!(dst, temp)

    return dst
end
function apply_pepso_up!(dst, src, op, isoms)
    (e, s, w, n) = isoms

    temp = _apply_pepso_up(src, op, e, w, n)
    mergeinto!(dst, temp)

    return dst
end
function apply_pepso_down!(dst, src, op, isoms)
    (e, s, w, n) = isoms

    temp = _apply_pepso_down(src, op, e, s, w)
    mergeinto!(dst, temp)

    return dst
end
function _apply_pepso_left(src, op, us, vw, vn)
    @tensoropt (
        ok => 2,
        ob => 2,
        ik => 2,
        ib => 2,
        s => D,
        w => D,
        n => D,
        e1 => D,
        s1 => D,
        w1 => D,
        n1 => D,
        e2 => D,
        s2 => D,
        w2 => D,
        n2 => D,
    ) dst[ok ob; e1 e2 s w n] :=
        op[ok ob ik ib; e2 s2 w2 n2] *
        src[ik ib; e1 s1 w1 n1] *
        us[s1 s2; s] *
        vw[w; w1 w2] *
        vn[n; n1 n2]
    return dst
end

function _apply_pepso_right(src, op, ue, us, vn)
    @tensoropt (
        ok => 2,
        ob => 2,
        ik => 2,
        ib => 2,
        e => D,
        s => D,
        n => D,
        e1 => D,
        s1 => D,
        w1 => D,
        n1 => D,
        e2 => D,
        s2 => D,
        w2 => D,
        n2 => D,
    ) dst[ok ob; e s w1 w2 n] :=
        op[ok ob ik ib; e2 s2 w2 n2] *
        src[ik ib; e1 s1 w1 n1] *
        ue[e1 e2; e] *
        us[s1 s2; s] *
        vn[n; n1 n2]
    return dst
end

function _apply_pepso_up(src, op, ue, vw, vn)
    @tensoropt (
        ok => 2,
        ob => 2,
        ik => 2,
        ib => 2,
        e => D,
        w => D,
        n => D,
        e1 => D,
        s1 => D,
        w1 => D,
        n1 => D,
        e2 => D,
        s2 => D,
        w2 => D,
        n2 => D,
    ) dst[ok ob; e s1 s2 w n] :=
        op[ok ob ik ib; e2 s2 w2 n2] *
        src[ik ib; e1 s1 w1 n1] *
        ue[e1 e2; e] *
        vw[w; w1 w2] *
        vn[n; n1 n2]
    return dst
end

function _apply_pepso_down(src, op, ue, us, vw)
    @tensoropt (
        ok => 2,
        ob => 2,
        ik => 2,
        ib => 2,
        e => D,
        s => D,
        w => D,
        e1 => D,
        s1 => D,
        w1 => D,
        n1 => D,
        e2 => D,
        s2 => D,
        w2 => D,
        n2 => D,
    ) dst[ok ob; e s w n1 n2] :=
        op[ok ob ik ib; e2 s2 w2 n2] *
        src[ik ib; e1 s1 w1 n1] *
        ue[e1 e2; e] *
        us[s1 s2; s] *
        vw[w; w1 w2]
    return dst
end

function apply_four_bond_stepper_locally_vert(src, op, isoms)
    (ue1, _, vw1, vn1), (ue2, us2, vw2, _) = isoms

    src1, src2 = map(sptensor, src)

    op1, op2 = op

    T = promote_type(scalartype(src1), scalartype(op1))
    ST = storagetype(first(src))

    D1, D2, D3, D4 = domain(src1)
    η1, η2, η3, η4 = domain(op1)

    uet = sptensor(ue1)

    vwt = sptensor(vw1)
    vnt = sptensor(vn1)

    ue = similar(uet, D1 * η1, D1)

    vw = similar(vwt, D3', D3' * η3')
    vn = similar(vnt, D4', D4' * η4')

    mergeinto!(ue, uet)

    mergeinto!(vw, vwt)
    mergeinto!(vn, vnt)

    ok, ob, _, _ = codomain(op1)

    dst1_cod = ok * ob
    dst1_dom = D1 * D2 * η2 * D3 * D4

    dst1 = SparseTensorMap(undef, T, dst1_cod, dst1_dom)
    ten1 = TensorMap(undef, T, codomain(dst1), D1 * fuse(D2 * η2) * D3 * D4)

    @tensoropt begin
        dst1[ok ob; e s1 s2 w n] =
            op1[ok ob ik ib; e2 s2 w2 n2] *
            src1[ik ib; e1 s1 w1 n1] *
            ue[e1 e2; e] *
            vw[w; w1 w2] *
            vn[n; n1 n2]
    end

    D1, D2, D3, D4 = domain(src2)
    η1, η2, η3, η4 = domain(op2)

    uet = sptensor(ue2)
    ust = sptensor(us2)

    vwt = sptensor(vw2)

    ue = similar(uet, D1 * η1, D1)
    us = similar(ust, D2 * η2, D2)

    vw = similar(vwt, D3', D3' * η3')

    mergeinto!(ue, uet)
    mergeinto!(us, ust)

    mergeinto!(vw, vwt)

    ok, ob, _, _ = codomain(op2)

    dst2_cod = ok * ob
    dst2_dom = D1 * D2 * D3 * D4 * η4

    dst2 = SparseTensorMap(undef, T, dst2_cod, dst2_dom)
    ten2 = TensorMap(undef, T, codomain(dst2), D1 * D2 * D3 * fuse(D4 * η4)')

    @tensoropt begin
        dst2[ok ob; e s w n1 n2] =
            op2[ok ob ik ib; e2 s2 w2 n2] *
            src2[ik ib; e1 s1 w1 n1] *
            us[s1 s2; s] *
            ue[e1 e2; e] *
            vw[w; w1 w2]
    end

    mergeinto!(ten1, dst1)
    mergeinto!(ten2, dst2)

    return ten1 => ten2
end

function applystepper!(pepo::GlobalUpdatablePEPO, stepper::SplitStepper)
    setisevolved!(pepo, false)

    broadcast(tensors(pepo), currentbuffer(pepo), tensors(stepper)) do pin, pev, mpo
        @tensoropt begin
            pout[ok ob; e ee s w ww n] := mpo[ok ob ik ib; ee ww] * pin[ik ib; e s w n]
        end
        mergeinto!(pev, pout)
    end

    return pepo
end

# function applysplitstepper!(
#     pin::AbstractTensorMap{S,2,4}, mpo::AbstractTensorMap{S,3,2}
# ) where {S}
#     @tensoropt begin
#         pout[o; e ee s w ww n] := mpo[o ik ib; ee ww] * pin[ik ib; e s w n]
#     end
#     return pout
# end
#
# function applysplitstepper!(
#     pin::AbstractTensorMap{S,1,4}, mpo::AbstractTensorMap{S,3,2}
# ) where {S}
#     @tensoropt begin
#         pout[ok ob; e ee s w ww n] := mpo[ok ob i; ee ww] * pin[i; e s w n]
#     end
#     return pout
# end

abstract type AbstractTimeEvolution end

Base.show(io::IO, m::MIME"text/plain", sim::AbstractTimeEvolution) = shownested(io, m, sim)

@kwdef struct DMPO{TAlg<:AbstractTruncationScheme} <: AbstractTimeEvolution
    approx::Bool = true
    breaktol::Float64 = 0.0
    compress::Bool = true
    extend::Bool = true
    numsteps::Int = 100
    order::Int = 1
    timestep::Float64 = 0.01
    trotterorder::Int = 1
    truncalg::TAlg = SU()
    verbose::Bool = true
end

showheader(::Type{<:DMPO}) = "Dynamical matrix product operator (DMPO)"


abstract type AbstractTEBD <: AbstractTimeEvolution end

Base.show(io::IO, m::MIME"text/plain", sim::AbstractTEBD) = shownested(io, m, sim)
showheader(::Type{<:AbstractTEBD}) = "Time-evolving block decimation (TEBD)"

Base.parent(tebd::AbstractTEBD) = getfield(tebd, :parent)

function Base.getproperty(tebd::AbstractTEBD, sym::Symbol)
    if sym === :parent
        return getfield(tebd, sym)
    else
        return getfield(parent(tebd), sym)
    end
end

@kwdef mutable struct TEBD{
    TAlg<:AbstractTruncationScheme,
} <: AbstractTimeEvolution
    breaktol::Float64 = 0.0
    compresstol::Float64 = eps()
    trotterlevel::Symbol = :max
    numsteps::Int = 100
    timestep::Float64 = 0.01
    trotterorder::Int = 1
    truncalg::TAlg = SU()
    verbose::Bool = true
end

Base.parent(tebd::TEBD) = tebd

struct AxialTEBD{TAlg} <: AbstractTEBD
    parent::TEBD{TAlg}
end

AxialTEBD(;kwargs...) = AxialTEBD(TEBD(;kwargs...))

struct GlobalTEBD{TAlg} <: AbstractTEBD
    parent::TEBD{TAlg}
end

GlobalTEBD(;kwargs...) = GlobalTEBD(TEBD(;kwargs...))

ModelStyle(::Type{<:GlobalTEBD}) = IsGlobal()

@kwdef struct TETNO{TAlg,Style} <: AbstractTimeEvolution
    breaktol::Float64 = 0.0
    numsteps::Int = 100
    timestep::Float64 = 0.1
    trotterorder::Int = 1
    truncalg::TAlg = SU()
    truncstyle::Style = SemiParallel()
    verbose::Bool = true
end

truncstyle(method::TETNO) = method.truncstyle

showheader(::Type{<:TETNO}) = "Time-evolving tensor network operator (TETNO)"

@kwdef struct TEPEPSO{TAlg} <: AbstractTimeEvolution
    breaktol::Float64 = 0.0
    numsteps::Int = 100
    timestep::Float64 = 0.1
    trotterorder::Int = 1
    truncalg::TAlg = IterativeSU()
    verbose::Bool = true
    extend::Bool = true
end

truncstyle(method::TEPEPSO) = Global()

function showheader(::Type{<:TEPEPSO})
    return "Time-evolving projected entangled pair super-operator (tePEPSO)"
end
