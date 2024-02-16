abstract type AbstractUpdateStyle end
abstract type AbstractTimeStepper end

struct LocalUpdate <: AbstractUpdateStyle end
struct GlobalUpdate <: AbstractUpdateStyle end

# Updates along a single bond on an axis. 
struct LocalStepper{G,MType} <: AbstractTimeStepper
    u::Lattice{G,MType,Matrix{MType}} #left
    v::Lattice{G,MType,Matrix{MType}} #right
end

function Base.getindex(stepper::LocalStepper, bond::Bond)
    u1, _ = stepper.u[bond]
    _, v2 = stepper.v[bond]
    return u1, v2
end

# Updates along both bonds on an axis
struct GlobalStepper{G,MType<:AbsTen{4,2}} <: AbstractTimeStepper
    t::NTuple{2,Lattice{G,MType,Matrix{MType}}}
end

GlobalStepper(t1::AbstractLattice, t2::AbstractLattice) = GlobalStepper((t1, t2))
GlobalStepper(t::AbstractLattice) = GlobalStepper(t, t)

currentstepper(s::GlobalStepper) = first(s.t)
Base.reverse(s::GlobalStepper) = GlobalStepper(reverse(s.t))

function GlobalStepper(u::L, v::L) where {G,L<:AbstractLattice{G,<:AbsTen{4,1}}}
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
    return GlobalStepper(t1, t2)
end

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

function applystepper!(pepo::GlobalUpdatablePEPO, stepper::GlobalStepper)
    setisevolved!(pepo, false)

    # pin = getstate(pepo)
    # pev = getevolved(pepo)

    broadcast(latticetensors(pepo), evolvedtensors(pepo), currentstepper(stepper)) do pin, pev, mpo
        # mpo = currentstepper(op)
        @tensoropt begin
            pout[ok ob; e ee s w ww n] := mpo[ok ob ik ib; ee ww] * pin[ik ib; e s w n]
        end

        mergeinto!(pev, pout)
    end

    return pepo
end

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

@kwdef struct TEBD{
    TAlg<:AbstractTruncationScheme,S<:AbstractUpdateStyle,C<:TensorKit.TruncationScheme
} <: AbstractTimeEvolution
    breaktol::Float64 = 0.0
    compress::C = truncerr(100 * eps())
    numsteps::Int = 100
    timestep::Float64 = 0.01
    trotterorder::Int = 1
    truncalg::TAlg = SU()
    updatestyle::S = LocalUpdate()
    verbose::Bool = true
end

showheader(::Type{<:TEBD}) = "Time-evolving block decimation (TEBD)"
