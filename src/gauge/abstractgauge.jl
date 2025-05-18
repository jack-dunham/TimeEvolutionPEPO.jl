abstract type AbstractGauge end

struct NoGauge end

@kwdef mutable struct WTG <: AbstractGauge
    test::Bool = true
    skiptol::Float64 = 1e-14
    postol::Float64 = eps()
end

@kwdef mutable struct MCF <: AbstractGauge
    tol::Float64=1e-10
    maxiter::Int=1000
end

function Base.show(io::IO, m::MIME"text/plain", sim::AbstractGauge)
    shownested(io, m, sim)
    return nothing
end
