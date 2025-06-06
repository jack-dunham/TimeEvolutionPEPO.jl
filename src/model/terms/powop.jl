"""
$(TYPEDEF)

Concrete type representing an approximation to a power-law decay term as a sum of axial 
exponential functions, that is, as `Hamiltonian{ExpOp}` or radial Gaussians 
`Hamiltonian{GaussOp2d}`.
"""
struct PowOp{T<:Union{GaussOp2d,ExpOp}} <: AbstractSingleTerm
    a::Float64
    α::Float64
    terms::Tuple{Vararg{T}}
    expand::Bool
    function PowOp(a, α, terms::Tuple{Vararg{T}}, expand) where {T<:Union{GaussOp2d,ExpOp}}
        t0 = first(terms)
        if any(t -> t0.C != t.C || t0.B != t.B, terms)
            throw(
                ArgumentError(
                    "mismatched C or B operators found when constructing `GaussOp`: " *
                    "all respective C and B operators must be the same every term",
                ),
            )
        end
        return new{T}(a, α, terms, expand)
    end
end
PowOp(a, α, H::AbstractTerm, expand) = PowOp(a, α, terms(H), expand)

const PowOp2d = PowOp{GaussOp2d}

isaxial(::PowOp{GaussOp2d}) = false

terms(p::PowOp) = p.terms

(op::PowOp{ExpOp})(r::Int) = mapreduce(op -> op.a * op.λ^r, +, op.terms.terms)
function (op::PowOp{GaussOp2d})(x::Int, y::Int)
    cumsum = 0

    gaussians = op.terms

    x = abs(x)
    y = abs(y)

    for g in gaussians.terms
        expsx = g.gx.terms.terms
        expsy = g.gy.terms.terms

        if iszero(x) && iszero(y)
            # Returning NaN instead of Inf is more useful here as it is ignored in plots
            return NaN
        elseif iszero(x)
            cumsum += mapreduce(op -> g.a * op.a * op.λ^(y), +, expsy)
        elseif iszero(y)
            println(cumsum)
            cumsum += mapreduce(op -> g.a * op.a * op.λ^(x), +, expsx)
        else
            itr = Iterators.product(expsx, expsy)
            cumsum += mapreduce(
                ops -> g.a * ops[1].a * ops[1].λ^(x) * ops[2].a * ops[2].λ^(y), +, itr
            )
        end
    end

    return cumsum
end

function PowOp(C, B, α, A=one(C); n=10, N=50, expand=false)
    X, Λ = approx_as_sum_of_exp(α, n; N=N)

    terms = mapreduce((x, λ) -> x * ExpOp(C, B, λ, A), +, X, Λ)

    return PowOp(1.0, α, terms, expand)
end
function PowOp2d(C, B, α, A=one(C); kmax=10, N=50, expand=true, kwargs...)
    X, Λ = approx_as_sum_of_exp(α / 2, kmax; N=N)

    M = -log.(Λ)

    terms = mapreduce((x, μ) -> x * GaussOp2d(C, B, μ, A; kwargs...), +, X, M)

    return PowOp(1.0, α, terms, expand)
end

function Base.show(io::IO, m::MIME"text/plain", term::PowOp{T}) where {T}
    n = length(terms(term))

    print(io, "$n-term PowOp{$T} approximation to α = $(term.α)")
    for g in terms(term)
        println(io)
        print(io, "  ")
        show(io, m, g)
    end
    return nothing
end

isexpandable(term) = false
isexpandable(term::PowOp) = term.expand

expandterm(term) = terms(term)
expandterm(term::PowOp) = map(t -> term.a * t, terms(term))
