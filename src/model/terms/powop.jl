"""
$(TYPEDEF)

Concrete type representing an approximation to a power-law decay term as a sum of axial 
exponential functions [`ExpOp`](@ref) or radial Gaussians [`GaussOp2d`](@ref).

# Constructors

    $(FUNCTIONNAME)(C, B, α, A=one(C); n=10, N=50, expand=false, kwargs...)

Constructs `op::$(FUNCTIONNAME)` such that `a * op` refers to terms of the form:
```math
\\hat{h}_{j} = a  \\sum_{l = 1}^{n} \\hat{g}^{[l]}_{j} \\approx 
a \\sum_{k > j}^{\\infty} \\frac{\\hat{C}_{j} \\hat{B}_{k}  }{|{k -j}|^{\\alpha}}
```
That is, an ``n`` term approximation to the power law ``r^{\\alpha}`` on the one-dimensional 
interval ``[1,N]``. If `expand=true`, then the expansion is automatically expanded into a 
Trotterized product. The extra `kwargs...` are passed in as keyword arguments to [`ExpOp`](@ref).

---

    $(FUNCTIONNAME){GaussOp2d}(C, B, α, A=one(C); n=10, N=50, expand=false, kwargs...)

Constructs `op::$(FUNCTIONNAME)` such that `a * op` refers to terms of the
form:
```math
\\hat{h}_{j} = a  \\sum_{l = 1}^{n} \\hat{g}^{[l]}_{j} \\approx 
a \\sum_{k}^{\\infty}
\\frac{\\hat{C}_{\\mathbf{r}_{j}} \\hat{B}_{\\mathbf{r}_k}  }{||\\mathbf{r}_k - \\mathbf{r}_j||^{\\alpha}}
```
That is, an ``n`` term approximation to the power law ``r^{\\alpha}`` on the disc
``\\{ r = \\sqrt{x^2 + y^2} \\mid x,y \\in \\mathbb{Z}_{\\leq N} \\}``. 
If `expand=true`, then the expansion is automatically expanded into a Trotterized product
(recommended).
The extra `kwargs...` are passed in as keyword arguments to [`GaussOp2d`](@ref).
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
                    "all respective C and B operators must be the same for every term",
                ),
            )
        end
        return new{T}(a, α, terms, expand)
    end
end
PowOp(a, α, H::AbstractTerm, expand) = PowOp(a, α, terms(H), expand)

"""
    const PowOp2d = PowOp{GaussOp2d}

Convenience alias.
"""
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
    X, Λ = approx_as_sum_of_exp(ComplexF64, α / 2, kmax; N=N)

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
