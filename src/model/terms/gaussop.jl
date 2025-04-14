"""
$(TYPEDEF)

Concrete type representing an approximation to Gaussian decay profile term as a sum of
exponential functions, that is,`ExpOp`.
```math
a e^{-\\mu r^2} C_{i} A_{i + 1} \\cdots B_{i + r} \\approx a \\sum_{k = 1}^{k_{\\max}} \\lambda_{k}^{r} C_{i} A_{i + 1} \\cdots B_{i + r}
```
where ``k_{\\max}`` is `length(terms)` (see below).

## Fields

$(TYPEDFIELDS)
"""
struct GaussOp <: AbstractAxialTerm
    "See above."
    a::ComplexF64
    "See above."
    μ::ComplexF64
    "The `ExpOp` terms approximating the Gaussian."
    terms::Tuple{Vararg{ExpOp}}
    function GaussOp(a, μ, terms)
        t0 = first(terms)
        if any(t -> t0.C != t.C || t0.B != t.B, terms)
            throw(
                ArgumentError(
                "mismatched C or B operators found when constructing `GaussOp`: " *
                "all respective C and B operators must be the same every term"
                )
            )
        end
        return new(a, μ, terms)
    end
end

terms(g::GaussOp) = g.terms
isexpandable(term::GaussOp) = false

GaussOp(a,μ,H::AbstractTerm) = GaussOp(a,μ,terms(H))
function GaussOp(C, B, μ::Number, A=one(C); N=50, force=false, ngmax=4, gtol=1e-10)
    f = r -> exp(-μ * r^2)
    errf = (m, l) -> maximum(abs.((x -> sum(@. m * l^x)).(1:N) - f.(1:N)))

    ng0 = force ? ngmax : 1

    X, Λ = generalised_approx_as_sum_of_exp(f, ng0; N=N, T=ComplexF64)

    errfinal = errf(X, Λ)

    ngfinal = 1

    for ng in (ng0 + 1):ngmax
        errfinal < gtol && break

        Xn, Λn = generalised_approx_as_sum_of_exp(f, ng; N=N, T=ComplexF64)

        errnew = errf(Xn, Λn)

        if errnew < errfinal
            errfinal = errnew
            X, Λ = Xn, Λn
            ngfinal = ng
        end
    end

    @info "Error for $(real(μ)) with $ngfinal terms: $errfinal"

    terms = mapreduce((x, λ) -> x * ExpOp(C, B, λ, A), +, X, Λ)

    return GaussOp(1.0, μ, terms)
end


(op::GaussOp)(r::Int) = mapreduce(op -> op.a * op.λ^(abs(r)), +, op.terms.terms)

function Base.getproperty(g::GaussOp, name::Symbol)
    if name === :C
        return first(g.terms).C
    elseif name === :B
        return first(g.terms).B
    else
        return getfield(g, name)
    end
end

"""
$(TYPEDEF)

Concrete type representing an approximation to a two-dimensional Gaussian decay profile 
term as a product of approximate one-dimensional Gaussians as `GaussOp` objects.
exponential functions, that is,`ExpOp`.
```math
a e^{-\\mu (x^2 + y^2)} C_{i,j} B_{i+x,j+y} \\approx a e^{-\\mu_{x} x^2 } e^{-\\mu_{y} y^2} C_{i,j} B_{i+x,j+y} 
```

## Fields

$(TYPEDFIELDS)

See also: [`GaussOp`](@ref), [`ExpOp`](@ref), and [`PowOp`](@ref).
"""
struct GaussOp2d <: AbstractRadialTerm
    a::ComplexF64
    gx::GaussOp
    gy::GaussOp
    expand::Bool
    function GaussOp2d(a, gx::GaussOp, gy::GaussOp, expand::Bool)
        if gx.C != gy.C || gx.B != gx.B
            throw(
                ArgumentError(
                "mismatched C or B operators found when constructing `GaussOp2d`: " *
                "respective C and B operators must be for most axial `GaussOp` approximations"
                )
            )
        end
        return new(a, gx, gy, expand)
    end
end

isexpandable(term::Union{GaussOp2d}) = term.expand

function (g::GaussOp2d)(x::Int, y::Int)
    expsx = g.gx.terms.terms
    expsy = g.gy.terms.terms

    cumsum = 0

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

    return cumsum
end

function Base.getproperty(g::GaussOp2d, name::Symbol)
    if name === :C
        return g.gx.C
    elseif name === :B
        return g.gx.B
    else
        return getfield(g, name)
    end
end

GaussOp2d(C, B, μx::Number, A=one(C); kwargs...) = GaussOp2d(C, B, μx, μx, A; kwargs...)
function GaussOp2d(C, B, μx::Number, μy::Number, A=one(C); expand=false, kwargs...)
    gx = GaussOp(C, B, μx, A; kwargs...)
    gy = GaussOp(C, B, μy, A; kwargs...)
    return GaussOp2d(1.0, gx, gy, expand)
end

function Base.show(io::IO, ::MIME"text/plain", term::GaussOp2d)

    gx = term.gx
    gy = term.gy

    nx = length(terms(gx))
    ny = length(terms(gy))


    print(io, "$((nx,ny))-term GuassOp2d approximation to μ = $((gx.μ,gy.μ))")
    return nothing
end
