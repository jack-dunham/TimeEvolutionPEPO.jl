"""
$(TYPEDEF)

Concrete type representing an approximation to Gaussian decay profile term as a sum of
exponential functions, that is,`ExpOp`.
```math
\\sum_{i = -\\infty}^{\\infty} \\sum_{r = 1}^{\\infty} a e^{-\\mu r^2} C_{i} A_{i + 1} \\cdots B_{i + r} 
\\approx
\\sum_{i = -\\infty}^{\\infty} \\sum_{r = 1}^{\\infty} a \\sum_{k = 1}^{k_{\\max}} \\lambda_{k}^{r} C_{i} A_{i + 1} \\cdots B_{i + r}
```
where ``k_{\\max}`` is the number of terms in approximation.
"""
struct GaussOp <: AbstractAxialTerm
    a::ComplexF64
    μ::ComplexF64
    terms::Tuple{Vararg{ExpOp}}
    function GaussOp(a, μ, terms)
        t0 = first(terms)
        if any(t -> t0.C != t.C || t0.B != t.B, terms)
            throw(
                ArgumentError(
                    "mismatched C or B operators found when constructing `GaussOp`: " *
                    "all respective C and B operators must be the same every term",
                ),
            )
        end
        return new(a, μ, terms)
    end
end

terms(g::GaussOp) = g.terms
isexpandable(term::GaussOp) = false

GaussOp(a, μ, H::AbstractTerm) = GaussOp(a, μ, terms(H))

"""
    $(FUNCTIONNAME)(C::AbstractMatrix, B::AbstractMatrix, μ::Number, [A = one(C)]; kwargs...)

Construct an axial term with interaction strength decaying approximately as a Gaussian.
The keyword arguments listed below control the approximation.

## Keywords
 - `N::Int=50`: The interval the approximation should be defined on.
 - `kmax::Int=4`: The namximum number of exponentially-decaying functions appearing in the 
    weighted sum.
 - `gtol::Float64=1e-10`: Break tolerance for considering succesive approximations with 
    respect to the exact decay profile.
 - `force::Bool=false`: Force an approximation of exactly `kmax` terms.
"""
function GaussOp(
    C::AbstractMatrix,
    B::AbstractMatrix,
    μ::Number,
    A::AbstractMatrix=one(C);
    N::Int=50,
    force::Bool=false,
    ngmax::Int=4,
    gtol::Float64=1e-10,
)
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
```math
\\sum_{i,j = -\\infty}^{\\infty} \\sum_{x,y = 1}^{\\infty} a e^{-\\mu (x^2 + y^2)} C_{i,j} B_{i+x,j+y} 
\\approx 
\\sum_{i,j= -\\infty}^{\\infty} \\sum_{x,y = 1}^{\\infty} a e^{-\\mu_{x} x^2 } e^{-\\mu_{y} y^2} C_{i,j} B_{i+x,j+y} 
```
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
                    "respective C and B operators must be for most axial `GaussOp` approximations",
                ),
            )
        end
        return new(a, gx, gy, expand)
    end
end

isexpandable(term::Union{GaussOp2d}) = term.expand

function (g::GaussOp2d)(x::Int, y::Int)
    expsx = terms(g.gx)
    expsy = terms(g.gy)

    cumsum = 0

    if iszero(x) && iszero(y)
        # Returning NaN instead of Inf is more useful here as it is ignored in plots
        return NaN
    elseif iszero(x)
        cumsum += mapreduce(op -> g.a * op.a * op.λ^(y), +, expsy)
    elseif iszero(y)
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

"""
    $(FUNCTIONNAME)(C::AbstractMatrix, B::AbstractMatrix, μx::Number, [μy = μx, A = one(C)]; kwargs...)

Construct a radial term with interaction strength decaying approximately as a Gaussian.
The keyword arguments `kwargs` are passed directly to the equivalant `GaussOp` 
constructor and are listed below. Currently indentical paramaters must be used for each 
directional approximation.

## Keywords
 - `N::Int=50`: The interval the approximation should be defined on.
 - `kmax::Int=4`: The namximum number of exponentially-decaying functions appearing in the 
    weighted sum.
 - `gtol::Float64=1e-10`: Break tolerance for considering succesive approximations with 
    respect to the exact decay profile.
 - `force::Bool=false`: Force an approximation of exactly `kmax` terms.
"""
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
