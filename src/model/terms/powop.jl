"""
$(TYPEDEF)

Concrete type representing an approximation to Gaussian decay profile term as a sum of exponential functions,
that is,`ExpOp`.
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
                "all respective C and B operators must be the same every term"
                )
            )
        end
        return new(a, μ, terms)
    end
end

GaussOp(a,μ,H::AbstractTerm) = GaussOp(a,μ,terms(H))

terms(g::GaussOp) = g.terms

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

struct GaussOp2d <: AbstractRadialTerm
    a::ComplexF64
    gx::GaussOp
    gy::GaussOp
    tied::Bool
    function GaussOp2d(a, gx::GaussOp, gy::GaussOp, tied::Bool)
        if gx.C != gy.C || gx.B != gx.B
            throw(
                ArgumentError(
                "mismatched C or B operators found when constructing `GaussOp2d`: " *
                "respective C and B operators must be for most axial `GaussOp` approximations"
                )
            )
        end
        return new(a, gx, gy, tied)
    end
end

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
    tied::Bool
    function PowOp(a, α, terms::Tuple{Vararg{T}}, tied) where {T<:Union{GaussOp2d,ExpOp}}
        t0 = first(terms)
        if any(t -> t0.C != t.C || t0.B != t.B, terms)
            throw(
                ArgumentError(
                "mismatched C or B operators found when constructing `GaussOp`: " *
                "all respective C and B operators must be the same every term"
                )
            )
        end
        return new{T}(a, α, terms, tied)
    end
end
PowOp(a, α, H::AbstractTerm, tied) = PowOp(a, α, terms(H), tied)


isaxial(::PowOp{GaussOp2d}) = false
istied(p::Union{PowOp,GaussOp2d}) = p.tied

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

function powop(C, B, α, A=one(C); n=10, N=50, tied=true)
    X, Λ = approx_as_sum_of_exp(α, n; N=N)

    terms = mapreduce((x, λ) -> x * expop(C, B, λ, A), +, X, Λ)

    return PowOp(1.0, α, terms, tied)
end
function powop2d(C, B, α, A=one(C); n=10, N=50, tied=false, kwargs...)
    X, Λ = approx_as_sum_of_exp(α / 2, n; N=N)

    M = -log.(Λ)

    terms = mapreduce((x, μ) -> x * gaussop2d(C, B, μ, A; kwargs...), +, X, M)

    return PowOp(1.0, α, terms, tied)
end

function gaussop(C, B, μ, A=one(C); N=50, force=false, ngmax=4, gtol=1e-10)
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

    terms = mapreduce((x, λ) -> x * expop(C, B, λ, A), +, X, Λ)

    return GaussOp(1.0, μ, terms)
end

gaussop2d(C, B, μx::Number, A=one(C); kwargs...) = gaussop2d(C, B, μx, μx, A; kwargs...)
function gaussop2d(C, B, μx::Number, μy::Number, A=one(C); tied=true, kwargs...)
    gx = gaussop(C, B, μx, A; kwargs...)
    gy = gaussop(C, B, μy, A; kwargs...)
    return GaussOp2d(1.0, gx, gy, tied)
end

function Base.show(io::IO, ::MIME"text/plain", term::GaussOp2d)

    gx = term.gx
    gy = term.gy

    nx = length(terms(gx))
    ny = length(terms(gy))


    print(io, "$((nx,ny))-term GuassOp2d approximation to μ = $((gx.μ,gy.μ))")
    return nothing
end

function Base.show(io::IO, m::MIME"text/plain", term::PowOp{T}) where {T}

    n = length(terms(term))

    print(io, "$n-term PowOp{$T} approximation to α = $(term.α)")
    for g in terms(term)
        println()
        print("  ")
        show(io, m, g)
    end
    return nothing
end
function fsmpowop(
    C,
    B,
    localops,
    linops,
    space,
    dt,
    wk,
    mks,
    lks;
    J=1.0,
    extend=true,
)
    Id = one(C)

    thermal = isnothing(linops) ? true : false

    T = Dict{NTuple{4,Int},Matrix{ComplexF64}}

    rules = (A=T(), B=T(), C=T(), D=T())

    X, Y, Z = PAULI

    D = kron(zero(Id), zero(Id))

    for Aop in localops

        @debug "one local term coefficient: $(Aop.a)"

        A = Matrix(Aop)

        if thermal
            D += -0.5 * (kron(A, Id) + kron(Id, transpose(A)))
        else
            D += -im * (kron(A, Id) - kron(Id, transpose(A)))
        end
    end

    if !(linops isa Nothing)
        for L in linops
            acomm = (kron(L' * L, one(L)) + kron(one(L), transpose(L' * L)))
            D += kron(L, transpose(L')) - 1 / 2 * acomm
        end
    end

    rules.A[1, 1, 1, 1] = one(D)
    rules.D[1, 1, 1, 1] = D

    if thermal
        c1 = c2 = -J / 2
    else
        c1 = -im * J
        c2 = im * J
    end

    C1 = c1 * kron(C, Id)
    B1 = kron(B, Id)

    C2 = c2 * kron(Id, transpose(C))
    B2 = kron(Id, transpose(B))

    # Get strides

    hstr = 1
    vstr = 1

    Id = kron(Id,Id)

    @debug "Generating Gaussian PEPSO" weight = wk numlocal = length(localops) 

    for (C, B) in zip((C1, C2), (B1, B2))
        gapprox = zip(mks, lks)

        for (nh, (mk, lk)) in enumerate(gapprox)
            # mklk = mk * lk 

            c = hstr + 2 * nh - 1
            u = hstr + 2 * nh

            # Start going east
            rules.C[c, 1, 1, 1] = wk * mk * lk * C

            # Finish going east
            rules.B[1, 1, c, 1] = B
            # Finish going west (unconventional)
            rules.B[u, 1, 1, 1] = B

            # Continue going east
            rules.A[c, 1, c, 1] = lk * Id
            # Continue going west (unconventional)
            rules.A[u, 1, u, 1] = lk * Id

            # Direction change rules
            for nv in 1:length(gapprox)
                v = vstr + nv

                # Change from going north to going east
                rules.A[c, v, 1, 1] = mk * lk * Id
                # Change from going north to going west (unconventional)
                rules.A[1, v, u, 1] = mk * lk * Id
            end
        end
        for (nv, (mk, lk)) in enumerate(gapprox)
            v = vstr + nv
            # Start going north
            rules.C[1, 1, 1, v] = wk * mk * lk * C

            # Finish going north
            rules.B[1, v, 1, 1] = B

            # Continue going north
            rules.A[1, v, 1, v] = lk * Id
        end

        hstr += 2 * length(gapprox)
        vstr += length(gapprox)
    end

    @info "TNO bond dimension: ($hstr,$vstr)"

    d = space
    di = dim(d)

    Dx = reduce(oplus, fill(oneunit(d), hstr))
    Dy = reduce(oplus, fill(oneunit(d), vstr))

    data = zeros(ComplexF64, di, di, di, di, hstr, vstr, hstr, vstr)

    fsmpepo!(data, rules, dt; extend=extend)

    pepo = TensorMap(data, d * d' * d' * d, Dx * Dy * Dx' * Dy')
    return pepo
end
# function MPO(eqn)
#     Id = Matrix{ComplexF64}(I, 2, 2)
#
#     twoloc = twolocalterms(eqn)
#     oneloc = onelocalterms(eqn)
#     nonloc = expterms(eqn)
#
#     if isempty(linops(eqn))
#         thermal = true
#         if !isempty(twoloc)
#             Id = one(first(twoloc).C)
#         elseif !isempty(oneloc)
#             Id = one(first(oneloc).D)
#         elseif !isempty(nonloc)
#             Id = one(first(nonloc).A)
#         else
#             throw(ArgumentError("cannot construct MPO; equation appears to have no terms"))
#         end
#     else
#         thermal = false
#         Id = one(Matrix{ComplexF64}(first(linops)))
#     end
#
#     chi = 2 * (length(twoloc) + length(nonloc))
#     T = BlockMatrix
#     A = T(undef, chi, chi)
#     B = T(undef, chi, 1)
#     C = T(undef, 1, chi)
#     D = T(undef, 1, 1)
#
#     nil = kron(zero(Id), zero(Id))
#
#     fill!(A, nil)
#     D[1, 1] = nil
#
#     i = 1
#
#     for t in twolocalterms(eqn)
#         C[1, i] = -t.a * im * kron(t.C, Id)
#         C[1, i + 1] = t.a * im * kron(Id, transpose(t.C))
#
#         B[i, 1] = kron(t.B, Id) # ket
#         B[i + 1, 1] = kron(Id, transpose(t.B)) # bra
#
#         i += 2
#     end
#
#     for t in expterms(eqn)
#         C[1, i] = -t.a * t.λ * im * kron(t.C, Id)
#         C[1, i + 1] = t.a * t.λ * im * kron(Id, transpose(t.C))
#
#         B[i, 1] = kron(t.B, Id)
#         B[i + 1, 1] = kron(Id, transpose(t.B))
#
#         A[i, i] = t.λ * kron(t.A, Id)
#         A[i + 1, i + 1] = t.λ * kron(Id, transpose(t.A))
#
#         i += 2
#     end
#
#     # LOCAL TERMS (divide each by 1/2 on assumption of two-dimensional simulation)
#     # Note we don't need the other factor of 1/2 as we are not overlapping operators here
#     # like we do with TEBD.
#
#     foreach(onelocalterms(eqn)) do term
#         # Absorb coefficient
#         H = Matrix{ComplexF64}(term)
#
#         D[1, 1] += -im / 2 * (kron(H, one(H)) - kron(one(H), transpose(H)))
#     end
#
#     foreach(linops(eqn)) do term
#         # Absorb coefficient
#         L = Matrix{ComplexF64}(term)
#
#         anticomm = kron(L' * L, one(L)) + kron(one(L), transpose(L' * L))
#         D[1, 1] += 1 / 2 * (kron(L, transpose(L')) - 1 / 2 * anticomm)
#     end
#
#     return MPO(A, B, C, D)
# end


isexpandable(term) = false
isexpandable(term::Union{GaussOp2d}) = false
isexpandable(term::Union{GaussOp, PowOp}) = !(term.tied)
isexpandable(::Hamiltonian) = false

expandterm(term) = terms(term)
#     if isexpandable(term)
#         subterms = terms(term)
#         rv = []
#         for subterm in subterms
#             append!(rv, expandterm(subterm))
#         end
#         return rv
#     else
#         return [term]
#     end
# end

