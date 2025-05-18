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
        println()
        print("  ")
        show(io, m, g)
    end
    return nothing
end

# DEPREC: This is now replaced by the contents of model/fsm.jl
function fsmpowop(C, B, localops, linops, space, dt, wk, mks, lks; J=1.0, extend=true)
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

    Id = kron(Id, Id)

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

# Add a Gaussian term to the FSM

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
isexpandable(term::PowOp) = term.expand

expandterm(term) = terms(term)
expandterm(term::PowOp) = map(t -> term.a * t, terms(term))
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
