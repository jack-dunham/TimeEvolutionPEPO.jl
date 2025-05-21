# Convenience const. 
# const AbsTen{N₁,N₂,S} = AbstractTensorMap{S,N₁,N₂}
# const TenAbs{N₂,N₁,S} = AbstractTensorMap{S,N₁,N₂}

const _DIR_TO_DOMIND = (east=1, south=2, west=3, north=4)

# Output the permutation of that results in the k'th element of p moved to the
# n'th position 
function nswap(k::Int, n::Int, p::NTuple{N}) where {N}
    if k == n
        return p
    elseif n > k
        return (p[1:(k - 1)]..., p[(k + 1):n]..., p[k]..., p[(n + 1):N]...)::typeof(p)
    else
        return (p[1:(n - 1)]..., p[k], p[n:(k - 1)]..., p[(k + 1):N]...)::typeof(p)
    end
end

# depreciate
function swap(k::Int, l::Int, N::Int)
    if !(k <= N && l <= N)
        throw(ArgumentError("k and l must be less than or equal to N"))
    end
    p = Vector(1:N)
    p[k] = l
    p[l] = k
    return Tuple(p)
end

@doc raw"""
    swap(k::Int, l::Int, p::NTuple)

Given `p`, return the `Tuple` with elements `k` and `l` swapped. Note, `k` and `l` must both 
be less than or equal to `length(p)`. 
"""
function swap(k::Int, l::Int, p::NTuple{N}) where {N}
    if !(k <= N && l <= N)
        throw(ArgumentError("k and l must be less than or equal to N"))
    end
    v = Vector{Int}([p...])
    v[k] = p[l]
    v[l] = p[k]
    return typeof(p)(v)
end

delta(x, y) = ==(x, y)
function delta(z...)
    for i in z
        !delta(z[1], i) && return false
    end
    return true
end

# type stable
function delta(T::Type{<:Number}, dim1::Int, dim2::Int, dims::Vararg{Int,N}) where {N}
    a = Array{T}(undef, dim1, dim2, dims...)
    for (i, _) in pairs(a)
        a[i] = T(delta(i))
    end
    return a
end

delta(I::CartesianIndex) = delta(Tuple(I)...)

function delta(T::Type{<:Number}, cod::VectorSpace, dom::VectorSpace)
    return TensorMap(s -> delta(T, s...), cod, dom)
end

function slice(ind::NTuple{N,Int}, sp::ProductSpace{S,N₁}) where {N,N₁,S<:IndexSpace}
    return *(one(S), one(S), (sp[i] for i in ind)...)
end

function shownested(io::IO, m::MIME"text/plain", sim::T) where {T}
    level = get(io, :nesting_level, 0)

    print(io, showheader(T))

    ion = IOContext(io, :nesting_level => level + 1)
    for name in fieldnames(T)
        println(ion, "")
        print(ion, "  "^(level + 1), name, ": ")
        show(ion, m, getfield(sim, name))
    end
    return nothing
end
showheader(x::DataType) = "$x"

function safepinv(t::AbstractTensorMap; kwargs...)
    try
        return mpinv(t; alg=TensorKit.SDD(), kwargs...)
    catch _
        @warn "Pseudo-inverse failed using `gesdd` trying `gesvd`..."
        return mpinv(t; alg=TensorKit.SVD(), kwargs...)
    end
end

function mpinv(t::AbstractTensorMap; alg=TensorKit.SDD(), kwargs...)
    calg = convert_svd_alg(alg)
    if sectortype(t) === Trivial
        return TensorMap(
            mpinv(block(t, Trivial()); alg=calg, kwargs...), domain(t) ← codomain(t)
        )
    else
        data = empty(t.data)
        for (c, b) in blocks(t)
            data[c] = mpinv(b; alg=calg, kwargs...)
        end
        return TensorMap(data, domain(t) ← codomain(t))
    end
end

convert_svd_alg(::TensorKit.SDD) = LinearAlgebra.DivideAndConquer()
convert_svd_alg(::TensorKit.SVD) = LinearAlgebra.QRIteration()
convert_svd_alg(::LinearAlgebra.DivideAndConquer) = LinearAlgebra.DivideAndConquer()
convert_svd_alg(::LinearAlgebra.QRIteration) = LinearAlgebra.QRIteration()

function mpinv(
    A::AbstractMatrix{T};
    atol::Real=0.0,
    rtol::Real=(eps(real(float(oneunit(T)))) * min(size(A)...)) * iszero(atol),
    alg=LinearAlgebra.DivideAndConquer(),
) where {T}
    m, n = size(A)
    Tout = typeof(zero(T) / sqrt(oneunit(T) + oneunit(T)))
    if m == 0 || n == 0
        return similar(A, Tout, (n, m))
    end
    if isdiag(A)
        indA = diagind(A)
        dA = view(A, indA)
        maxabsA = maximum(abs, dA)
        tol = max(rtol * maxabsA, atol)
        B = fill!(similar(A, Tout, (n, m)), 0)
        indB = diagind(B)
        B[indB] .= (x -> abs(x) > tol ? pinv(x) : zero(x)).(dA)
        return B
    end
    SVD = svd(A; alg=alg)
    tol2 = max(rtol * maximum(SVD.S), atol)
    Stype = eltype(SVD.S)
    Sinv = fill!(similar(A, Stype, length(SVD.S)), 0)
    index = SVD.S .> tol2
    Sinv[index] .= pinv.(view(SVD.S, index))
    return SVD.Vt' * (Diagonal(Sinv) * SVD.U')
end

permutebasis(mat::AbstractMatrix, perm) = only(permutebasis((mat,), perm))
function permutebasis(allmat, perm)
    @assert allequal(size, allmat)

    n, m = size(first(allmat))

    @assert n == m

    pmat = zeros(n, n)

    for i in 1:n
        pmat[i, perm[i]] = 1
    end

    return map(x -> pmat' * x * pmat, allmat)
end
