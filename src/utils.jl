# Convenience const. 
const AbsTen{N₁,N₂,S} = AbstractTensorMap{S,N₁,N₂}
const TenAbs{N₂,N₁,S} = AbstractTensorMap{S,N₁,N₂}

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

westbond(t::AbstractTensorMap) = codomain(t)[1]
southbond(t::AbstractTensorMap) = codomain(t)[2]
downbond(t::AbstractTensorMap) = codomain(t)[3]

eastbond(t::AbstractTensorMap) = domain(t)[1]
northbond(t::AbstractTensorMap) = domain(t)[2]
upbond(t::AbstractTensorMap) = domain(t)[3]

# Alt permute() 
function _permute(t::AbstractTensorMap{<:IndexSpace,N,M}, p::Tuple) where {N,M}
    if !(length(p) == N + M)
        throw(DimensionMismatch())
    else
        return permute(t, p[1:N], p[(N + 1):(N + M)])
    end
end

# Swap codomain and domain (Base.transpose reverses indices)
function _transpose(tsrc::AbsTen{N,M}) where {N,M}
    p1 = NTuple{M,Int}((N + 1):(N + M))
    p2 = NTuple{N,Int}(1:N)
    return permute(tsrc, p1, p2)
end

# TODO move to ITC
function permutecod(t::ITC.TensorPair, p::NTuple{M,Int}) where {M}
    return ITC.TensorPair(permutecod(t.top, p), permutecod(t.bot, p))
end

function permutecod(t::AbsTen{N,M}, p::NTuple{N,Int}) where {N,M}
    p2 = NTuple{M,Int}((N + 1):(M + N))
    return permute(t, p, p2)
end

function permutecod!(tdst, tsrc::AbsTen{N,M}, p::NTuple{N,Int}) where {N,M}
    p2 = NTuple{M,Int}((N + 1):(M + N))
    return permute!(tdst, tsrc, (p, p2))
end

# TODO move to ITC
function permutedom(t::ITC.TensorPair, p::NTuple{M,Int}) where {M}
    return ITC.TensorPair(permutedom(t.top, p), permutedom(t.bot, p))
end

function permutedom(t::AbsTen{N,M}, p::NTuple{M,Int}) where {N,M}
    p1 = NTuple{N,Int}(1:N)
    p2 = map(x -> x + N, p)
    return permute(t, p1, p2)
end

function permutedom!(tdst, tsrc::AbsTen{N,M}, p::NTuple{M,Int}) where {N,M}
    p1 = NTuple{N,Int}(1:N)
    p2 = map(x -> x + N, p)
    return permute!(tdst, tsrc, (p1, p2))
end

function _leftorth(t::AbsTen{N,2}) where {N}
    p1 = NTuple{N + 1,Int}((1:N..., N + 2))
    p2 = NTuple{N,Int}(1:N)

    Q, R = leftorth(t, p1, (N + 1,))

    t_out = permute(Q, p2, (N + 2, N + 1))
    r_out = permute(R, (), (2, 1))

    return t_out, r_out
end

function _rightorth(t::AbsTen{N,2}) where {N}
    p1 = NTuple{N + 1,Int}((1:N..., N + 1))
    p2 = NTuple{N,Int}(2:(N + 1))

    L, Q = rightorth(t, (N + 2,), p1)

    t_out = permute(Q, p2, (N + 2, 1))
    l_out = permute(L, (), (2, 1))

    return l_out, t_out
end

ftsvd(t::TensorMap; trunc::TensorKit.TruncationScheme=notrunc()) = tsvd(t; trunc=trunc)
function ftsvd(t::TensorMap{ComplexSpace}; trunc::TensorKit.TruncationSpace)
    T = eltype(t)

    d = trunc.space

    S = TensorMap(zeros, T, d, d)

    U = TensorMap(zeros, T, codomain(t), codomain(S))
    V = TensorMap(zeros, T, domain(S), domain(t))

    u, s, v = psvd(t.data; rank=dim(d))

    ci = x -> CartesianIndices(x)

    ds = Diagonal(s)

    copyto!(U.data, ci(u), u, ci(u))
    copyto!(S.data, ci(ds), ds, ci(ds))
    copyto!(V.data, ci(v'), v', ci(v'))

    return U, S, V
end

function hermitianize!(t::AbsTen{2,4})
    t_adj = cj(permutecod(t, (2, 1)))
    dual_swap = isometry(codomain(t), codomain(t_adj))
    axpby!(1//2, dual_swap * t_adj, 1//2, t)
    return t
end

function posapprox(t; tol=10.0 * eps())
    D, V = eigh(t)

    for (_, v) in blocks(D)
        v[v .< tol] .= 0
    end

    rv = V * D * V'

    err = norm(t - rv)

    return V * D * V', err
end

function shownested(io::IO, m::MIME"text/plain", sim::T) where {T}
    level = get(io, :nesting_level, 0)

    print(io, showheader(T))

    ion = IOContext(io, :nesting_level => level + 1)
    for name in fieldnames(T)
        println("")
        print(ion, "  "^(level + 1), name, ": ")
        show(ion, m, getfield(sim, name))
    end
    return nothing
end

# function fsvd(t::TensorMap{D}; trunc::TensorKit.TruncationSpace) where {D<:IndexSpace}
#     d = trunc.space
#
#     S = similar(t, d, d)
#
#     U = similar(t, codomain(t), codomain(S))
#     V = similar(t, domain(S), domain(t))
#
#     for c in sectors(d)
#         println(dim(c))
#         u, s, v = psvd(TensorKit.block(t, c); rank=dim(c))
#         copy!(U.data[c], u)
#         copy!(S.data[c], diagm(s))
#         copy!(V.data[c], v')
#     end
#
#     return U, S, V
# end
