struct SitePEPO{T<:TensorMap}
    tensor::T
    bra::Int
    ket::Int
end

struct VectorSitePEPO{T<:TensorMap}
    tensor::T
    perm::Tuple
    bra::Int
    ket::Int
end

# function LinearAlgebra.vec(p::SitePEPO)
#     t = p.tensor
#     N₁ = numout(t)
#     N = numind(t)
#     perm = ((p.bra, Tuple(2:N₁+1)...), Tuple(N₁+2:N))  
#     return VectorSitePEPO(permute(t, perm...),perm, 1, p.ket )
# end

function LinearAlgebra.vec(p::SitePEPO)
    t = p.tensor
    N = numind(t)
    u = nswap(1, t.bra, N)
    return VectorSitePEPO(_permute(t,u), u)
end

function _vec(t::TensorMap{<:IndexSpace, N, M}) where {N,M}
    permute(t, (Tuple(N+1:N+M)..., Tuple(1:N)...)), (Tuple(M+1:N+M), Tuple(1:M))
end

function unvec(v::VectorSitePEPO)
    perm = v.perm
    inv = invperm(perm)
    return _permute(t, inv) 
end

function _permute(t::AbstractTensorMap{<:IndexSpace, N, M}, p::Tuple)  where {N,M}
    if !(length(p) == N + M) 
        throw(DimensionMismatch())
    else
        return permute(t, p[1:N], p[N+1:N+M])
    end
end

# Output a permutation of length N that results in the k'th element moved to the
# n'th position
function nswap(k::Int, n::Int, N::Int)
    if k==n
        return Tuple(1:N)
    elseif n>k 
        return(invperm(nswap(n,k,N)))
    else 
        return (Tuple(1:n-1)..., k, Tuple(n:k-1)..., Tuple(k+1:N)...)
    end
end

function swap(k::Int, l::Int, N::Int)
    if !(k<=N && l<=N) 
        throw(ArgumentError("k and l must be less than or equal to N"))
    end
    p = Vector(1:N)
    p[k] = l
    p[l] = k
    return Tuple(p)
end

