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
function permutecod(t::CompositeTensor{2}, p::NTuple{M,Int}) where {M}
    return CompositeTensor(ITC.mapbefore(i -> permutecod(i, p), t))
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
function permutedom(t::CompositeTensor{2}, p::NTuple{M,Int}) where {M}
    return CompositeTensor(ITC.mapbefore(i -> permutedom(i, p), t))
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

function contractphysical(tsrc::ITC.AbsTen{2,4,S}) where {S}
    tdst = similar(tsrc, one(S), domain(tsrc))
    return contractphysical!(tdst, tsrc)
end
function contractphysical(tsrc_1::ITC.AbsTen{2,4,S}, tsrc_2::ITC.AbsTen{4,2,S}) where {S}
    DomType = ProductSpace{S,8}

    outdom = DomType(map((x, y) -> x * y', domain(tsrc_1), codomain(tsrc_2)))
    out = similar(tsrc_1, one(S), outdom)

    @tensoropt out[D1 D5 D2 D6 D3 D7 D4 D8] =
        tsrc_1[k b; D1 D2 D3 D4] * tsrc_2[D5 D6 D7 D8; k b]

    return out
end
function contractphysical!(tdst::AbsTen{0,4}, tsrc::AbsTen{2,4})
    @tensoropt tdst[1 2 3 4] = tsrc[x x; 1 2 3 4]
    return tdst
end

function mergevirtual(t::ITC.AbsTen{0,8,S}) where {S}
    DomType = ProductSpace{S,4}

    outdom = DomType(map(i -> fuse(domain(t, i) * domain(t, i + 1)), 1:2:8))

    return TensorMap(t.data, codomain(t), outdom)
end

function contractphysical_mergevirtual(tsrc_1::AbsTen{2,4}, tsrc_2::AbsTen{4,2})
    contracted_tensor = contractphysical(tsrc_1, tsrc_2)
    return mergevirtual(contracted_tensor)
end

function hermitianize!(t::AbsTen{2,4})
    t_adj = cj(permutecod(t, (2, 1)))
    dual_swap = isometry(codomain(t), codomain(t_adj))
    axpby!(1//2, dual_swap * t_adj, 1//2, t)
    return t
end

function posapprox!(t::AbsTen{N,N}; tol=1e3 * eps(), verbose=false) where {N}
    tol ≥ -Inf || return t

    if verbose
        told = deepcopy(t)
    end

    D, V = eigh!(t)

    for (_, v) in blocks(D)
        T = scalartype(D)
        @assert T <: Real
        v[v .< tol] .= zero(T)
    end

    copy!(t, V * D * V')

    if verbose
        normalize!(told)
        err = norm(told - normalize(t))
        @info "positive approximation error $err from tolerance $tol"
    end

    return t
end
