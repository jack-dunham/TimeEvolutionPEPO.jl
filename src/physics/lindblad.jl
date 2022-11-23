struct Trotter <: TensorStyle end

abstract type AbstractSuperOperator{L<:LatticeStyle,O<:AbstractTensorMap} end

struct SuperOperator{T<:Union{Trotter,MPO},L<:LatticeStyle,O<:AbstractTensorMap} <:
       AbstractSuperOperator{L,O}
    x::Lattice{T,L,O}
    y::Lattice{T,L,O}
end

const SuperOperatorTrotter{L,O<:AbsTen{N,N} where {N}} = SuperOperator{Trotter,L,O}
const SuperOperatorMPO{L,O<:AbsTen{3,3}} = SuperOperator{MPO,L,O}

locality(::SuperOperatorTrotter{<:LatticeStyle,<:AbsTen{N,N}}) where {N} = Int(N / 2)
locality(::Lattice{Trotter,<:LatticeStyle,<:AbsTen{N,N}}) where {N} = Int(N / 2)

# fuck knows if any of this works

function to_mpo(L::SuperOperatorTrotter)
    tx = _to_mpo(L.x)
    ty = permutedims(_to_mpo(permutedims(L.y)))
    return SuperOperator(tx, ty)
end

function _to_mpo(mat::Lattice{Trotter,L,O}) where {L,O}
    # The locality is the number of sites the map acts on.
    l = locality(mat)
    local mpos
    for y in 1:l
        m = decompose(mat[1, y]) # Get A1, A2, A3, etc 
        if y == 1
            mpos = Matrix{eltype(m)}(undef, l, l)
        end
        mpos[:, y] = m

        for x in 2:l # Look at the next gate in unit cell 
            m = decompose(mat[x, y]) # decompose this

            circshift!(m, x - 1)
            mpos[:, y] = contract_along_ph.(m, mpos[:, y])
        end
    end
    Size = Infinite{l,l}()

    return Lattice(MPO(), Size, mpos)
end

# Add dummy indices to edge tensors.
function to_mpo_form(t::AbstractTensorMap{<:IndexSpace,2,3})
    cod = codomain(t)
    return TensorMap(t.data, insertunit(cod, 1), domain(t))
end
function to_mpo_form(t::AbstractTensorMap{<:IndexSpace,3,2})
    dom = domain(t)
    return TensorMap(t.data, codomain(t), insertunit(dom, 1))
end

function contract_along_ph(
    m_d::AbstractTensorMap{<:IndexSpace,3,3}, m_u::AbstractTensorMap{<:IndexSpace,3,3}
)
    m_d = mpo_permute_for_down(m_d)
    m_u = mpo_permute_for_up(m_u)

    m_du = permute(m_d * m_u, (5, 1, 2, 3), (6, 4, 7, 8))
    m = merge_bonds(m_du)
    return m
end

function mpo_permute_for_up(m::AbstractTensorMap{<:IndexSpace,3,3}) # bond ind on both
    return permute(m, (2, 3), (1, 4, 5, 6))
end
function mpo_permute_for_down(m::AbstractTensorMap{<:IndexSpace,3,3}) # bond ind on both
    return permute(m, (1, 2, 3, 4), (5, 6))
end

# (1 2 3 4; 5 6 7 8) -> ((1,2) 3 4; (5,6) 7 8)
function merge_bonds(m::AbstractTensorMap{<:IndexSpace,4,4}) # merge (1,2) and (5,6)
    cod = codomain(m)
    dom = domain(m)

    sp_l = fuse(cod[1] * cod[2])
    sp_r = fuse(dom[1] * dom[2])

    return TensorMap(m.data, sp_l * cod[3] * cod[4], sp_r * dom[3] * dom[4])
end

# Decompose a super operator map tensor into an MPO L -> L1 - L2 - L3 ...
function decompose(t::AbstractTensorMap)
    # Initialisation
    U1, V1 = liouvillian_split_initial(t)
    # Add dummy leg to the left-most tensor
    M1 = to_mpo_form(U1)::TensorMap{<:IndexSpace,3,3}
    Ms = Vector{typeof(M1)}([])
    push!(Ms, M1)

    # Recursively split the tensor
    Ms, Vl = liouvillian_split(Ms, V1)

    # Add dummy leg to the right most tensor
    Ml = to_mpo_form(permute(Vl, (1, 2, 3), (4, 5)))::TensorMap{<:IndexSpace,3,3}

    # Append right-most tensor to the vector
    push!(Ms, Ml)
    return Ms
end

function liouvillian_split_initial(t)
    N = numin(t) # We have: numin(t) = numout(t). 

    # Put all the indices corresponding to the first site into the codomain (and only these
    # indices.)
    U, S, V, err = tsvd(
        t, (1, 2, N + 1, N + 2), (3:N..., ((N + 3):(2N))...); trunc=truncdim(4)
    )

    @info "MPO truncation error: $err"

    # We need to do something with the matrix of singular values.
    Up = U * sqrt(S)
    Vp = sqrt(S) * V

    # Return indices to their rightful places: [D₁ D₂; E₁ N₁ N₂].
    Upp = permute(Up, (1, 2), (5, 3, 4))
    return Upp, Vp
end

function liouvillian_split(vec, v::AbstractTensorMap{<:IndexSpace,1})
    # The only codomain index is assumed to be bond index. We have four indices per site 
    # (two ket and two bra) which all currently live in the domain.
    l = Int(numin(v) / 4)
    N = 2 * l

    # Recursion terminates when we only have one site's worth of indices left to split, as 
    # then the tensor requires no further decomposition.
    if l == 1
        return vec, v
    else
        # Put all the indices corresponding to the first site into the codomain as well as 
        # the only bond index (first index and only index already in codomain). The 
        # latter index will become the left (west) bond index for the MPO formed from U.
        # All the remaining indices stay in the domain. 
        U, S, V, err = tsvd(
            v,
            (1, 2, 3, N + 2, N + 3),
            ((4:(N + 1))..., ((N + 4):(2N + 1))...);
            trunc=truncerr(1e-12),
        )
        @info "MPO truncation error: $err"

        # We need to do something with the matrix of singular values.
        Up = U * sqrt(S)
        Vp = sqrt(S) * V

        # Return indices to their rightful places: [W₁ D₁ D₂; E₁ N₁ N₂]
        Upp = permute(Up, (1, 2, 3), (6, 4, 5))

        # Add this MPO tensor to the list.
        push!(vec, Upp)

        # Now split the remaining N - 1 sites!
        return liouvillian_split(vec, Vp)
    end
end

#FIX:
@doc raw"""
    dissipator(n::Int, l)

Construct the `n`-site vectorised dissipator corresponding to a single 1-local Lindblad 
operator `l` acting on a 2-dimensional Hilbert space represented by $ℂ^2$. The Lindblad 
operator argument `l` defaults to the Pauli-lowering operator $σ^{-}$.
"""
function local_dissipator(N::Int, L::AbstractMatrix=σm)
    L_list = [one(L) for _ in 1:N]
    L_list[1] = L

    Li = Vector(undef, N)

    Ds = Vector{Matrix{eltype(L)}}([])

    for i in 1:N
        if N == 1
            Li[1] = L_list[1]
        else
            Li[i] = kron(circshift(L_list, i - 1)...)
        end
        rv =
            1 / 2 * (
                2 * (kron(conj(Li[i]), Li[i])) - kron(one(Li[i]), (Li[i]' * Li[i])) -
                kron(transpose(Li[i]) * conj(Li[i]), one(Li[i]))
            )
        push!(Ds, rv)
    end
    return sum(Ds) / N
end
# function dissipator(n::Int; l=σm)
#     L = Dict([])

#     Llist = [one(l) for _ in 1:n]
#     Llist[1] = l

#     Ds = []

#     for i in 1:n
#         if n == 1
#             L[1] = Llist[1]
#         else
#             L[i] = ⊗(circshift(Llist, i - 1)...)
#         end
#         push!(
#             Ds,
#             2 * (cj(L[i]) ⊗ L[i]) - one(L[i]) ⊗ (L[i]' * L[i]) -
#             (cj(L[i])' * cj(L[i])) ⊗ one(L[i]),
#         )
#     end
#     return sum(Ds)
# end

@doc raw"""
    coherent(H::AbstractTensorMap)

Construct the super-operator $\mathcal{H}$ corresponding to the coherent part of 
the Liouvillian super-operator. That is, given a 2-local $H_{\alpha, \beta}$ provided by 
`H`, constructs the `TensorMap`
```math
\mathcal{H}_{\alpha,\beta} = - i (\mathbb{I}_{\alpha, \beta} \otimes H_{\alpha,\beta} 
- H^{T}_{\alpha,\beta} \otimes \mathbb{I}_{\alpha, \beta})
```
where $\alpha$ and $\beta$ are lattice indices corresponding to any two adjacent
lattice sites.
"""
coherent(H::AbstractTensorMap) = -im * (one(H) ⊗ H - cj(H)' ⊗ one(H))

coherent(H::AbstractMatrix) = -im * (kron(one(H), H) - kron(transpose(H), one(H)))

#TODO: Move to utils.jl
# (1 2 3 4 ... N) -> (1 N/2 2 N/2 + 1 ...)
# Used for the following vector space reordering:
# (K₁ ⊗ K₂ ⊗ ⋯  ⊗ Kₙ) ⊗ (H₁ ⊗ H₂ ⊗ ⋯  ⊗ Hₙ) -> (K₁ ⊗ H₁) ⊗ (K₂ ⊗ K₂) ⊗ ⋯  ⊗ (Kₙ ⊗ Hₙ)
# i.e. pairs ket and bra sectors for each unit cell together.
function interlace_perm(::Val{N}) where {N}
    p = (isodd(x) ? Int((x + 1) / 2) : Int(x / 2) + N for x in 1:(2N))
    return Tuple(p)
end

function liouvillian(
    spaces::NTuple{N,S}, H::T, L::NTuple{M,T}
) where {N,M,S<:IndexSpace,T<:AbstractMatrix}
    # n = Int(log2(size(H)[1]))
    data = coherent(H)
    for l in L
        data = data + local_dissipator(N, l)
    end

    sp = ⊗((adjoint.(spaces))..., spaces...)
    t = TensorMap(data, sp, sp)

    p = interlace_perm(Val(N))

    return LiouvillianMapTensor(permute(t, p, 2N .+ p))
end
