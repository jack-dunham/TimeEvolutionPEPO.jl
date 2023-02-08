struct Trotter <: TensorStyle end

abstract type AbstractSuperOperator{L<:AbstractLattice,O<:AbstractTensorMap} end

struct SuperOperator{T<:Union{Trotter,MPO},L<:AbstractLattice,O<:AbstractTensorMap} <:
       AbstractSuperOperator{L,O}
    x::OnLattice{T,L,O}
    y::OnLattice{T,L,O}
end

const SuperOperatorTrotter{L,O<:AbsTen{N,N} where {N}} = SuperOperator{Trotter,L,O}
const SuperOperatorMPO{L,O<:AbsTen{3,3}} = SuperOperator{MPO,L,O}

locality(::SuperOperatorTrotter{<:AbstractLattice,<:AbsTen{N,N}}) where {N} = Int(N / 2)
locality(::OnLattice{Trotter,<:AbstractLattice,<:AbsTen{N,N}}) where {N} = Int(N / 2)

function SuperOperator(
    lattice,
    data_x::AbstractMatrix{<:AbstractMatrix},
    data_y::AbstractMatrix{<:AbstractMatrix}=data_x,
)
    tensors_x = mats2tensors(lattice, data_x)
    tensors_y = mats2tensors(permutedims(lattice), data_y)

    return SuperOperator(tensors_x, tensors_y)
end

function SuperOperator(lattice, a::NTuple{2,AbstractMatrix{<:Number}})
    return SuperOperator(lattice, a[1], a[2])
end
function SuperOperator(
    lattice, data_x::AbstractMatrix{<:Number}, data_y::AbstractMatrix{<:Number}=data_x
)
    s = size(lattice)
    data_x_mat = fill(data_x, s)
    data_y_mat = fill(data_y, s)
    return SuperOperator(lattice, data_x_mat, data_y_mat)
end

function mats2tensors(lattice, data)
    L = lattice
    nx, ny = size(L)
    spaces = OnLattice(NotTensor(), lattice, lattice.localspace)

    #WARN: Not type stable! Hacky...not general
    locality = Int(log(4, size(data[1, 1])[1]))

    tensors = [
        TensorMap(
            data[x, y],
            getsuperspace(spaces, x, y, Val(locality)),
            getsuperspace(spaces, x, y, Val(locality)),
        ) for x in 1:nx, y in 1:ny
    ]

    return OnLattice(Trotter(), L, interlace_permute.(tensors))
end

function getsuperspace(spaces, x, y, ::Val{l}) where {l}
    # ket_space = circshift(spaces[:, y], -x + 1)
    ket_space = [circshift(spaces, (-x + 1, 0))[i, y] for i in 1:l]
    return *(adjoint.(ket_space)..., ket_space...)
end

# fuck knows if any of this works

function to_mpo(L::SuperOperatorTrotter)
    tx = _to_mpo(L.x)
    ty = permutedims(_to_mpo(permutedims(L.y)))
    return SuperOperator(tx, ty)
end

function _to_mpo(mat::OnLattice{Trotter,L}) where {L}
    l = locality(mat)
    nx, ny = size(mat)

    m = decompose.(mat)

    println("up: ", space(mat[1, 1]))
    println("down: ", space(mat[2, 1]))
    rv = Matrix{eltype(m[1])}(undef, nx, ny)

    # We need a MPO for each trotter gate in the unit cell.
    for y in 1:ny
        for x in 1:nx
            rv[x, y] = m[x, y][1] # Periodic in first indices
            for i in 1:(l - 1)
                M_x = m[x + i, y][i + 1]
                rv[x, y] = contract_along_ph(M_x, rv[x, y])
            end
        end
    end
    return OnLattice(MPO(), L(), rv)
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

# Simple decomposition of a two-local operator (1 2 3 4)(5 6 7 8), with truncation
function twolocal_decompose(t::AbstractTensorMap)
    U, S, V = tsvd(t, (1, 2, 5, 6), (3, 4, 7, 8); trunc=truncerr(1e-10))
    L = permute(U * sqrt(S), (1, 2, 5), (3, 4))
    R = permute(sqrt(S) * V, (1, 2, 3), (4, 5))
    return L, R
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

    @info "MPO truncation error: $err, with bond dimension $(dim(codomain(S)))"

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
            trunc=truncerr(1e-10),
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

#TODO: Move to utils.jl
# (1 2 3 4 ... N) -> (1 N/2 2 N/2 + 1 ...)
# Used for the following vector space reordering:
# (K₁ ⊗ K₂ ⊗ ⋯  ⊗ Kₙ) ⊗ (H₁ ⊗ H₂ ⊗ ⋯  ⊗ Hₙ) -> (K₁ ⊗ H₁) ⊗ (K₂ ⊗ K₂) ⊗ ⋯  ⊗ (Kₙ ⊗ Hₙ)
# i.e. pairs ket and bra sectors for each unit cell together.
function interlace_perm(::Val{N}) where {N}
    p = (isodd(x) ? Int((x + 1) / 2) : Int(x / 2) + Int(N / 2) for x in 1:(N))
    return Tuple(p)
end

function interlace_permute(t::AbstractTensorMap{<:IndexSpace,N,N}) where {N}
    perm = interlace_perm(Val(N))
    return permute(t, perm, N .+ perm)
end

# function liouvillian(
#     space::BondSpace{<:LatticeStyle{Nx,Ny}},
#     H::T, # need the size of this in a type stable way.
#     L::NTuple{M,T},
# ) where {Nx,Ny,M,T<:AbstractMatrix}
#     map_x = _liouvillian(space, H, L)
#     map_y = _liouvillian(permutedims(space, (2, 1)), H, L)
#     return SuperOperator(map_x, map_y)
# end
#
# function _liouvillian(
#     space::BondSpace{<:LatticeStyle{Nx,Ny}}, H::T, L::NTuple{M,T}
# ) where {Nx,Ny,M,T<:AbstractMatrix}
#     # n = Int(log2(size(H)[1]))
#
#     data = coherent(H)
#     for l in L
#         data = data + local_dissipator(Nx, l)
#     end
#
#     p = interlace_perm(Val(Nx))
#
#     maps = Matrix(undef, Nx, Ny)
#
#     for y in 1:Ny
#         spaces = space[:, y]
#         for x in 1:Nx
#             spaces_x = circshift(spaces, 1 - x)
#
#             sp = ⊗((adjoint.(spaces_x))..., spaces...)
#
#             t_x = TensorMap(data, sp, sp)
#
#             maps[x, y] = permute(t_x, p, 2 * Nx .+ p)
#         end
#     end
#
#     return maps
# end
# function _liouvillian(
#     spaces::NTuple{N,S}, H::T, L::NTuple{M,T}
# ) where {N,M,S<:IndexSpace,T<:AbstractMatrix}
#     # n = Int(log2(size(H)[1]))
#     data = coherent(H)
#     for l in L
#         data = data + local_dissipator(N, l)
#     end

#     sp = ⊗((adjoint.(spaces))..., spaces...)
#     t = TensorMap(data, sp, sp)

#     p = interlace_perm(Val(N))

#     map = permute(t, p, 2N .+ p)

#     return map
# end
