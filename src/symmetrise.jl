#### REDUNDENT


@doc raw"""
    struct SymmetryGroup{G<:Group, N::Integer}
        
Stores the charges and degeneracies?
TODO: docstring
"""
struct SymmetryGroup{G<:Group}
    charges::Array{<:AbstractIrrep{G},4}
    degeneracy::Dict{<:AbstractIrrep{G},Integer}
    basis::GradedSpace{<:AbstractIrrep{G}}
end

@doc raw"""
    SymmetryGroup{G}(g::TensorMap{ComplexSpace}) where {G<:Group}

Constructs a `SymmetryGroup` type given a group `G` and a group generator `g`.
"""
function SymmetryGroup(
    g::TensorMap{ComplexSpace}, basis::GradedSpace{<:AbstractIrrep{G}}
) where {G<:Group}
    return SymmetryGroup{G}(find_charges(G, g)..., basis)
end

struct Liouvillian{
    S<:IndexSpace,
    I<:Sector,
    A<:Union{TensorKit.SectorDict{I,<:DenseMatrix},<:DenseMatrix},
    F₁,
    F₂,
} <: AbstractPhysicalTensor{S,4,4,I,A,F₁,F₂}
    tensor::TensorMap{S,4,4,I,A,F₁,F₂}
    mapping
end

const SymmetricLiouvillian{S<:GradedSpace,I<:Sector,A<:TensorKit.SectorDict{I,<:DenseMatrix},F₁,F₂} = Liouvillian{S,I,A, F₁,F₂}

function SymmetricLiouvillian(t::TensorMap{S,4,4,I, A, F₁, F₂}, mapping ) where {S<:GradedSpace, I<:Sector, A<:TensorKit.SectorDict{I,<:DenseMatrix}, F₁,F₂}
    return SymmetricLiouvillian{S,I,A,F₁,F₂}(t, mapping)
end

const StandardLiouvillian{A<:DenseMatrix} = Liouvillian{ComplexSpace,Trivial,A,Nothing,Nothing}

function StandardLiouvillian(
    t::TensorMap{ComplexSpace,4,4,Trivial,A,Nothing,Nothing}, nothing
) where {A<:DenseMatrix}
    return StandardLiouvillian{A}(t, nothing)
end

@doc raw"""
    find_charges(G, g)

Given the vectorised form of a group generator, `g`, outputs a list of the charges and 
their respective degeneracies. Assumes `g` is diagonalised (i.e. assumes that the basis used
is one such that `g` is diagonal), and that each diagonal element of `g` corresponds to a 
valid irrep label of the group `G`. 
"""
function find_charges(G::Type{<:TensorKit.Group}, g)
    chrgs = Array{Irrep[G]}(undef, 2, 2, 2, 2)
    degn = Dict{Irrep[G],Integer}([])
    for i in 1:2, j in 1:2, k in 1:2, l in 1:2
        chg = g[i, j, k, l, i, j, k, l]
        irrep = Irrep[G](Integer(chg))
        chrgs[i, j, k, l] = irrep # TODO: Convert `chg` to (half)-integer 
        if haskey(degn, irrep)
            degn[irrep] = degn[irrep] + 1
        else
            degn[irrep] = 1
        end
    end
    return chrgs, degn
end

@doc raw"""
    find_blocks(chrgs, degn, L::TensorMap{ComplexSpace, 4, 4})

Outputs the blocks corresponding to each charge sector of a symmetric Liouvillian. 
"""
function find_blocks(SG::SymmetryGroup{G}, L::StandardLiouvillian) where {G<:Group}
    L = L
    chrgs = SG.charges
    degn = SG.degeneracy
    blocks = Dict{Irrep[G],Matrix{ComplexF64}}([])
    cols = Dict{Irrep[G],Vector{Vector{ComplexF64}}}([])
    mapping = Dict{Irrep[G],Vector{CartesianIndex}}([])
    for key in keys(degn)
        cols[key] = []
        mapping[key] = []
    end
    # For each input state, find the linear combination of output states that this state
    # maps to. Output states should have the same charge as the input state.
    for i in 1:2, j in 1:2, k in 1:2, l in 1:2 # Input state
        chrg = chrgs[i, j, k, l]
        col = Vector{ComplexF64}()
        for a in 1:2, b in 1:2, c in 1:2, d in 1:2 # Output state
            if chrgs[a, b, c, d] == chrg
                push!(col, L[a, b, c, d, i, j, k, l])
            else
                # Check if symmetry-required non-zero elements are actually zero. If not
                # then L is not block diagonal so throw an error.
                if !(L[a, b, c, d, i, j, k, l] ≈ zero(L[a, b, c, d, i, j, k, l]))
                    throw(
                        ArgumentError(
                            "Liouvillian has non-zero elements at incompatible positions"
                        ),
                    )
                end
            end
        end
        push!(cols[chrg], col)
        push!(mapping[chrg], CartesianIndex(i, j, k, l))
    end
    for chrg in keys(degn)
        col = cols[chrg]
        if length(col) == 1
            ele = reshape(col[1], 1, 1)
            blocks[chrg] = ele
        else
            blocks[chrg] = reduce(hcat, col)
        end
    end
    return blocks, mapping
end

function symmetrise(SG::SymmetryGroup{<:Group}, L::StandardLiouvillian)
    blcks, mapping = find_blocks(SG, L)
    lin = TensorMap(zeros, ComplexF64, SG.basis^4, SG.basis^4)
    for i in keys(SG.degeneracy)
        blocks(lin)[i] = blcks[i]
    end
    return SymmetricLiouvillian(lin, mapping)
end

function unsymmetrise(L::SymmetricLiouvillian)
    tensor = L.tensor
    map = L.mapping
    blcks = blocks(tensor)
    cL = zeros(ComplexF64, 2, 2, 2, 2, 2, 2, 2, 2)
    # cL = TensorMap(zeros, ComplexF64, ℂ^4, ℂ^4)
    for (irrep, block) in blcks
        irrepmap = map[irrep]
        for i in CartesianIndices(block)
            row = i[1]
            col = i[2]
            cL[irrepmap[row], irrepmap[col]] = block[i]
        end
    end
    return StandardLiouvillian(
        TensorMap(cL, ℂ^2 * ℂ^2 * ℂ^2 * ℂ^2, ℂ^2 * ℂ^2 * ℂ^2 * ℂ^2), nothing
    )
end

#TODO: nothing -> identity mapping
