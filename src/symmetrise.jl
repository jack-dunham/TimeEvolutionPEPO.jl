@doc raw"""
    find_charges(g, G)

Given the vectorised form of a group generator, `g`, outputs a list of the charges and 
their respective degeneracies. Assumes `g` is diagonalised (i.e. assumes that the basis used
is one such that `g` is diagonal), and that each diagonal element of `g` corresponds to a 
valid irrep label of the group `G`. 
"""
function find_charges(g, G::Type{<:TensorKit.Group})
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

# TODO: probably remove, not a relavent method, as it assumes the user knows which
# subspaces each state belongs to.
function find_charges(g::TensorMap{S}) where {S<:GradedSpace}
    return # TODO: Return the eigenvalues of g?
end

@doc raw"""
    find_blocks(chrgs, degn, L::TensorMap{ComplexSpace, 4, 4})

Outputs the blocks corresponding to each charge sector of a symmetric Liouvillian. 
"""
function find_blocks(chrgs, degn, L::TensorMap{ComplexSpace,4,4})
    blocks = Dict{Int64,Matrix}([])
    cols = Dict{Int64,Vector}([])
    for key in keys(degn)
        cols[key] = []
    end
    # For each input state, find the linear combination of output states that this state
    # maps to. Output states should have the same charge as the input state.
    for i in 1:2, j in 1:2, k in 1:2, l in 1:2 # Input state
        chrg = chrgs[i, j, k, l]
        col = Array{ComplexF64}([])
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
    return blocks
end

@doc raw"""
    struct SymmetryGroup{G<:Group, N::Integer}
        
Stores the charges and degeneracies?
TODO: docstring
"""
struct SymmetryGroup{G<:TensorKit.Group,N<:Integer}
    charges::Array{<:AbstractIrrep{G}}
    degeneracy::Dict{<:AbstractIrrep{G},Integer}
end

@doc raw"""
    SymmetryGroup{G}(g::TensorMap{ComplexSpace}) where {G<:TensorKit.Group}

Constructs a `SymmetryGroup` type given a group `G` and a group generator `g`.
"""
function SymmetryGroup{G}(g::TensorMap{ComplexSpace}) where {G<:TensorKit.Group} 
    return SymmetryGroup{G, Integer(log(2,numind(g)))}(find_charges(g))
end
