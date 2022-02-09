# TODO: Charges should be appropriate `TensorKit.Sector` objects.
@doc raw"""
    find_charges(g)

Given the vectorised form of a group generator, `g`, outputs a list of the charges and 
their respective degeneracies.
"""
function find_charges(g)
    chrgs = zeros(Int64, 2, 2, 2, 2)
    degn = Dict{Int64,Int64}([])
    for i in 1:2, j in 1:2, k in 1:2, l in 1:2
        chg = g[i, j, k, l, i, j, k, l]
        chrgs[i, j, k, l] = Int64(chg) #convert to integer 
        if haskey(degn, chg)
            degn[chg] = degn[chg] + 1
        else
            degn[chg] = 1
        end
    end
    return chrgs, degn
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
