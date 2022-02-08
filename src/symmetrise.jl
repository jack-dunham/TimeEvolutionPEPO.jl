@doc raw"""
    find_charges(g)
Given the vectorised form of a group generator, `g`, outputs a list of the charges and 
their respective degeneracies.
"""
function find_charges(g)
	chrgs = zeros(Int64,2,2,2,2)
	degn = Dict{Int64, Int64}([])
	for i in 1:2, j in 1:2, k in 1:2, l in 1:2
		chg = g[i,j,k,l,i,j,k,l]
		chrgs[i,j,k,l] = Int64(chg) #convert to integer 
		if haskey(degn, chg)
			degn[chg] = degn[chg]+1
		else
			degn[chg] = 1
		end
	end
	return chrgs, degn
end
