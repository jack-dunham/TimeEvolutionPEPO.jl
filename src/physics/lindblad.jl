@doc raw"""
    dissipator(n::Int, l)

Construct the `n`-site vectorised dissipator corresponding to a single 1-local Lindblad 
operator `l` acting on a 2-dimensional Hilbert space represented by $ℂ^2$. The Lindblad 
operator argument `l` defaults to the Pauli-lowering operator $σ^{-}$.
"""
function dissipator(n::Int; l=σm)
    L = Dict([])

    Llist = [one(l) for _ in 1:n]
    Llist[1] = l

    Ds = []

    for i in 1:n
        if n == 1
            L[1] = Llist[1]
        else
            L[i] = ⊗(circshift(Llist, i - 1)...)
        end
        push!(
            Ds,
            2 * (cj(L[i]) ⊗ L[i]) - one(L[i]) ⊗ (L[i]' * L[i]) -
            (cj(L[i])' * cj(L[i])) ⊗ one(L[i]),
        )
    end
    return sum(Ds)
end

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
