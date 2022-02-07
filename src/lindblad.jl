"""
    dissipator(n::Int, l)

Construct the `n`-site vectorised dissipator corresponding to a single Lindblad 
operator `l` acting on a two-dimensional one-local Hilbert space represented by ℂ^2. The 
Lindblad operator argument `l` defaults to the Pauli-lowering operator (σ^{-}).
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
