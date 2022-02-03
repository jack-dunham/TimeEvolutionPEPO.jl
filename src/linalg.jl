# Linear algebra functions

"""
	compose(t1,t2)

Composes two linear maps of type `TensorMap` by contracting the domain of `t1` with the 
codomain of `t2`. This corresponds to applying the linear map t2 followed by the linear 
map t1.

"""
function compose(t1::AbstractTensorMap, t2::AbstractTensorMap)
    No₁ = length(codomain(t1))
    Nc₁ = length(domain(t1))

    Nc₂ = length(codomain(t2))
    No₂ = length(domain(t2))

    if Nc₁ != Nc₂
        throw(
            DimensionMismatch(
                "Number of indices in the domain of t1 does not match the number of" *
                "indices in the codomain of t2",
            ),
        )
    end

    tO = TensorMap(zeros, codomain(t1), domain(t2))

    oindA = Tuple(1:No₁)
    cindA = Tuple((No₁ + 1):(No₁ + Nc₁))

    cindB = Tuple(1:Nc₂)
    oindB = Tuple((Nc₂ + 1):(Nc₂ + No₂))

    indleft = Tuple(1:No₁)
    indright = Tuple((No₁ + 1):(No₁ + No₂))

    rv = TensorKit.TO.contract!(
        1, t1, :N, t2, :N, 0, tO, oindA, cindA, oindB, cindB, indleft, indright
    )
    return rv
end

function LinearAlgebra.tr(t::PEPOTensor{S}) where {S<:IndexSpace}
    t = t.tensor
    @tensor tTr[1 2; 3 4] := t[1 2 x; 3 4 x]
end

function LinearAlgebra.ishermitian(p::PEPOTensor; phys_only=false)
    t = p.tensor
    phys_only || return LinearAlgebra.ishermitian(t)
    tT = permute(t, (1, 2, 6), (4, 5, 3))
    for (key, block) in blocks(t)
        block ≈ conj(blocks(tT)[key]) || return false
    end
    return true
end

function LinearAlgebra.vec(p::PEPOTensor)
    t = p.tensor 
    return _vec(t)
end

function _vec(t::TensorMap{S,3,3}) where {S<:EuclideanSpace}
    dom = TensorKit.domain(t)
    in = dom[3] # Physical index corresponding to the input
    cod = TensorKit.codomain(t)
    out = cod[3] # Physical index corresponding to the output
    tt = permute(t, (1,2,3,6),(4,5))
    u = unitary(fuse(out ⊗ in), out ⊗ flip(in))
    return @tensor ttt[1 2 ph; 3 4]:=u[ph; x1 x2 ]*tt[1 2 x1 x2; 3 4] 
end

