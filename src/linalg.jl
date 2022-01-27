# Linear algebra functions

using TensorKit # temp for dev

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
