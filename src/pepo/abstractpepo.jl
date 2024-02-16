abstract type AbstractPEPO end

issymmetric(pepo::AbstractPEPO) = issymmetric(latticetensors(pepo))
physicalspace(pepo::AbstractPEPO) = getindex.(codomain.(latticetensors(pepo)), 1)
uniqueindices(pepo::AbstractPEPO) = uniqueindices(latticetensors(pepo))

TensorKit.truncspace(p::AbstractPEPO, bond::Bond) = truncspace(getweights(p), bond)

quicknormalize!(pepo::AbstractPEPO) = normalize!.(latticetensors(pepo))

TensorKit.scalartype(pepo::AbstractPEPO) = scalartype(latticetensors(pepo))

function setweights!(pepo::AbstractPEPO, s, bond::Bond)
    copy!(getweights(pepo, bond), s)
    return pepo
end

getbonds(pepo::AbstractPEPO, args...) = getbonds(latticetensors(pepo), args...)

function changeaxis(pepo::AbstractPEPO)
    pepo_permuted = broadcast(latticetensors(pepo)) do tensor
        permute(tensor, (1, 2), (4, 3, 6, 5))
    end
    return PEPO(permutedims(pepo_permuted), changeaxis(getweights(pepo)))
end

function changeaxis!(pepo_permuted::AbstractPEPO, pepo::AbstractPEPO)
    broadcast(
        latticetensors(pepo_permuted), permutedims(latticetensors(pepo))
    ) do pdst, psrc
        permute!(pdst, psrc, ((1, 2), (4, 3, 6, 5)))
    end
    return pepo_permuted
end

function bondspace(pepo::AbstractPEPO, bond)
    return domain.(latticetensors(pepo), getaxis(bond))[left(bond)]
end

hermitianize!(pepo::AbstractPEPO) = hermitianize!.(latticetensors(pepo))
