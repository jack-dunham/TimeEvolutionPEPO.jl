abstract type AbstractBulkTensor end

struct HilbertSchmidt{PType<:AbsTen{2,4}} <: AbstractBulkTensor
    data::PType
end

struct Trace{PType<:AbsTen{2,4}} <: AbstractBulkTensor
    data::PType
end

const ContractableTensor = Union{AbstractBulkTensor,AbsTen{0,4}}
const ContractableTensors{L} = OnLattice{L,<:ContractableTensor}

gettensor(bulk_tensor::AbstractBulkTensor) = bulk_tensor.data

function Base.convert(
    ::Type{T}, bulk_tensor::AbstractBulkTensor
) where {T<:AbstractTensorMap}
    return convert(T, gettensor(bulk_tensor))
end
# TODO: potentially combine these, but would require type param in AsbtractBulkTensor
function Base.convert(::Type{<:HilbertSchmidt{<:TType}}, tensor::AbsTen{2,4}) where {TType}
    return HilbertSchmidt(convert(TType, tensor))
end
function Base.convert(::Type{<:Trace{<:TType}}, tensor::AbsTen{2,4}) where {TType}
    return Trace(convert(TType, tensor))
end

function Base.similar(bulk_tensor::T) where {T<:AbstractBulkTensor}
    return convert(T, similar(gettensor(bulk_tensor)))
end

MacroTools.@forward AbstractBulkTensor.data TensorKit.codomain,
TensorKit.domain,
TensorKit.space

contractphysical_maybe(bulk_tensor::Trace) = contractphysical(bulk_tensor)
contractphysical_maybe(bulk_tensor::AbsTen{0,4}) = bulk_tensor
contractphysical_maybe(bulk_tensor::AbsTen{2,4}) = bulk_tensor
contractphysical_maybe(bulk_tensor::HilbertSchmidt) = convert(TensorMap, bulk_tensor)

function contractphysical(traceobj::Trace)
    tsrc = traceobj.data
    virtual_space = domain(tsrc)
    tdst = similar(tsrc, one(virtual_space), virtual_space)
    return contractphysical!(tdst, tsrc)
end
function contractphysical(tsrc_1::AbsTen{2,4}, tsrc_2::AbsTen{4,2})
    temp_space = *(map((x, y) -> x * y', domain(tsrc_1), codomain(tsrc_2))...)
    out = similar(tsrc_1, one(temp_space), temp_space)
    @tensoropt out[D1 D5 D2 D6 D3 D7 D4 D8] =
        tsrc_1[k b; D1 D2 D3 D4] * tsrc_2[D5 D6 D7 D8; k b]
    return out
end
function contractphysical!(tdst::AbsTen{0,4}, tsrc::AbsTen{2,4})
    @tensoropt tdst[1 2 3 4] = traceobj[x x; 1 2 3 4]
    return tdst
end

function contractphysical_mergevirtual(traceobj::HilbertSchmidt)
    t1 = traceobj.data
    t2 = t1'
    return contractphysical_mergevirtual(t1, t2)
end

function contractphysical_mergevirtual(tsrc_1::AbsTen{2,4}, tsrc_2::AbsTen{4,2})
    contracted_tensor = contractphysical(tsrc_1, tsrc_2)
    return mergevirtual(contracted_tensor)
end

function mergevirtual(t::AbstractTensorMap{S,0,8}) where {S}
    dom = (fuse(domain(t)[i] * domain(t)[i + 1]) for i in 1:2:7)
    return TensorMap(t.data, one(S), *(dom...))
end

bondspace(tensor::HilbertSchmidt, i::Int) = domain(tensor)[i] * domain(tensor)[i]'
bondspace(tensor::Union{Trace,AbsTen{0,4}}, i::Int) = domain(tensor)[i]
bondspaces(tensor::ContractableTensor) = (bondspace(tensor, i) for i in 1:4)

function bondspaces_onlattice(onlattice::ContractableTensors)
    return (bondspace.(onlattice, i) for i in 1:4)
end

swapaxes(tensor::AbsTen{0,4}) = permute(tensor, (), (2, 1, 4, 3))
swapaxes(tensor::AbsTen{2,4}) = permute(tensor, (1, 2), (4, 3, 6, 5))
swapaxes(tensor::T) where {T<:AbstractBulkTensor} = convert(T, swapaxes(gettensor(tensor)))

invertaxes(tensor::AbsTen{0,4}) = permute(tensor, (), (3, 4, 1, 2))
invertaxes(tensor::AbsTen{2,4}) = permute(tensor, (1, 2), (5, 6, 3, 4))
