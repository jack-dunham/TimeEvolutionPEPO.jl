abstract type AbstractBulkTensors end

struct HilbertSchmidt{Lat,PType<:AbsTen{2,4}} <: AbstractBulkTensors
    data::OnLattice{Lat,PType,Matrix{PType}}
end

struct Trace{Lat,PType<:AbsTen{2,4}} <: AbstractBulkTensors
    data::OnLattice{Lat,PType,Matrix{PType}}
end

const ContractableTensor = Union{AbsTen{2,4},AbsTen{0,4}}
const ContractableTensors = OnLattice{<:AbstractLattice,<:ContractableTensor}

const ContractableObjects = Union{
    AbstractBulkTensors,OnLattice{<:AbstractLattice,<:ContractableTensor}
}

gettensors(bulk_tensor::AbstractBulkTensors) = bulk_tensor.data

detrace(trace::Trace) = contractphysical.(gettensors(trace))
detrace(hilbert::HilbertSchmidt) = gettensors(hilbert)
detrace(x) = x

function contractphysical(tsrc::AbsTen{0,4})
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

function contractphysical_mergevirtual(tsrc::AbsTen{2,4})
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

_bondspace(tensor::AbsTen{2,4}, i::Int) = domain(tensor)[i] * domain(tensor)[i]'
_bondspace(tensor::AbsTen{0,4}, i::Int) = domain(tensor)[i]
_bondspaces(tensor::ContractableTensor) = (bondspace(tensor, i) for i in 1:4)

function _bondspaces(onlattice::ContractableTensors)
    return (_bondspace.(onlattice, i) for i in 1:4)
end

swapaxes(tensor::AbsTen{0,4}) = permute(tensor, (), (2, 1, 4, 3))
swapaxes(tensor::AbsTen{2,4}) = permute(tensor, (1, 2), (4, 3, 6, 5))

invertaxes(tensor::AbsTen{0,4}) = permute(tensor, (), (3, 4, 1, 2))
invertaxes(tensor::AbsTen{2,4}) = permute(tensor, (1, 2), (5, 6, 3, 4))
