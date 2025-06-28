"""
$(TYPEDEF)

Concrete type representing a *normalized* density matrix with associated trace environment 
of type `T`.

See also: [`trace`](@ref), and [`partialtrace`](@ref).
"""
struct DensityMatrix{P<:AbstractUnitCell,T<:Renormalization}
    data::P
    environment::T
    function DensityMatrix(
        data::P, env::T
    ) where {P<:AbstractUnitCell,T<:Renormalization}
        # data = deepcopy(dealias(data))
        data = deepcopy(data)
        # tracenormalize!(data, env)

        trval = trace(data, env)
        @debug "" trace = trval

        if all(trval .≈ zero(scalartype(trval)))
            @warn "state cannot be normalized" trace = trval
        else
            @. rmul!(data, 1 / trval)
        end

        return new{P,T}(data, env)
    end
end
TensorKit.spacetype(::Type{<:DensityMatrix{UC}}) where {UC} = spacetype(eltype(UC))
TensorKit.scalartype(::Type{<:DensityMatrix{UC}}) where {UC} = scalartype(eltype(UC))

"""
$(TYPEDEF)

Concrete type representing a *normalized* purified density matrix with associated trace 
environment of type `T`, i.e. a density matrix of the form ``\\rho = X^{\\dagger}X`` where
``X`` is represented by tensor network.

See also: [`DensityMatrix`](@ref), [`trace`](@ref), and [`partialtrace`](@ref).
"""
struct PurifiedDensityMatrix{P<:HilbertSchmidt,T<:Renormalization}
    data::P
    environment::T
    function PurifiedDensityMatrix(
        data::P, env::T
    ) where {P<:HilbertSchmidt,T<:Renormalization}
        data = deepcopy(data)
        # data = deepcopy(delalias(data))
        # tracenormalize!(data, env)

        trval = trace(data, env)
        @debug "" trace = trval

        if all(trval .≈ zero(scalartype(trval)))
            @warn "state cannot be normalized" trace = trval
        else
            data .= broadcast(data, trval) do ct, norm
                ten, _ = ct
                tenA = ten / sqrt(norm)
                tenB = ten / sqrt(norm)'
                return CompositeTensor(tenA, tenB')
            end
        end

        return new{P,T}(data, env)
    end
end

function Base.show(io::IO, rho::DensityMatrix)
    if get(io, :compact, false)
        summary(rho)
        return nothing
    end
    println(io, "DensityMatrix of unit cell size $(size(rho.data)) normalized using:")
    show(io, rho.environment.alg)
    return nothing
end
function Base.summary(rho::DensityMatrix)
    print("$(size(rho.data))-DensityMatrix")
    return nothing
end

tracenormalize(object, normalizer) = tracenormalize!(deepcopy(object), normalizer)

function tracenormalize!(tensors::AbstractUnitCell, boundary)
    norms = trace(tensors, boundary)
    println(norms)
    @debug "" norms
    broadcast(tensors, norms) do t, n
        rmul!(t, 1 / n)
    end
    return tensors
end

TRGroups.getdata(rho::DensityMatrix) = rho.data

function DensityMatrix(
    pepo::Union{AbstractPEPO,Simulation,DensityMatrix},
    alg::AbstractBoundaryAlgorithm;
    kwargs...,
)
    data = TRGroups.getdata(pepo)
    return DensityMatrix(data, alg; kwargs...)
end

function DensityMatrix(
    pepo::Union{AbstractPEPO,Simulation,DensityMatrix},
    env::Renormalization,
    kwargs...,
)
    data = TRGroups.getdata(pepo)
    TRGroups.renormalize!(env, contractphysical(data))
    return DensityMatrix(data, env)
end

function DensityMatrix(tensors::AbstractUnitCell, alg::AbstractBoundaryAlgorithm; kwargs...)
    env = Renormalization(contractphysical(tensors), alg; kwargs...)
    TRGroups.renormalize!(env)
    return DensityMatrix(tensors, env)
end

function PurifiedDensityMatrix(
    pepo::Union{AbstractPEPO,Simulation,DensityMatrix},
    alg::AbstractBoundaryAlgorithm;
    kwargs...,
)
    data = TRGroups.getdata(pepo)
    return PurifiedDensityMatrix(data, alg; kwargs...)
end

function PurifiedDensityMatrix(
    tensors::AbstractUnitCell, alg::AbstractBoundaryAlgorithm; kwargs...
)
    data = hilbertschmidt(tensors)

    env = Renormalization(data, alg; kwargs...)
    TRGroups.renormalize!(env)

    return PurifiedDensityMatrix(data, env)
end

"""
$(TYPEDSIGNATURES)

Renormalize the environment associated to `rho`, and use the result to normalize the density
matric to unit trace.
"""
function TRGroups.renormalize!(rho::DensityMatrix)
    tensors = rho.data
    env = rho.environment
    TRGroups.renormalize!(rho.environment, contractphysical(tensors))
    tracenormalize!(tensors, env)
    return rho
end

"""
    expval(rho::TensorMap, ops::Vararg{T, N}; inds=NTuple{N}(1:N))

Calculate the expectation value of product of local operators `ops` applied to the indices
`inds` respectively. The keyword `inds` defaults to applying all `ops` to the first indices
of `rho` consecutively.
"""
function expval(
    rho::TensorMap{<:Number,<:IndexSpace,N,N},
    ops::Vararg{<:Matrix,M};
    inds=NTuple{N,Int}(1:N),
) where {N,M}
    pdim = dims(codomain(rho))

    # Sanity check things
    if !allunique(inds)
        throw(ArgumentError("all indices must be unqiue"))
    elseif maximum(inds) > N
        throw(ArgumentError("index out of range"))
    end

    ops = eltype(ops)[ops...]

    j = 1
    # Loop through every index,
    matrices = map(1:N) do i

        # Is this an index we are applying an operator to?
        if !(i ∈ inds) || isempty(ops)
            # If not, then it will be a 1-local identity
            p = pdim[i]
            return Matrix{T}(I, p, p)
        end

        # If so, get the next matrix in the queue

        op = popfirst!(ops)

        d1, d2 = size(op)

        # Check if it is square
        if d1 != d2
            throw(ArgumentError("matrix not square"))
        else
            d = d1 = d2
        end

        # The local space dimension for this index
        pi = pdim[i]

        count = 1
        # If this is true, the op must straddle more than one lattice site
        while pi < d
            # Check if next index is specific in inds
            if count ∈ inds
                pi *= pdim[i + count]
            else
                throw(
                    ArgumentError(
                        "cannot apply $d-dimensional matrix between sites $i and $count as sites are not adjacent",
                    ),
                )
            end
            count += 1
        end

        @assert d == pi

        # Iterate the index queue
        i += count

        return op
    end

    observable = TensorMap(reduce(⊗, matrices), space(rho))

    return tr(observable * rho)
end
# function expval(rho::TensorMap, a::Vararg{Matrix{ComplexF64},N}) where {N}
#     observable = mapreduce(⊗, enumerate(a)) do (i, op)
#         return TensorMap(op, domain(rho)[i], codomain(rho)[i])
#     end
#     return tr(observable * rho)
# end

"""
    partialtranspose(rho::TensorMap, i) -> typeof(rho)

Returns the tensor obtained by transposing the `i`th index of `rho`.
"""
function partialtranspose(t::AbsTen{N,N}, i) where {N}
    codp = NTuple{N,Int}(j == i ? N + i : j for j in 1:N)
    domp = NTuple{N,Int}(j == i ? i : N + j for j in 1:N)
    tp = permute(t, (codp, domp))
    return tp
end

"""
    negativity(rho) -> Number

Calculate the entanglement negativity of bipartite `rho` represented as a `TensorMap` with
`codomain(rho) == domain(rho)`.
"""
function negativity(rho)
    pt = partialtranspose(rho, 1)

    tracenorm = tr(sqrt(pt' * pt))

    return 1 / 2 * (tracenorm - 1)
end

"""
    correlator(rho::DensityMatrix, A1, i1, A2, i2)

Calculate the two-point correlator of `rho` between observables `A1` and `A2` on sites `i1`
end `i2` respectively.
"""
function correlator(rho::DensityMatrix, A1, i1, A2, i2)
    dm = partialtrace(rho, i1, i2)
    return expval(dm, A1, A2)
end

"""
$(TYPEDSIGNATURES)

Calculate the full trace density of the density matrix `rho` for each site in the unit cell.
Note, if `rho` is constructed properly, then `trace(rho) .≈ one(scalartype(rho))`.
"""
trace(rho::DensityMatrix) = trace(rho.data, rho.environment)
function trace(network::AbstractUnitCell, contractor)
    rv = similar(network, scalartype(network))
    vals = map(i -> trace(network, contractor, i), CartesianIndices(network))
    copyto!(rv, vals)
    return rv
end
function trace(network::AbstractUnitCell, contractor, ind)
    rv = zero(scalartype(network))

    x = ind[1]
    y = ind[2]

    boundary = get_contraction_boundary(contractor, x:x, y:y)

    trval = _partialtrace(rv, boundary..., network[x:x, y:y], Val((false)))::typeof(rv)

    return trval
end

function trace(network::HilbertSchmidt, contractor, ind)
    x = ind[1]
    y = ind[2]

    FL, FR, ACU, _, ACD, _ = get_contraction_boundary(contractor, x:x, y:y)

    M = network[x, y]

    return tracecontract(FL, FR, ACU, ACD, M)
end

"""
$(TYPEDSIGNATURES)

Calculate the partial trace of `rho` returning a `TensorMap`. Indices in
`inds` define those sites that remain *untraced*. Site indices in `inds` are mapped
to tensor indices in the same order as in `inds`.
"""
function partialtrace(
    rho::DensityMatrix, inds::Vararg{T,N}
) where {T<:Union{Dims{2},CartesianIndex{2}},N}
    return partialtrace(rho.data, rho.environment, inds...)
end

function partialtrace(rho::PurifiedDensityMatrix, ind::Union{Dims{2},CartesianIndex{2}})
    network = rho.data
    contractor = rho.environment

    x = ind[1]
    y = ind[2]

    FL, FR, ACU, _, ACD, _ = get_contraction_boundary(contractor, x:x, y:y)

    M = network[x, y]

    return onelocalcontract(FL, FR, ACU, ACD, M)
end

function partialtrace(network::AbstractUnitCell, trace, inds::Vararg{T,N}) where {T,N}
    RT = tensormaptype(spacetype(eltype(network)), N, N, scalartype(eltype(network)))

    if length(inds) == 0
        throw(ArgumentError("you must provide at least one site to remain untraced"))
    end

    cinds = map(CartesianIndex, inds)

    p = TupleTools.sortperm(cinds)

    cinds_sorted = TupleTools.permute(cinds, p)

    sortedrho = _partialtrace(network, trace, cinds_sorted...)

    rho = permute(sortedrho, (TupleTools.invperm(p), N .+ TupleTools.invperm(p)))

    return rho::RT
end

function _partialtrace(
    network::AbstractUnitCell, contractor, inds::Vararg{CartesianIndex{2},N}
) where {N}
    lx = minimum(x -> x[1], inds)
    ly = minimum(x -> x[2], inds)

    rx = maximum(x -> x[1], inds)
    ry = maximum(x -> x[2], inds)

    boundary = get_contraction_boundary(contractor, lx:rx, ly:ry)

    issite = Val(
        Tuple(CartesianIndex(ind) ∈ inds for ind in Iterators.product(lx:rx, ly:ry))
    )

    # Get the untraced site tensors
    sites = map(i -> network[i[1], i[2]], inds)

    cod = mapreduce(x -> getindex(codomain(x), 1), *, sites)

    dom = mapreduce(x -> getindex(codomain(x), 2)', *, sites)

    rv = TensorMap{scalartype(network)}(undef, cod, dom)

    MS = [network[i, j] for i in lx:rx, j in ly:ry]

    trval = _partialtrace(rv, boundary..., MS, issite)::typeof(rv)

    return trval
end

function get_contraction_boundary(con::Renormalization, xr::UnitRange, yr::UnitRange)
    return get_contraction_boundary(con.runtime, xr, yr)
end
function get_contraction_boundary(vumps::VUMPSRuntime, xr::UnitRange, yr::UnitRange)
    if length(yr) != 1
        throw(ArgumentError("cannot contract across multiple rows using VUMPS"))
    end

    i2 = first(yr)

    l = first(xr)
    r = last(xr)

    _, _, AR, AC = TRGroups.unpack(vumps.mps)

    fixed_points = vumps.fixedpoints

    FL = fixed_points.left[l, i2]
    FR = fixed_points.right[r, i2]

    ACU = AC[l, i2]
    ARU = Tuple(AR[i, i2] for i in (l + 1):r)

    ACD = AC[l, i2 + 1]'
    ARD = Tuple(AR[i, i2 + 1]' for i in (l + 1):r)

    return FL, FR, ACU, ARU, ACD, ARD
end

function get_contraction_boundary(
    ctmrgrt::CornerMethodRuntime, xr::UnitRange, yr::UnitRange
)
    ctmrg = ctmrgrt.primary

    C1, C2, C3, C4 = getboundary(ctmrg.corners, xr, yr)
    T1, T2, T3, T4 = map(ts -> Tuple(t for t in ts), getboundary(ctmrg.edges, xr, yr))

    return C1, C2, C3, C4, T1, T2, T3, T4
end

# function _partialtrace(
#     network::AbstractLattice, vumps::VUMPSRuntime, inds::Vararg{CartesianIndices{2},N}
# ) where {N}
#     i2 = first(inds)[1]
#     for i in inds
#         if i[2] != i2
#             throw(ArgumentError("cannot contract across multiple rows using VUMPS"))
#         end
#     end
#
#     l = minimum(inds)
#     r = maximum(inds)
#
#     issite = Val(Tuple(i ∈ inds for i in l:r))
#
#     _, _, AR, AC = ITC.unpack(vumps.mps)
#
#     fixed_points = vumps.fixedpoints
#
#     FL = fixed_points.left[l, i2]
#     FR = fixed_points.right[r, i2]
#
#     ACU = AC[l, i2]
#     ACD = AC[l, i2 + 1]'
#
#     ARU = Tuple(AR[i, i2] for i in (l + 1):r)
#     ARD = Tuple(AR[i, i2 + 1]' for i in (l + 1):r)
#
#     MS = [network[i, i2] for i in l:r]
#
#     sites = map(i -> network[i, i2], inds)
#
#     cod = mapreduce(x -> getindex(codomain(x), 1), *, sites)
#     dom = mapreduce(x -> getindex(codomain(x), 2)', *, sites)
#
#     rv = TensorMap(undef, scalartype(vumps), cod, dom)
#
#     return _partialtrace(rv, FL, FR, ACU, ARU, ACD, ARD, MS, issite)::typeof(rv)
# end

@generated function _partialtrace(
    rv::OutType,
    FL,
    FR,
    ACU::T1,
    ARU::NTuple{N,T1},
    ACD::T2,
    ARD::NTuple{N,T2},
    MS::AbstractArray{<:AbsTen{2}},
    sites::Val{S},
) where {OutType,N,T1,T2,S}
    symb = (s, i) -> Symbol("$(s)_$i")
    ist = i -> S[i] ? symb(:b, i) : symb(:k, i)

    e_ARU = [
        Expr(:ref, Expr(:ref, :ARU, i), symb(:n, i + 1), symb(:u, i + 1), symb(:u, i)) for
        i in 1:N
    ]
    e_ARD = [
        Expr(:ref, Expr(:ref, :ARD, i), symb(:d, i + 1), symb(:d, i), symb(:s, i + 1)) for
        i in 1:N
    ]

    e_MS = [
        Expr(
            :ref,
            Expr(:ref, :MS, i),
            symb(:k, i),
            ist(i),
            symb(:h, i),
            symb(:s, i),
            symb(:h, i - 1),
            symb(:n, i),
        ) for i in 1:(N + 1)
    ]

    e_ACU = Expr(:ref, :ACU, symb(:n, 1), symb(:u, 1), symb(:u, 0))
    e_ACD = Expr(:ref, :ACD, symb(:d, 1), symb(:d, 0), symb(:s, 1))
    e_FL = Expr(:ref, :FL, symb(:h, 0), symb(:u, 0), symb(:d, 0))
    e_FR = Expr(:ref, :FR, symb(:h, N + 1), symb(:u, N + 1), symb(:d, N + 1))

    e_einsum = Expr(:call, :*, e_FL, e_FR, e_ACU, e_ACD, e_MS..., e_ARU..., e_ARD...)

    e_cod = Symbol[]
    e_dom = Symbol[]

    for i in 1:(N + 1)
        if S[i]
            push!(e_cod, symb(:k, i))
            push!(e_dom, symb(:b, i))
        end
    end

    if OutType <: Number
        e_out = :rv
    else
        e_out = Expr(:typed_vcat, :rv, Expr(:row, e_cod...), Expr(:row, e_dom...))
    end

    quote
        @tensoropt $e_out = $e_einsum
        return rv
    end
end

function get_bond_symbol(i, j, dir)
    if dir == :h
        str = "$(i)$(j)_$(i + 1)$(j)"
    elseif dir == :v
        str = "$(i)$(j)_$(i)$(j + 1)"
    end
    return Symbol(str)
end

# function trace(ctmrg::CornerMethodRuntime, network, i1::UnitRange, i2::UnitRange)
#     return contract(ctmrg.primary, network, i1, i2)
# end
# function _partialtrace(
#     network::AbstractLattice,
#     ctmrgrt::CornerMethodRuntime,
#     inds::Vararg{CartesianIndices{2},N},
# ) where {N}
#     ctmrg = ctmrgrt.primary
#
#     lx = minimum(x -> x[1], inds)
#     ly = minimum(x -> x[2], inds)
#
#     rx = maximum(x -> x[1], inds)
#     ry = maximum(x -> x[2], inds)
#
#     issite = Val(Tuple(ind ∈ inds for ind in Iterators.product(lx:rx, ly:ry)))
#
#     cs = getboundary(ctmrg.corners, lx:rx, ly:ry)
#     es = map(ts -> Tuple(t for t in ts), getboundary(ctmrg.edges, lx:rx, ly:ry))
#
#     sites = map(i -> network[i...], inds)
#
#     cod = mapreduce(x -> getindex(codomain(x), 1), *, sites)
#     dom = mapreduce(x -> getindex(codomain(x), 2)', *, sites)
#
#     rv = TensorMap(undef, scalartype(network), cod, dom)
#
#     return _partialtrace(rv, cs..., es..., network, issite)
# end

@generated function _partialtrace(
    rv::OutType,
    C1,
    C2,
    C3,
    C4,
    T1::NTuple{Nx,E},
    T2::NTuple{Ny,E},
    T3::NTuple{Nx,E},
    T4::NTuple{Ny,E},
    MS::AbstractArray{<:AbsTen{2}},
    sites::Val{S},
) where {OutType,Nx,Ny,E,S}
    gh = (i, j) -> get_bond_symbol(i, j, :h)
    gv = (i, j) -> get_bond_symbol(i, j, :v)
    symb = (s, i) -> Symbol("$(s)_$(i)")

    ist = ind -> S[ind] ? symb(:b, ind) : symb(:k, ind)

    e_T1s = [
        Expr(:ref, Expr(:ref, :T1, i), gv(i + 1, 1), (gh(i + 1, 1), gh(i, 1))...) for
        i in 1:Nx
    ]

    e_T2s = [
        Expr(
            :ref,
            Expr(:ref, :T2, j),
            gh(Nx + 1, j + 1),
            (gv(Nx + 2, j), gv(Nx + 2, j + 1))...,
        ) for j in 1:Ny
    ]
    e_T3s = [
        Expr(
            :ref,
            Expr(:ref, :T3, i),
            gv(i + 1, Ny + 1),
            (gh(i, Ny + 2), gh(i + 1, Ny + 2))...,
        ) for i in 1:Nx
    ]
    e_T4s = [
        Expr(:ref, Expr(:ref, :T4, j), gh(1, j + 1), (gv(1, j + 1), gv(1, j))...) for
        j in 1:Ny
    ]

    e_MS = [
        Expr(
            :ref,
            Expr(:ref, :MS, i, j),
            symb(:k, ind),
            ist(ind),
            gh(i + 1, j + 1),
            gv(i + 1, j + 1),
            gh(i, j + 1),
            gv(i + 1, j),
        ) for (ind, (i, j)) in enumerate(Iterators.product(1:Nx, 1:Ny))
    ]

    e_C1 = Expr(:ref, :C1, gh(1, 1), gv(1, 1))
    e_C2 = Expr(:ref, :C2, gh(Nx + 1, 1), gv(Nx + 2, 1))
    e_C3 = Expr(:ref, :C3, gh(Nx + 1, Ny + 2), gv(Nx + 2, Ny + 1))
    e_C4 = Expr(:ref, :C4, gh(1, Ny + 2), gv(1, Ny + 1))

    e_cod = Symbol[]
    e_dom = Symbol[]

    for i in 1:(Nx * Ny)
        if S[i]
            push!(e_cod, symb(:k, i))
            push!(e_dom, symb(:b, i))
        end
    end

    if OutType <: Number
        e_out = :rv
    else
        e_out = Expr(:typed_vcat, :rv, Expr(:row, e_cod...), Expr(:row, e_dom...))
    end

    e_einsum = Expr(
        :call, :*, e_C1, e_C2, e_C3, e_C4, e_T1s..., e_T2s..., e_T3s..., e_T4s..., e_MS...
    )

    quote
        @tensoropt $e_out = $e_einsum
        return rv
    end
end

# tempoary
function contract(tensors, network)
    inds = collect(CartesianIndices(network))

    RT = promote_type(scalartype(tensors), scalartype(network))

    rv = similar(network, RT)

    rv .= contract.(Ref(tensors), Ref(network), inds)
    return rv
end

contract(tensors, network, i1::Int, i2::Int) = contract(tensors, network, i1:i1, i2:i2)
contract(tensors, network, inds) = contract(tensors, network, inds[1], inds[2])

function contract(vumps::VUMPSRuntime, network, i1::UnitRange, i2::UnitRange)
    if length(i2) > 1
        throw(
            ArgumentError(
                "Cannot contract a VUMPS boundary around more than one vertical tensor."
            ),
        )
    else
        i2 = i2[begin]
    end

    _, _, AR, AC = TRGroups.unpack(vumps.mps)

    fixed_points = vumps.fixedpoints

    l = i1[begin]
    r = i1[end]

    FL = fixed_points.left[l, i2]
    FR = fixed_points.right[r, i2]

    ACU = AC[l, i2]
    ACD = AC[l, i2 + 1]'

    ARU = tuple((AR[i, i2] for i in (l + 1):r)...)
    ARD = tuple((AR[i, i2 + 1]' for i in (l + 1):r)...)

    return _contractall(FL, FR, ACU, ARU, ACD, ARD, network[i1, i2])
end

@generated function _contractall(
    FL,
    FR,
    ACU::T1,
    ARU::NTuple{N,T1},
    ACD::T2,
    ARD::NTuple{N,T2},
    MS::AbstractArray{<:AbsTen{0}},
) where {N,T1,T2}
    symb = (s, i) -> Symbol("$(s)_$i")

    e_ARU = [
        Expr(:ref, Expr(:ref, :ARU, i), symb(:n, i + 1), symb(:u, i + 1), symb(:u, i)) for
        i in 1:N
    ]
    e_ARD = [
        Expr(:ref, Expr(:ref, :ARD, i), symb(:d, i + 1), symb(:d, i), symb(:s, i + 1)) for
        i in 1:N
    ]

    e_MS = [
        Expr(
            :ref, Expr(:ref, :MS, i), symb(:h, i), symb(:s, i), symb(:h, i - 1), symb(:n, i)
        ) for i in 1:(N + 1)
    ]

    e_ACU = Expr(:ref, :ACU, symb(:n, 1), symb(:u, 1), symb(:u, 0))
    e_ACD = Expr(:ref, :ACD, symb(:d, 1), symb(:d, 0), symb(:s, 1))
    e_FL = Expr(:ref, :FL, symb(:h, 0), symb(:u, 0), symb(:d, 0))
    e_FR = Expr(:ref, :FR, symb(:h, N + 1), symb(:u, N + 1), symb(:d, N + 1))

    e_einsum = Expr(:call, :*, e_FL, e_FR, e_ACU, e_ACD, e_MS..., e_ARU..., e_ARD...)

    quote
        # @tensoropt rv = scalar($e_einsum)
        @tensoropt rv = $e_einsum
    end
end

function lefteigen(TMS, A, M, numeigs=1, which=:LM; kwargs...)
    fl = TRGroups._initfixedpoints(rand, A[1, 1], M[1, 1], :left)
    return eigsolve(fl, numeigs, which; kwargs...) do x0
        fl = x0
        for tm in TMS
            fl = fl * tm
        end
        return fl
    end
end

function testextrap(; Jy=1.2, D=4)
    s = ComplexSpace(2)
    S = ComplexSpace(D)

    rho = fill((1, 0, 0), 2, 2)

    xi0 = 100.0
    model = PEPOKit.Heisenberg(LADDER[1]; Jx=0.5, Jy=Jy, Jz=1.0)
    method = PEPOKit.TEBD(;
        breaktol=1e-5,
        trotterorder=1,
        numsteps=10000,
        timestep=0.01,
        truncalg=PEPOKit.SimpleUpdate(; zeta=zeta0),
    )
    sim = Simulation(
        [PEPOKit.OnAxis(model, 1), PEPOKit.OnAxis(model, 2)],
        rho,
        S;
        physical=s,
        method=method,
    )

    alg = VUMPS(; bonddim=16)

    xis = []
    xvals = []
    clens = []

    for i in 1:25
        xi0 += 20.0

        method = PEPOKit.TEBD(;
            breaktol=1e-5,
            trotterorder=1,
            numsteps=10000,
            timestep=0.01,
            truncalg=PEPOKit.SimpleUpdate(; zeta=zeta0),
        )
        sim = Simulation(sim.model, PEPOKit.quantumstate(sim), method, SimulationInfo())

        simulate!(sim)

        if i < 6
            continue
        end

        pepo = PEPOKit.quantumstate(sim)

        dm = DensityMatrix(pepo, VUMPS(; bonddim=32))

        rhos = map(ind -> ReducedDensityMatrix(dm, ind), uniqueindices(pepo))
        expt = map(x -> expval(x, PAULI[1]), rhos)

        clen = PEPOKit.correlationlength(dm)

        push!(xis, xi0)
        push!(xvals, real.(expt)[1, 1])
        push!(clens, clen)
    end

    return xis, xvals, clens
end

function testCFconv(; D=4)
    s = ComplexSpace(2)
    S = ComplexSpace(D)

    rho = fill((1, 0, 0), 2, 2)

    xvals = []
    clens = []

    method = PEPOKit.TEBD(;
        breaktol=1e-5,
        trotterorder=1,
        numsteps=10000,
        timestep=0.01,
        truncalg=PEPOKit.SimpleUpdate(; zeta=Inf),
    )

    for Jy in (Js = 1.45:0.05:1.55)
        model = PEPOKit.Heisenberg(LADDER[1]; Jx=0.5, Jy=Jy, Jz=1.0)

        sim = Simulation(
            [PEPOKit.OnAxis(model, 1), PEPOKit.OnAxis(model, 2)],
            rho,
            S;
            physical=s,
            method=method,
        )

        simulate!(sim)

        pepo = PEPOKit.quantumstate(sim)

        dm = DensityMatrix(pepo, VUMPS(; bonddim=32))

        rhos = map(ind -> ReducedDensityMatrix(dm, ind), uniqueindices(pepo))
        expt = map(x -> expval(x, PAULI[1]), rhos)

        clen = PEPOKit.correlationlength(dm)

        push!(xvals, real.(expt)[1, 1])
        push!(clens, clen)
    end

    return Js, xvals, clens
end

function TRGroups.TransferMatrix(dm::DensityMatrix, yind=1)
    AL, _, _, _ = TRGroups.unpack(dm.environment.runtime.mps)
    bulk = contractphysical.(dm.data)
    tms = TRGroups.TransferMatrix.(AL[:, yind], bulk[:, yind], AL[:, yind + 1])
    return tms
end
function TRGroups.TransferMatrix(dm::PurifiedDensityMatrix, yind=1)
    AL, _, _, _ = TRGroups.unpack(dm.environment.runtime.mps)
    tms = TRGroups.TransferMatrix.(AL[:, yind], dm.data[:, yind], AL[:, yind + 1])
    return tms
end

function transferspectrum(dm::PurifiedDensityMatrix, n=3)
    return transferspectrum(dm.data, dm.environment.runtime, n)
end
function transferspectrum(dm::DensityMatrix, n=3)
    return transferspectrum(dm.data, dm.environment.runtime, n)
end
function transferspectrum(bulk, rt::VUMPSRuntime, n::Int=3)
    mps = rt.mps

    AL, _, _, _ = TRGroups.unpack(mps)

    if eltype(bulk) <: TensorMap
        bulk = contractphysical.(bulk)
    end

    tms = TRGroups.TransferMatrix.(AL[:, 1], bulk[:, 1], AL[:, 2])

    fl = TRGroups.initfixedpoint(rand, AL[1, 1], bulk[1, 1], :left)

    xis, _, _ = eigsolve(fl, n, :LM) do x0
        fl = x0
        for tm in tms
            fl = fl * tm
        end
        return fl
    end

    @. xis = xis / xis[1]

    return xis[1:n]
end
function transferspectrum(bulk, ctmrg::CornerMethodRuntime, n::Int)
    return transferspectrum(bulk, edges(ctmrg), n::Int)
end
function transferspectrum(bulk, edges::Edges, n::Int=3)
    T1, _, T3, _ = edges

    y = 1

    ES = map(axes(bulk, 1)) do x
        t1 = T1[x, y - 1]
        t3 = T3[x, y + 1]

        M = contractphysical(bulk[x, y])

        @tensor E[lt w lb; rt e rb] := t1[n; rt lt] * M[e s w n] * t3[s; lb rb]

        d1 = dim(codomain(E))
        d2 = dim(domain(E))

        return reshape(convert(Array, E), d1, d2)
    end

    E0 = pop!(ES)

    while !isempty(ES)
        E = pop!(ES)
        E0 = E * E0
    end

    vals, _, _ = eigsolve(E0, E0[1, :], n, :LM)

    return vals / vals[1]
end

function _correlationlength(dm::Union{DensityMatrix,PurifiedDensityMatrix})
    xis = transferspectrum(dm, 4)

    eps1 = -log(abs(xis[2]))
    ind = isapprox(abs(xis[2]), abs(xis[3])) ? 4 : 3
    eps2 = -log(abs(xis[ind]))

    return eps1, eps2
end

"""
    correlationlength(state::DensityMatrix) -> Float64
    correlationlength(state::PEPO, env::AbstractBoundary) -> Float64
    correlationlength(state::PEPO, alg::AbstractBoundaryAlgorithm) -> Float64

Calculate the correlation length zeta of the quantum state `state` by constructing the transfer
matrix and computing the the inverse of the sub-leading eigenvalue.

The correlation length characterizes how quickly correlations decay with distance in a
quantum system. For a state with exponentially decaying correlations of the form
C(r) ∝ exp(-r/ξ), the correlation length ξ quantifies the characteristic distance over
which quantum correlations persist.
"""
function correlationlength(dm::Union{DensityMatrix,PurifiedDensityMatrix})
    unit_cell_length = size(dm.data, 1)
    eps1, _ = _correlationlength(dm)
    return unit_cell_length / eps1
end

function correlationlength(f, qs, chis=8:1:16)
    eps1_vec = Float64[]
    delt_vec = Float64[]

    for chi in chis
        dm = DensityMatrix(qs, f(chi))
        eps1, eps2 = _correlationlength(dm)

        push!(eps1_vec, eps1)
        push!(delt_vec, eps2 - eps1)
    end

    return delt_vec, eps1_vec
end

function _testising(; zeta=Inf, D=2)
    rho = fill(ThermalState(), 2, 2)

    # βc = log(1 + sqrt(2)) / 2
    βc = 1 / 1.2737

    H = Heisenberg(; Jz=-1.0, hx=2.5)
    # H = Heisenberg(; Jz=-1.0)
    # Until b = 0.8

    alg = NTU(; optalg=EVB(; gauge=WTG()), zeta=zeta, rankreduce=false)

    method_coarse = TEBD(; timestep=0.01 * βc, truncalg=alg)
    method_fine = TEBD(; timestep=0.01 * βc, truncalg=alg)
    # method_coarse = TEBD(; timestep=0.01 * βc, truncalg=PEPOKit.SimpleUpdate(; zeta=zeta))
    # method_coarse = TEBD(; timestep=0.0001 * βc, truncalg=PEPOKit.SimpleUpdate(; zeta=zeta))

    model = [OnAxis(H, 1), OnAxis(H, 2)]

    alg = VUMPS(; bonddim=16, maxiter=200, tol=1e-10)

    s = ComplexSpace(2)
    S = ComplexSpace(D)

    output = Float64[]

    sim = Simulation(model, rho, S; physical=s, method=method_coarse)

    #
    t0 = sim.info.iterations * method_coarse.timestep
    t0 = 0

    # Observables
    Zs = Float64[]
    # Ls = Tuple{Vector{Float64},Vector{Float64}}[]
    Ls = Float64[]
    ts = Float64[]
    converged = Bool[]

    numsteps = 120

    simulate!(sim, 110; maxshots=numsteps) do sim
        if sim.info.iterations < 90
            return sim
        end

        dm = DensityMatrix(quantumstate(sim), alg)
        rdm = ReducedDensityMatrix(dm, (1, 1))

        Z = abs(expval(rdm, PAULI[3]))
        L = correlationlength(dm)

        # ds,es = PEPOKit.correlationlength(
        #     D -> VUMPS(bonddim=D, tol=1e-10), 
        #     PEPOKit.quantumstate(sim), 
        #     20:1:20
        # )

        push!(Zs, Z)
        push!(Ls, L)

        push!(converged, dm.environment.info.converged)
        push!(ts, 1 / (t0 + sim.info.iterations * method_coarse.timestep))
    end

    return Zs, Ls, ts, converged
end

function testising(; D=2)

    # xis = [1e4,2e4,4e4,8e4]
    # xis = [exp10(i) for i in 6:0.5:7]
    xis = [exp10(i) for i in 6.5:6.5]
    # xis = [exp10(i) for i in 3:0.5:4]
    push!(xis, Inf)

    Zs = Vector{Float64}[]
    # Ls = Vector{Tuple{Vector{Float64},Vector{Float64}}}[]
    Ls = Vector{Float64}[]
    Ts = Vector{Float64}[]
    converged = Vector{Bool}[]

    # Ds = [5,6,7]

    foreach(xis) do xi
        zs, ls, ts, cs = _testising(; zeta=zeta, D=D)

        push!(Zs, zs)
        push!(Ls, ls)
        push!(Ts, ts)

        push!(converged, cs)
    end

    return xis, Zs, Ls, Ts, converged
end

function test_ising_extrap(; hx=0.0, D=5, numsteps=400)
    X, _, Z = PAULI

    if hx == 0.0
        βc = log(1 + sqrt(2)) / 2
    elseif hx == 2.5
        βc = 1 / 1.2737
    end

    xis = [exp10(i) for i in 5:0.1:7]

    push!(xis, Inf)

    Ls = Float64[]
    Zs = Float64[]

    rho = fill(ThermalState(), 2, 2)

    s = ComplexSpace(2)
    S = ComplexSpace(D)

    H = -1.0 * LocalOp(Z, Z) + hx * LocalOp(X)
    model = Model(H)

    alg = VUMPS(; bonddim=16, maxiter=200, tol=1e-10)

    for xi in xis
        method = TEBD(; timestep=0.0025 * βc, truncalg=PEPOKit.SimpleUpdate(; zeta=zeta))

        sim = Simulation(model, rho, S; physical=s, method=method)

        simulate!(identity, sim, numsteps; maxshots=numsteps)

        dm = DensityMatrix(quantumstate(sim), alg)

        rdm = partialtrace(dm, (1, 1))
        Z = abs(expval(rdm, PAULI[3]))

        L = correlationlength(dm)

        push!(Ls, L)
        push!(Zs, Z)
    end

    return xis, Zs, Ls
end

function alltestbias(; D=5)

    # xis = [1e4,2e4,4e4,8e4]
    # Xis = collect(200:50.0:700)
    # xis = [exp10(i) for i in 3:0.5:4]

    Hs = [0.002, 0.001, 0.0005]
    Zs = Vector{Float64}[]
    Ls = Vector{Float64}[]
    Ts = Vector{Float64}[]
    # Ds = [5,6,7]

    foreach(hz) do hz
        zs, ls, ts = testquantum(; zeta=Inf, D=D, hz=hz)

        push!(Zs, zs)
        push!(Ls, ls)
        push!(Ts, ts)
    end

    return Xis, Zs, Ls, Ts
end

function alltestquantum(; D=5, N=1)
    Xis = [exp10(i) for i in 4:0.2:5]
    Ds = [6, 7, 8]

    Zs = Vector{Float64}[]
    Ls = Vector{Float64}[]
    Ts = Vector{Float64}[]
    # Ds = [5,6,7]

    # foreach(Xis) do xi
    foreach(Ds) do D
        zs, ls, ts = testquantum(; zeta=Inf, D=D, N=N)

        push!(Zs, zs)
        push!(Ls, ls)
        push!(Ts, ts)
    end

    return Xis, Zs, Ls, Ts
end

function testquantum(; zeta=Inf, D, N=1)
    X, _, Z = PAULI

    rho = fill(ThermalState(), 2, 2)

    βc = 1 / 1.2737

    # model = Model(-1.0 * LocalOp(Z,Z) + 2.5 * LocalOp(X))
    model = Model(PEPOKit.Ising(-1.0, Z, [2.5 * LocalOp(X), 0.005 * LocalOp(Z)]))

    # model = Model([0.5 * h, -1.0 * LocalOp(Z,Z), 0.5 * h])

    truncalg = PEPOKit.SimpleUpdate(; zeta=zeta)
    # truncalg = PEPOKit.NTU(;zeta = xi, optalg=EVB())

    # Need 400 iterations total
    # method = TEBD(; truncalg=truncalg)
    method = TEBD(;
        truncalg=PEPOKit.NTU(;
            rankreduce=true, zeta=zeta, optalg=DZM(; isotol=1e-8, gauge=NoGauge())
        ),
    )
    # method = TEBD(; truncalg=PEPOKit.FET( 
    #     alg=VUMPS(bonddim=4, tol=1e-8), rankreduce=false, optalg=PEPOKit.EVB()))
    # alg = CTMRG(; bonddim=25, maxiter=200, tol=1e-8, randinit=false)
    alg = VUMPS(; bonddim=4, maxiter=200, tol=1e-8)

    s = ComplexSpace(2)
    S = ComplexSpace(D)

    output = Float64[]
    timestep = 0.02βc / (8 * N)
    sim = Simulation(model, rho, s, S; method=method, timestep=timestep)
    # sim = Simulation(model, rho, s, S; method=TEBD(), timestep=timestep)

    t0 = 0

    # Observables
    Zs = Float64[]
    Ls = Float64[]
    Ts = Float64[]

    # simulate!(identity, sim, 80)
    simulate!(identity, sim; numsteps=300 * N)

    sim = Simulation(model, quantumstate(sim), method, sim.info; timestep=timestep)

    simulate!(sim; numsteps=200 * N, maxshots=40) do sim
        dm = DensityMatrix(quantumstate(sim), alg)
        # dm = PurifiedDensityMatrix(quantumstate(sim), alg)
        rdm = partialtrace(dm, (1, 1))

        M = abs(expval(rdm, Z))

        push!(Zs, M)

        try
            L = correlationlength(dm)
            push!(Ls, L)
        catch _
            push!(Ls, NaN)
        end

        push!(Ts, 1 / (t0 + sim.info.iterations * timestep))
    end

    return Zs, Ls, Ts
end

function extrapquantum(; D)
    X, _, Z = PAULI

    rho = fill(ThermalState(), 2, 2)

    βc = 1 / 1.2737

    h = 2.5 * LocalOp(X)

    model = Model(-1.0 * LocalOp(Z, Z) + 2.5 * LocalOp(X))
    # model = Model([0.5 * h, -1.0 * LocalOp(Z,Z), 0.5 * h])

    # truncalg = PEPOKit.NTU(zeta = xi)

    # Need 400 iterations total

    # alg = CTMRG(; bonddim=16, maxiter=200, tol=1e-8, randinit=false)
    alg = VUMPS(; bonddim=32, maxiter=200, tol=1e-10)

    s = ComplexSpace(2)
    S = ComplexSpace(D)

    output = Float64[]

    # Observables
    Zs = Float64[]
    Ls = Float64[]

    Xs = [exp10(i) for i in 4:0.1:5]
    for xi in Xs
        truncalg = PEPOKit.SimpleUpdate(; zeta=zeta)
        method = TEBD(; timestep=(0.02 * βc) / 4, truncalg=truncalg)

        sim = Simulation(model, rho, S; physical=s, method=method)
        simulate!(identity, sim, 200)

        dm = DensityMatrix(quantumstate(sim), alg)
        rdm = ReducedDensityMatrix(dm, (1, 1))

        M = abs(expval(rdm, Z))
        L = correlationlength(dm)

        push!(Zs, M)
        push!(Ls, L)
    end

    return Xs, Zs, Ls
end

function alltestaxy()
    zs = []
    ps = []
    ns = []
    js = collect(0.05:0.05:2.0)
    for J in js
        z, p, n = testaxy(J)
        push!(zs, z)
        push!(ps, p)
        push!(ns, n)
    end

    return js, zs, ps, ns
end
function testaxy(J)
    X, Y, Z = PAULI

    D = Dissipator(LADDER[1])

    model = Model(J * LocalOp(X, X) - J * LocalOp(Y, Y) + D)

    s = U1Space(-1 => 1, 1 => 1)
    sp = [s s'; s' s]

    S = U1Space(0 => 8)

    ρ = [(0, 0, -1) (0, 0, 1); (0, 0, 1) (0, 0, -1)]
    sim = Simulation(model, ρ, sp, S; timestep=0.1, method=TEBD())

    simulate!(identity, sim; numsteps=100, maxshots=1)

    dm = DensityMatrix(quantumstate(sim), VUMPS(; bonddim=16))

    rdm1 = partialtrace(dm, (1, 1))
    rdm12 = partialtrace(dm, (1, 1), (2, 1))

    return sim,
    real(expval(rdm1, PAULI[3])), real(tr(rdm12 * rdm12)),
    real(negativity(rdm12))
end
