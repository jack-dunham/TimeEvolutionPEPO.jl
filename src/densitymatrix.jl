"""
    struct DensityMatrix{P<:AbstractLattice, T<:InfiniteContraction}

Concrete type representing a *normalized* density matrix with associated trace environment.
"""
struct DensityMatrix{P<:AbstractLattice,T<:InfiniteContraction}
    data::P
    environment::T
    function DensityMatrix(
        data::P, env::T
    ) where {P<:AbstractLattice,T<:InfiniteContraction}
        data = deepcopy(data)
        # tracenormalize!(data, env)

        trval = trace(data, env)

        if all(trval .≈ zero(scalartype(trval)))
            @warn "state cannot be normalized" trace = trval
        else
            broadcast(data, trval) do t, n
                rmul!(t, 1 / n)
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
    show(io, rho.environment.algorithm)
    return nothing
end
function Base.summary(rho::DensityMatrix)
    print("$(size(rho.data))-DensityMatrix")
    return nothing
end

tracenormalize(object, normalizer) = tracenormalize!(deepcopy(object), normalizer)

function tracenormalize!(tensors::AbstractLattice, boundary)
    norms = trace(tensors, boundary)
    println(norms)
    @debug "" norms
    broadcast(tensors, norms) do t, n
        rmul!(t, 1 / n)
    end
    return tensors
end

tensors(rho::DensityMatrix) = rho.data

function DensityMatrix(
    pepo::Union{AbstractPEPO,Simulation,DensityMatrix},
    alg::AbstractBoundaryAlgorithm;
    kwargs...,
)
    data = tensors(pepo)
    return DensityMatrix(data, alg; kwargs...)
end
function DensityMatrix(
    pepo::Union{AbstractPEPO,Simulation,DensityMatrix}, env::InfiniteContraction, kwargs...
)
    data = tensors(pepo)
    renormalize!(env, contractphysical(data))
    return DensityMatrix(data, env)
end

function DensityMatrix(tensors::AbstractLattice, alg::AbstractBoundaryAlgorithm; kwargs...)
    env = newcontraction(alg, contractphysical(tensors); kwargs...)
    runcontraction!(env)
    return DensityMatrix(tensors, env)
end

function renormalize!(rho::DensityMatrix)
    tensors = rho.data
    env = rho.environment
    renormalize!(rho.environment, contractphysical(tensors))
    tracenormalize!(tensors, env)
    return rho
end

"""
    struct ReduceDensityMatrix{S, N, T<:AbstractTensorMap{S,N,N}} <: AbstractTensorMap{S,N,N}

Concrete type representing a reduced density matrix as a single tensor of type `T`. For 
convenience, much of the `AbstractTensorMap` interface works for this data type, however no 
effort is made to esnure the resulting object representes a physical density matrix.
"""
struct ReducedDensityMatrix{S,N,T<:AbstractTensorMap{S,N,N}} <: AbstractTensorMap{S,N,N}
    data::T
    inds::NTuple{N,CartesianIndex{2}}
    function ReducedDensityMatrix(
        data::T, inds::NTuple{N,Union{CartesianIndex{2},NTuple{2}}}
    ) where {S,N,T<:AbstractTensorMap{S,N,N}}
        inds_cartesian = map(CartesianIndex, inds)
        return new{S,N,T}(data, inds_cartesian)
    end
end

function Base.show(io::IO, rho::ReducedDensityMatrix)
    tupinds = map(x -> convert(Tuple, x), rho.inds)
    println(io, "ReducedDensityMatrix over sites:")
    foreach(x -> println(io, x), tupinds)
    println("with data:")
    Base.show(io, rho.data)
    return nothing
end

TensorKit.codomain(rho::ReducedDensityMatrix) = codomain(rho.data)
TensorKit.domain(rho::ReducedDensityMatrix) = domain(rho.data)

TensorKit.blocksectors(rho::ReducedDensityMatrix) = blocksectors(rho.data)

TensorKit.block(rho::ReducedDensityMatrix, s::Sector) = TensorKit.block(rho.data, s)
TensorKit.blocks(rho::ReducedDensityMatrix) = TensorKit.blocks(rho.data)
TensorKit.hasblock(rho::ReducedDensityMatrix, s::Sector) = TensorKit.hasblock(rho.data, s)

function TensorKit.adjoint(rho::ReducedDensityMatrix)
    return ReducedDensityMatrix(adjoint(rho.data), rho.inds)
end

function Base.similar(
    rho::ReducedDensityMatrix{S,N}, T::Type{<:Number}, space::TensorMapSpace{S,N,N}
) where {S,N}
    return ReducedDensityMatrix(similar(rho.data, T, space), rho.inds)
end

function TensorKit.storagetype(
    ::Union{T,Type{T}}
) where {S,N,A,T<:ReducedDensityMatrix{S,N,A}}
    return storagetype(A)
end

tensorind(rho::ReducedDensityMatrix, i::NTuple{2,Int}) = tensorind(rho, CartesianIndex(i))
function tensorind(rho::ReducedDensityMatrix, i::CartesianIndex{2})
    inds = rho.inds
    if i ∉ inds
        throw(ArgumentError("index $i not in the indices of $rho"))
    else
        return findfirst(isequal(i), inds)
    end
end
tensorind(::ReducedDensityMatrix, i::Int) = i

siteind(rho::ReducedDensityMatrix, i::Int) = rho.inds[i]

function ReducedDensityMatrix(qs::DensityMatrix, sites...)
    tensor = partialtrace(qs, sites...)
    return ReducedDensityMatrix(tensor, sites)
end

function permutesites(rho::ReducedDensityMatrix{S,N}, p::NTuple{N,Int}) where {S,N}
    return ReducedDensityMatrix(
        permute(rho.data, p, p .+ N), TupleTools.permute(rho.inds, p)
    )
end

function TensorKit.permute(
    rho::ReducedDensityMatrix{S,N}, p::NTuple{2,NTuple{N,Int}}
) where {S,N}
    return permute(rho, p[1], p[2])
end
function TensorKit.permute(
    ::ReducedDensityMatrix{S,N}, p1::NTuple{N,Int}, p2::NTuple{N,Int}
) where {S,N}
    throw(
        ArgumentError(
            "TensorKit.permute not supported for ReducedDensityMatrix; use `permutesites` instead.",
        ),
    )
end

LinearAlgebra.ishermitian(rho::ReducedDensityMatrix) = ishermitian(rho.data)

"""
    expval(rho::ReducedDensityMatrix, ops::Vararg{T, N}; inds=NTuple{N}(1:N))

Calculate the expectation value of the local operators `ops` applied to the indices
`inds` respectively. The keyword `inds` defaults to the first `N` indices for `N` local
operators.
"""
function expval(
    rho::ReducedDensityMatrix{S,N}, ops::Vararg{T,M}; inds=NTuple{M}(1:M)
) where {S,T,N,M}
    pdim = dims(codomain(rho))

    inds = map(x -> tensorind(rho, x), inds)

    j = 1
    matrices = map(1:N) do i
        mat = zeros(ComplexF64, pdim[i], pdim[i])
        if i ∈ inds
            mat += ops[j]
            j += 1
        else
            mat += I
        end
        return mat
    end

    return expval(rho, matrices...)
end
function expval(rho::ReducedDensityMatrix{S,N}, a::Vararg{Matrix{ComplexF64},N}) where {S,N}
    observable = mapreduce(⊗, enumerate(a)) do (i, op)
        return TensorMap(op, domain(rho)[i], codomain(rho)[i])
    end
    return tr(observable * rho)
end

"""
    partialtranspose(rho::ReducedDensityMatrix{S,N,T}, i) -> T

Returns the tensor obtained by transposing the `i`th index of `rho`.
"""
partialtranspose(rho::ReducedDensityMatrix, i) = partialtranspose(rho.data, i)
function partialtranspose(t::AbsTen{N,N}, i) where {N}
    codp = NTuple{N,Int}(j == i ? N + i : j for j in 1:N)
    domp = NTuple{N,Int}(j == i ? i : N + j for j in 1:N)
    tp = permute(t, (codp, domp))
    return tp
end

"""
    negativity(rho::ReducedDensityMatrix{S,2}) -> Number

Calculate the entanglement negativity of bipartite `rho`.
"""
function negativity(t)
    pt = partialtranspose(t, 1)

    tracenorm = tr(sqrt(pt' * pt))

    return 1 / 2 * (tracenorm - 1)
end

"""
    correlator(rho::DensityMatrix, A1, i1, A2, i2)

Calculate the two-point correlator of `rho` between observables `A1` and `A2` on sites `i1`
end `i2` respectively.
"""
function correlator(rho::DensityMatrix, A1, i1, A2, i2)
    dm = ReducedDensityMatrix(rho, i1, i2)
    return expval(dm, A1, A2)
end

"""
    trace(rho::DensityMatrix) -> AbstractLattice

Calculate the full trace density of the density matrix `rho` for each site in the unit cell.
Note, if `rho` is constructed properly, then `trace(rho) .≈ one(scalartype(rho))`.
"""
trace(rho::DensityMatrix) = trace(rho.data, rho.environment)
function trace(network::AbstractLattice, contractor)
    rv = similar(network, scalartype(network))
    vals = map(i -> trace(network, contractor, i), CartesianIndices(network))
    copyto!(rv, vals)
    return rv
end
function trace(network::AbstractLattice, contractor, ind)
    rv = zero(scalartype(network))

    x = ind[1]
    y = ind[2]

    boundary = get_contraction_boundary(contractor, x:x, y:y)

    trval = _partialtrace(rv, boundary..., network[x:x, y:y], Val((false)))::typeof(rv)

    return trval
end
"""
    partialtrace(rho::DensityMatrix, inds...) -> ReducedDensityMatrix

Calculate the partial trace of `rho` returning a `ReducedDensityMatrix` object. Indices in
`inds` define those sites that well remain *untraced*. Site indices in `inds` are mapped
to tensor indices in the same order as in `inds`.
"""
partialtrace(rho::DensityMatrix, inds::Vararg) =
    partialtrace(rho.data, rho.environment, inds...)

function partialtrace(network::AbstractLattice, trace, inds::Vararg{T,N}) where {T,N}
    if length(inds) == 0
        throw(ArgumentError("you must provide at least one site to remain untraced"))
    end

    cinds = map(CartesianIndex, inds)

    p = TupleTools.sortperm(cinds)

    cinds_sorted = TupleTools.permute(cinds, p)

    sortedrho = _partialtrace(network, trace, cinds_sorted...)

    rho = permute(sortedrho, (TupleTools.invperm(p), N .+ TupleTools.invperm(p)))

    return rho
end

function _partialtrace(
    network::AbstractLattice, contractor, inds::Vararg{CartesianIndex{2},N}
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

    rv = TensorMap(undef, scalartype(network), cod, dom)

    MS = [network[i, j] for i in lx:rx, j in ly:ry]

    trval = _partialtrace(rv, boundary..., MS, issite)::typeof(rv)

    return trval
end

function get_contraction_boundary(con::InfiniteContraction, xr::UnitRange, yr::UnitRange)
    return get_contraction_boundary(con.runtime, xr, yr)
end
function get_contraction_boundary(vumps::VUMPSRuntime, xr::UnitRange, yr::UnitRange)
    if length(yr) != 1
        throw(ArgumentError("cannot contract across multiple rows using VUMPS"))
    end

    i2 = first(yr)

    l = first(xr)
    r = last(xr)

    _, _, AR, AC = ITC.unpack(vumps.mps)

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

    _, _, AR, AC = ITC.unpack(vumps.mps)

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


function lefteigen(TMS, A, M, numeigs = 1, which = :LM; kwargs...)
    fl = ITC._initfixedpoints(rand, A[1, 1], M[1,1], :left)
    return eigsolve(fl, numeigs, which;kwargs...) do x0
        fl = x0
        for tm in TMS
            fl = fl * tm
        end
        return fl
    end
end


function testextrap(;Jy = 1.2, D = 4)
    s = ComplexSpace(2)
    S = ComplexSpace(D)

    rho = fill((1,0,0),2,2)

    xi0 = 100.0
    model = PEPOKit.Heisenberg(LADDER[1];  Jx = 0.5, Jy = Jy, Jz= 1.0);
    method = PEPOKit.TEBD(breaktol = 1e-5, trotterorder=1, numsteps=10000, timestep=0.01, truncalg=PEPOKit.SU(ξ = xi0))
    sim = Simulation([PEPOKit.OnAxis(model,1),PEPOKit.OnAxis(model,2)], rho, S; physical=s, method=method)

    alg = VUMPS(bonddim=16)

    xis = []
    xvals = []
    clens = []

    for i in 1:25

        xi0 += 20.0

        method = PEPOKit.TEBD(breaktol = 1e-5, trotterorder=1, numsteps=10000, timestep=0.01, truncalg=PEPOKit.SU(ξ = xi0))
        sim = Simulation(sim.model, PEPOKit.quantumstate(sim), method, SimulationInfo())

        simulate!(sim)

        if i < 6
            continue
        end

        pepo = PEPOKit.quantumstate(sim)

        dm = DensityMatrix(pepo, VUMPS(bonddim=32))

        rhos = map(ind -> ReducedDensityMatrix(dm, ind), uniqueindices(pepo))
        expt = map(x -> expval(x, PAULI[1]), rhos)

        clen = PEPOKit.correlationlength(dm)

        push!(xis, xi0)
        push!(xvals, real.(expt)[1,1])
        push!(clens, clen)

    end

    return xis, xvals, clens
end

function testCFconv(; D = 4)
    s = ComplexSpace(2)
    S = ComplexSpace(D)

    rho = fill((1,0,0),2,2)


    xvals = []
    clens = []

    method = PEPOKit.TEBD(breaktol = 1e-5, trotterorder=1, numsteps=10000, timestep=0.01, truncalg=PEPOKit.SU(ξ = Inf))

    for Jy in (Js = 1.45:0.05:1.55)

        model = PEPOKit.Heisenberg(LADDER[1];  Jx = 0.5, Jy = Jy, Jz= 1.0);

        sim = Simulation([PEPOKit.OnAxis(model,1),PEPOKit.OnAxis(model,2)], rho, S; physical=s, method=method)

        simulate!(sim)

        pepo = PEPOKit.quantumstate(sim)

        dm = DensityMatrix(pepo, VUMPS(bonddim=32))

        rhos = map(ind -> ReducedDensityMatrix(dm, ind), uniqueindices(pepo))
        expt = map(x -> expval(x, PAULI[1]), rhos)

        clen = PEPOKit.correlationlength(dm)

        push!(xvals, real.(expt)[1,1])
        push!(clens, clen)

    end

    return Js, xvals, clens
end
function _correlationlength(dm::DensityMatrix, op = one(PAULI[1]))
    mps = dm.environment.runtime.mps

    AL, _, _, _ = ITC.unpack(mps)

    bulk = contractphysical.(dm.data)

    funcop = b -> TensorMap(kron(op, one(op)), codomain(b), codomain(b))

    bulkop = broadcast(dm.data) do b
        ten =  funcop(b) * b
        return contractphysical(ten)
    end
            

    tms = ITC.TransferMatrix.(AL[:, 1], bulk[:, 1], AL[:, 2])
    tmsop = ITC.TransferMatrix.(AL[:, 1], bulkop[:, 1], AL[:, 2])

    fl = ITC._initfixedpoints(rand, AL[1, 1], bulk[1,1], :left)

    xis, _, _ = eigsolve(fl, 4, :LM) do x0
        fl = x0
        for tm in tms
            fl = fl * tm
        end
        return fl
    end

    @. xis = xis / xis[1] 

    eps1 = -log(abs(xis[2]))
    ind = isapprox(abs(xis[2]), abs(xis[3])) ? 4 : 3
    eps2 = -log(abs(xis[ind]))

    display(xis)

    return eps1, eps2
end

function correlationlength(dm::DensityMatrix) 
    eps1, _ = _correlationlength(dm)
    return 1 / eps1
end

function correlationlength(f, qs, chis = 8:1:16)

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

function _testising(;xi = Inf, D = 2)

    rho = fill(ThermalState(),2,2)

    # βc = log(1 + sqrt(2)) / 2
    βc = 1/1.2737

    H = Heisenberg(;  Jz = -1.0, hx = 2.5);
    # Until b = 0.8
    method_coarse = TEBD(
        numsteps=1050, 
        timestep=0.001*βc, 
        truncalg=PEPOKit.NTU(ξ = xi)
    )
    method_fine= TEBD(
        numsteps=20, 
        timestep=0.02*βc, 
        truncalg=PEPOKit.SU(ξ = xi)
    )

    model = [OnAxis(H,1), OnAxis(H,2)]

    alg = VUMPS(bonddim=4, maxiter=200, tol=1e-10)

    s = ComplexSpace(2)
    S = ComplexSpace(D)

    output = Float64[]

    sim = Simulation(model, rho, S; physical = s, method = method_coarse)

    # simulate!(sim)
    #
    t0 = sim.info.iterations * method_coarse.timestep
    t0 = 0

    # sim = Simulation(model, quantumstate(sim), method_fine, SimulationInfo())

    # Observables
    Zs = Float64[]
    Ls = Tuple{Vector{Float64},Vector{Float64}}[]
    ts = Float64[]
    converged = Bool[]

    simulate!(sim; maxshots=525) do sim

        if sim.info.iterations < 950
             return sim
        end


        dm = DensityMatrix(quantumstate(sim), alg)
        rdm = ReducedDensityMatrix(dm, (1,1))

        Z = abs(expval(rdm, PAULI[3]))
        L = correlationlength(dm)

        # ds,es = PEPOKit.correlationlength(
        #     D -> VUMPS(bonddim=D, tol=1e-10), 
        #     PEPOKit.quantumstate(sim), 
        #     20:1:20
        # )
        
        push!(Zs, Z)
        push!(Ls, ([L],[L]))

        push!(converged, dm.environment.info.converged)
        push!(ts, 1 / (t0 + sim.info.iterations * method_coarse.timestep))
    end

    return Zs, Ls, ts, converged
end

function testising(;D = 2)

    # xis = [1e4,2e4,4e4,8e4]
    xis = [exp10(i) for i in 5:0.5:6]
    # xis = [exp10(i) for i in 3:0.5:4]
    push!(xis, Inf)

    Zs = Vector{Float64}[]
    Ls = Vector{Tuple{Vector{Float64},Vector{Float64}}}[]
    Ts = Vector{Float64}[]
    converged = Vector{Bool}[]

    out = foreach(xis) do xi
        zs,ls,ts,cs = _testising(;xi = xi, D = D)

        push!(Zs,zs)
        push!(Ls,ls)
        push!(Ts,ts)

        push!(converged,cs)
    end

    return xis,Zs,Ls,Ts,converged
end

function test_ising_extrap(;D = 2, numsteps = 1000)
    βc = log(1 + sqrt(2)) / 2

    xis = [exp10(i) for i in 4:0.1:6]

    Ls = Float64[]
    Zs = Float64[]

    rho = fill(ThermalState(),2,2)

    s = ComplexSpace(2)
    S = ComplexSpace(D)

    H = Heisenberg(;  Jz = -1.0);
    model = [OnAxis(H,1), OnAxis(H,2)]

    alg = VUMPS(bonddim=16, maxiter=200, tol=1e-10)

    for xi in [xis..., Inf]

        method = TEBD(
            numsteps=numsteps, 
            timestep=0.001*βc, 
            truncalg=PEPOKit.SU(ξ = xi)
        )

        sim = Simulation(model, rho, S; physical = s, method = method)

        simulate!(identity, sim; maxshots=numsteps)

        dm = DensityMatrix(quantumstate(sim), alg)

        rdm = ReducedDensityMatrix(dm, (1,1))
        Z = abs(expval(rdm, PAULI[3]))

        L = correlationlength(dm)

        push!(Ls, L)
        push!(Zs, Z)
    end

    return xis, Zs, Ls
end

