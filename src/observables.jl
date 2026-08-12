# Observables on PEPSKit boundary environments. A mixed-state reduced density matrix needs the
# boundary of the traced network: `InfiniteSquareNetwork(pepo)` sandwiches the PEPO with its adjoint
# and gives the Hilbert-Schmidt norm instead.

"""
$(TYPEDEF)

A density-matrix PEPO together with the boundary environment of its trace. `partialtrace` returns
reduced density matrices normalised to unit trace.

# Constructors

    DensityMatrix(pepo::InfinitePEPO, bonddim; tol = 1e-10, maxiter = 200, verbosity = 0)
"""
struct DensityMatrix{P <: InfinitePEPO, E <: CTMRGEnv, I}
    state::P
    environment::E
    info::I
end

function DensityMatrix(
        pepo::InfinitePEPO, bonddim; tol = 1.0e-10, maxiter = 200, verbosity = 0
    )
    size(pepo, 3) == 1 ||
        throw(DimensionMismatch("only single-layer density matrices are supported"))

    network = InfiniteSquareNetwork(tracenetwork(pepo))
    env₀ = initialize_ctmrg_environment(
        ComplexF64, network, RandomInitialization(), _indexspace(bonddim)
    )
    env, info = leading_boundary(env₀, network; tol, maxiter, verbosity)

    return DensityMatrix(pepo, env, info)
end

"""
    $(TYPEDSIGNATURES)

Partition function obtained by tracing the physical indices of `pepo`, whose boundary environment is
the one a mixed-state reduced density matrix is contracted against.
"""
function tracenetwork(pepo::InfinitePEPO)
    return InfinitePartitionFunction(
        map(PEPSKit.trace_physicalspaces, PEPSKit.unitcell(pepo)[:, :, 1])
    )
end

isconverged(dm::DensityMatrix) = dm.info.converged

"""
    $(TYPEDSIGNATURES)

Reduced density matrix on `inds`, normalised to unit trace. Indices need not be adjacent.
"""
function partialtrace(dm::DensityMatrix, inds...)
    return reduced_densitymatrix(
        [CartesianIndex(i) for i in inds], dm.state, dm.environment
    )
end

"""
    $(TYPEDSIGNATURES)

Expectation value of a product of one-site operators in the reduced density matrix `rdm`, one operator
per site in the order the sites were given to [`partialtrace`](@ref).
"""
function expval(rdm::AbstractTensorMap, ops::AbstractMatrix...)
    n = length(ops)
    numout(rdm) == n || throw(
        ArgumentError("got $n operators for a $(numout(rdm))-site reduced density matrix")
    )
    d = size(first(ops), 1)
    all(op -> size(op) == (d, d), ops) ||
        throw(DimensionMismatch("operators differ in size"))

    # The dense reshape runs its composite index with the last site slowest, so the operator for the
    # first site is the last `kron` factor.
    return tr(reshape(convert(Array, rdm), d^n, d^n) * reduce(kron, reverse(ops)))
end

Base.show(io::IO, dm::DensityMatrix) =
    print(io, "DensityMatrix of unit cell size ", size(dm.state)[1:2])
