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

Expectation value of a one-site operator in the reduced density matrix `rdm`.
"""
function expval(rdm::AbstractTensorMap, op::AbstractMatrix)
    numind(rdm) == 2 ||
        throw(ArgumentError("expected a one-site reduced density matrix, got $(numind(rdm)) indices"))
    m = reshape(convert(Array, rdm), size(op))
    return tr(m * op)
end

Base.show(io::IO, dm::DensityMatrix) =
    print(io, "DensityMatrix of unit cell size ", size(dm.state)[1:2])
