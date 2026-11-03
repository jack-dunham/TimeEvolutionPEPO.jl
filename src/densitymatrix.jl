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

"""
$(TYPEDEF)

A purified density matrix `ρ = X†X`, with `X` the PEPO and the boundary environment of the resulting
two-layer network. For thermal evolution only: a Lindblad state is a density matrix already and takes
[`DensityMatrix`](@ref).

# Constructors

    PurifiedDensityMatrix(pepo::InfinitePEPO, bonddim; tol = 1e-10, maxiter = 200, verbosity = 0)
"""
struct PurifiedDensityMatrix{P <: InfinitePEPO, E <: CTMRGEnv, I}
    state::P
    environment::E
    info::I
end

function PurifiedDensityMatrix(
        pepo::InfinitePEPO, bonddim; tol = 1.0e-10, maxiter = 200, verbosity = 0
    )
    size(pepo, 3) == 1 ||
        throw(DimensionMismatch("only single-layer density matrices are supported"))

    # Sandwiching the PEPO with its adjoint is what `InfiniteSquareNetwork` builds for a PEPO.
    network = InfiniteSquareNetwork(pepo)
    env₀ = initialize_ctmrg_environment(
        ComplexF64, network, RandomInitialization(), _indexspace(bonddim)
    )
    env, info = leading_boundary(env₀, network; tol, maxiter, verbosity)

    return PurifiedDensityMatrix(pepo, env, info)
end

isconverged(dm::PurifiedDensityMatrix) = dm.info.converged

function partialtrace(dm::PurifiedDensityMatrix, inds...)
    return reduced_densitymatrix(
        [CartesianIndex(i) for i in inds], dm.state, dm.state, dm.environment
    )
end

Base.show(io::IO, dm::PurifiedDensityMatrix) =
    print(io, "PurifiedDensityMatrix of unit cell size ", size(dm.state)[1:2])

"""
    $(TYPEDSIGNATURES)

Largest correlation length of the boundary, over the rows and columns of the unit cell. PEPSKit's
`correlation_length` returns the per-row and per-column values along with the transfer spectra.
"""
function correlationlength(dm::Union{DensityMatrix, PurifiedDensityMatrix}; kwargs...)
    horizontal, vertical, = correlation_length(dm.state, dm.environment; kwargs...)
    return max(maximum(horizontal), maximum(vertical))
end

"""
    $(TYPEDSIGNATURES)

Two-point correlator of `A1` at `i1` with `A2` at `i2`.
"""
function correlator(dm::Union{DensityMatrix, PurifiedDensityMatrix}, A1, i1, A2, i2)
    return expval(partialtrace(dm, i1, i2), A1, A2)
end

"""
    $(TYPEDSIGNATURES)

Entanglement negativity `(‖ρᵗᴮ‖₁ - 1) / 2` of a two-site reduced density matrix, with the transpose
taken on the second site.
"""
function negativity(rdm::AbstractTensorMap)
    numout(rdm) == 2 || throw(
        ArgumentError("negativity needs a two-site reduced density matrix, got $(numout(rdm)) sites")
    )
    d = dim(space(rdm, 1))
    arr = reshape(convert(Array, rdm), d, d, d, d)      # (out1, out2, in1, in2)
    transposed = reshape(permutedims(arr, (1, 4, 3, 2)), d^2, d^2)

    return (sum(abs, eigvals(transposed)) - 1) / 2
end
