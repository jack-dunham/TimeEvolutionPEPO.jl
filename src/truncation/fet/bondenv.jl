# const Post_HS_MPOTensor{S,T} = MPOTensor{
#     S,:POST_HS_SIG,6,6,T,MPOSignature{:POST_HS_SIG,2,2,1,1,2,2,1,1}
# }

# const Post_HS_InfMPO{Nx,Ny,S,P<:Post_HS_MPOTensor{S}} = InfMPO{Nx,Ny,S,:POST_HS_SIG,P}

abstract type AbstractMetricTensor{S,N₁,N₂,T} <: AbstractPhysicalTensor{S,N₁,N₁,T} end
# For truncating a bond
abstract type AbstractTruncationMetricTensor{S,T} <: AbstractMetricTensor{S,4,4,T} end
# For calculating traces
abstract type AbstractTraceMetricTensor{S,T} <: AbstractMetricTensor{S,2,2,T} end

abstract type AbstractTruncationMetric{L,P<:AbstractMetricTensor} <:
              AbstractTensorLattice{L,P} end

struct TruncationMetricTensor{S,T} <: AbstractTruncationMetricTensor{S,T}
    tensor::T
end

struct TruncationMetric{L,P<:AbstractTruncationMetricTensor} <:
       AbstractTruncationMetric{L,P}
    data::Matrix{P}
end

# function hsbondenv(vhs::TensorMap{ComplexSpace,6,6}, D::Integer)
#     sp = hcat(codomain(vhs)..., domain(vhs)...)
#     A = TensorMap(randn, ComplexF64, sp[9] * ℂ^D, sp[6] * ℂ^D)
#     λ, AL, C, AR, FL, FR = vumps(A, vhs; maxiter=100, verbose=true)
#     return _construct_hsbondenv(vhs, AL, C, AR, FL, FR)
# end

# function randhsmps(vhs::TensorMap{S,6,6}, χ::Int) where {S<:IndexSpace}
#     unitsp = oneunit(S)
#     bondsp = ⊕(fill(unitsp,χ)...)

#     sp = hcat(codomain(vhs)..., domain(vhs)...)
#     A = TensorMap(randn, ComplexF64, sp[9] * bondsp, sp[6] * bondsp)
#     return A
# end

# alg could be ::VUMPS, ::CTMRG, ::FPCM, ::RG 

@doc raw"""
    truncmetric(state::Union{AbstractBoundaryState, AbstractNeighbourhoodState})
    truncmetric(mpo::InfMPO, tensors::Union{AbstractBoundary, AbstractNeighbourhood})

Calculate the truncation metric from the state of a boundary algorithm. Alternatively, given 
an MPO and some boundary tensors, calculate the truncation metric. 
"""
function truncmetric(Dη, state::AbstractBoundaryState)
    tensors = state.data
    mpo = state.mpo
    return truncmetric(Dη, mpo, tensors)
end
# MPO is a hilbertschmidt mpo
function truncmetric(Dη, mpo::InfMPO, tensors::VUMPS)
    Y = vumps_fetmetric(Dη, mpo, tensors.A, tensors.FL, tensors.FR)
    return Y
end

# TODO: Multisite
function hsbondenv(
    vhs::TensorMap{S,6,6}, A::TensorMap{S,2,2}; alg::VUMPS=VUMPS()
) where {S<:IndexSpace}
    # unitsp = oneunit(S)
    # bondsp = ⊕(fill(unitsp,χ)...)

    # sp = hcat(codomain(vhs)..., domain(vhs)...)
    # A = TensorMap(randn, ComplexF64, sp[9] * bondsp, sp[6] * bondsp)
    maxiter = alg.maxiter
    verbose = alg.verbose

    @time begin
        λ, AL, C, AR, FL, FR = vumps(A, vhs; maxiter=maxiter, verbose=verbose)
    end

    return construct_hsbondenv(vhs, AL, C, AR, FL, FR), _AC(AL, C)
end

# correct as of new convention
function construct_hsbondenv(
    M::AbstractTensorMap{S,6,6}, # a hilbertschmidt tensor
    AL::AbstractTensorMap{S,2,2},
    C::AbstractTensorMap{S,1,1},
    AR::AbstractTensorMap{S,2,2},
    FL::AbstractTensorMap{S,3,3},
    FR::AbstractTensorMap{S,3,3},
) where {S<:IndexSpace}
    @info "Constructing bond environment..."
    @time begin
        @tensoropt begin
            γ[-1 -2 4 5; 7 8 -10 -11] :=
                FL[10 11 b1; 1 2 a1] *
                AL[9 a1; 6 a2] *
                (AL')[12 b2; 3 b1] *
                M[1 2 3 4 5 6; 7 8 9 10 11 12] *
                C[a2; a3] *
                (C')[b3; b2] *
                M[-1 -2 -3 -4 -5 -6; -7 -8 -9 -10 -11 -12] *
                (AR')[-12 b4; -3 b3] *
                AR[-9 a3; -6 a4] *
                FR[-7 -8 a4; -4 -5 b4]
        end
    end
    return γ
end
# Multisite
function vumps_fetmetric(
    M1::AbstractTensorMap{S,6,6}, # (x, y)     a hilbertschmidt mpo tensor
    M2::AbstractTensorMap{S,6,6}, # (x + 1, y) a hilbertschmidt mpo tensor
    ALu::AbstractTensorMap{S,2,2},# (x, y)
    ALd::AbstractTensorMap{S,2,2},# (x, y + 1)
    Cu::AbstractTensorMap{S,1,1}, # (x, y)
    Cd::AbstractTensorMap{S,1,1}, # (x, y + 1)
    ARu::AbstractTensorMap{S,2,2},# (x + 1, y)
    ARd::AbstractTensorMap{S,2,2},# (x + 1, y + 1)
    FL::AbstractTensorMap{S,3,3}, # (x, y)
    FR::AbstractTensorMap{S,3,3}, # (x + 1, y)
) where {S<:IndexSpace}
    @info "Constructing bond environment..."
    @time begin
        @tensoropt begin
            γ[-1 -2 4 5; 7 8 -10 -11] :=
                FL[10 11 b1; 1 2 a1] *
                ALu[9 a1; 6 a2] *
                (ALd')[12 b2; 3 b1] *
                M1[1 2 3 4 5 6; 7 8 9 10 11 12] *
                Cu[a2; a3] *
                (Cd')[b3; b2] *
                M2[-1 -2 -3 -4 -5 -6; -7 -8 -9 -10 -11 -12] *
                (ARd')[-12 b4; -3 b3] *
                ARu[-9 a3; -6 a4] *
                FR[-7 -8 a4; -4 -5 b4]
        end
    end
    return γ
end
#PERF:
function vumps_fetmetric(
    M1::AbstractTensorMap{S,2,2}, # (x, y)     a hilbertschmidt mpo tensor
    M2::AbstractTensorMap{S,2,2}, # (x + 1, y) a hilbertschmidt mpo tensor
    ALu::AbstractTensorMap{S,2,1},# (x, y)
    ALd::AbstractTensorMap{S,2,1},# (x, y + 1)
    Cu::AbstractTensorMap{S,1,1}, # (x, y)
    Cd::AbstractTensorMap{S,1,1}, # (x, y + 1)
    ARu::AbstractTensorMap{S,2,1},# (x + 1, y)
    ARd::AbstractTensorMap{S,2,1},# (x + 1, y + 1)
    FL::AbstractTensorMap{S,1,2}, # (x, y)
    FR::AbstractTensorMap{S,2,1}, # (x + 1, y)
    U::AbstractTensorMap{S,1,2},    # for splitting the indices
) where {S<:IndexSpace}
    @info "Constructing bond environment..."
    @time begin
        @tensoropt begin
            y[11 10; 9 12] :=
                FL[3; 1 2] *
                ALu[1 4; 6] *
                (ALd')[8; 3 5] *
                M1[2 5; 7 4] *
                U[7; 9 10] *
                Cu[6; 13] *
                (Cd')[15; 8] *
                (U')[11 12; 14] *
                M2[14 17; 19 16] *
                ARu[13; 16 18] *
                (ARd')[20; 15 17] *
                FR[18 19; 20]
        end
    end
    return y
end

function vumps_fetmetric(Dη, M::InfMPO, A::InfMPS, FL::LeftEnv, FR::RightEnv)
    nx, ny = size(M)

    # Useful for obtaining the type of the matrix elements.
    # this is very janky way of doing this
    s = space(M[1, 1], :w)[1]
    p0 = TruncationMetricTensor(similar(M[1, 1].tensor, s * s, s * s)) #this is discarded!

    Ym = Matrix{typeof(p0)}(undef, nx, ny)

    # Get the tensor matrices of the MPS
    AL = lefttensor(A)
    AR = righttensor(A)
    C = bondtensor(A)

    U = [unitary(codomain[x, y][1], Dη[x, y] * Dη[x, y]) for x in 1:nx, y in 1:ny]

    ## Calculate the HS bond environment for each site in unit cell
    for x in 1:nx, y in 1:ny
        Ym[x, y] = TruncationMetricTensor(
            vumps_fetmetric(
                M[x, y],
                M[x + 1, y],
                AL[x, y],
                AL[x, y + 1],
                C[x, y],
                C[x, y + 1],
                AR[x + 1, y],
                AR[x + 1, y + 1],
                FL[x, y],
                FR[x + 1, y],
                U[x, y],
            ),
        )
    end
    return TruncationMetric(Y)
end

function hsnorm(vp, D)
    hs = _hilbertschmidt(vp)
    sp = hcat(codomain(hs)..., domain(hs)...)
    A = TensorMap(randn, ComplexF64, sp[9] * ℂ^D, sp[6] * ℂ^D)
    λ, AL, C, AR, FL, FR = vumps(A, hs; maxiter=100, verbose=true)
    AC = _AC(AL, C)
    @tensoropt begin
        n = scalar(
            FL[10 11 b1; 1 2 a1] *
            AC[9 a1; 6 a2] *
            hs[1 2 3 4 5 6; 7 8 9 10 11 12] *
            (AC')[12 b2; 3 b1] *
            FR[7 8 a2; 4 5 b2],
        )
    end
    return abs(n)#? why abs?...
end

#=
              9  6
              │  ▲
              ▼  │
            ┌────┴──┐
      1◄────┤       │◄────7
      2◄────┤       │◄────8
            │   M   │
     10────►│       ├────►4
     11────►│       ├────►5
            └─┬─────┘
              │  ▲
              ▼  │
              3  12
=#
#vectorised 
function hilbertschmidt(
    vp1::TensorMap{S,5,3}, vp2::TensorKit.AdjointTensorMap{S,3,5}
) where {S<:IndexSpace}
    @tensoropt begin
        vhs[1 2 3 4 5 6; 7 8 9 10 11 12] := vp1[b k 1 2 3; 7 8 9] * vp2[4 5 6; b k 10 11 12]
    end
    return vhs
end
#not vectorised
function hilbertschmidt(
    p1::TensorMap{S,4,4}, p2::TensorKit.AdjointTensorMap{S,4,4}
) where {S<:IndexSpace}
    @tensoropt begin
        hs[1 2 3 4 5 6; 7 8 9 10 11 12] := p1[k 1 2 3; b 7 8 9] * p2[b 4 5 6; k 10 11 12]
    end
    return hs
end
function hilbertschmidt(
    vp1::TensorMap{S,4,2}, vp2::TensorKit.AdjointTensorMap{S,2,4}
) where {S<:IndexSpace}
    @tensoropt begin
        vhs[1 2 3 4; 5 6 7 8] := vp1[b k 1 2; 5 6] * vp2[3 4; b k 7 8]
    end
    return vhs
end

#PERF: 
function hilbertschmidt(
    p1::TensorMap{S,3,3}, p2::TensorKit.AdjointTensorMap{S,3,3}
) where {S<:IndexSpace}
    @tensoropt begin
        vhs[1 -4 2 -5; 4 -1 5 -2] := p1[1 2 k; 4 5 b] * p2[-1 -2 b; -4 -5 k]
    end
    return vhs
end
hilbertschmidt(vp::TensorMap) = hilbertschmidt(vp, vp')

function hilbertschmidt(P::Lattice{L}) where {L}
    nx,ny = size(P)
    mat = hilbertschmidt.(P)
    return InfMPO{L}(mat)
end
