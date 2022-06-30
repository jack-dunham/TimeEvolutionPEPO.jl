struct HilbertSchmidtMPO{T<:TensorMap}
    tensor::T
end

# function hsbondenv(vhs::TensorMap{ComplexSpace,6,6}, D::Integer)
#     sp = hcat(codomain(vhs)..., domain(vhs)...)
#     A = TensorMap(randn, ComplexF64, sp[9] * ℂ^D, sp[6] * ℂ^D)
#     λ, AL, C, AR, FL, FR = vumps(A, vhs; maxiter=100, verbose=true)
#     return _construct_hsbondenv(vhs, AL, C, AR, FL, FR)
# end
function hsbondenv(vhs::TensorMap{S,6,6}, χ::Integer) where {S<:IndexSpace}
    unitsp = oneunit(S)
    bondsp = ⊕(fill(unitsp,χ)...)

    sp = hcat(codomain(vhs)..., domain(vhs)...)
    A = TensorMap(randn, ComplexF64, sp[9] * bondsp, sp[6] * bondsp)

    @time begin
    λ, AL, C, AR, FL, FR = vumps(A, vhs; maxiter=100, verbose=true)
    end

    return _construct_hsbondenv(vhs, AL, C, AR, FL, FR)
end

# correct as of new convention
function _construct_hsbondenv(
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

function hsnormalt(vp, D)
    d = 2
    η = 1
    hsd = convert(Array, _hilbertschmidt(vp)) # (d, η, d, d, η, d, d, η, d, d, η,d)  

    hsp = permutedims(hsd, (1, 2, 10, 11, 3, 12, 7, 8, 4, 5, 9, 6))
    hs = TensorMap(
        reshape(hsp, d * η * d * η, d * d, d * η * d * η, d * d),
        ℂ^(d * η * d * η) * ℂ^(d * d),
        ℂ^(d * η * d * η) * ℂ^(d * d),
    )

    sp = hcat(codomain(hs)..., domain(hs)...)
    A = TensorMap(randn, ComplexF64, sp[4] * ℂ^D, ℂ^D)
    λ, AL, C, AR, FL, FR = vumps(A, hs; maxiter=100, verbose=true)
    AC = _AC(AL, C)
    @tensoropt begin
        n = scalar(
            FL[b1; 1 a1] * AC[4 a1; a2] * hs[1 2; 3 4] * (AC')[b2; 2 b1] * FR[3 a2; b2]
        )
    end
    return n
end

function tracevumps_test(D)
    p = blochpepo([0.33, 0.33, 0.33], 2)
    vp = permute(p, (4, 1, 2, 3), (5, 6))
    @tensor vtr[2 3; 5 6] := vp[s s 2 3; 5 6]
    sp = hcat(codomain(vtr)..., domain(vtr)...)
    A = TensorMap(randn, ComplexF64, sp[4] * ℂ^D, ℂ^D)

    λ, AL, C, AR, FL, FR = vumps(A, vtr; maxiter=20, verbose=true, tol=1e-20)
    @assert isapprox(_AC(AL, C), _CA(C, AR), rtol=1e-4)
    AC = _AC(AL, C)
    @tensoropt begin
        ptr[o; i] :=
            FL[b1; 1 a1] * AC[4 a1; a2] * p[o 1 2; i 3 4] * (AC')[b2; 2 b1] * FR[3 a2; b2]
    end

    ptr = ptr / tr(ptr)

    return ptr
end

# function _construct_hsbondenv(hs::HilbertSchmidtMPO,
#     mps::BoundaryInfMPS, env::BoundaryEnvironment)
#     return _construct_hsbondenv(M.tensor, mps.AL, mps.C, mps.AR, env.FL, env.FR)
# end

# old convention
# function hilbertschmidt(vp1::TensorMap{S,4,3}, vp2::TensorMap{S,4,3}) where {S<:IndexSpace}
#     @tensoropt vhs[1 2 3 4 5 6; 7 8 9 10 11 12] :=
#         vp1[1 2 3 s; 10 11 12] * (vp2')[4 5 6; 7 8 9 s]
#     return vhs
# end

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
function _hilbertschmidt(
    vp1::TensorMap{S,5,3}, vp2::TensorKit.AdjointTensorMap{S,3,5}
) where {S<:IndexSpace}
    @tensoropt begin
        vhs[1 2 3 4 5 6; 7 8 9 10 11 12] := vp1[b k 1 2 3; 7 8 9] * vp2[4 5 6; b k 10 11 12]
    end
    return vhs
end
function _hilbertschmidt(
    vp1::TensorMap{S,4,2}, vp2::TensorKit.AdjointTensorMap{S,2,4}
) where {S<:IndexSpace}
    @tensoropt begin
        vhs[1 2 3 4; 5 6 7 8] := vp1[b k 1 2; 5 6] * vp2[3 4; b k 7 8]
    end
    return vhs
end
_hilbertschmidt(vp::TensorMap) = _hilbertschmidt(vp, vp')

# function hilbertschmidt(
#     vp1::VectorSitePEPO{<:TensorMap{S}}, vp2::AdjointVectorSitePEPO{<:TensorMap{S}}
# ) where {S<:IndexSpace}
#     return _hilbertschmidt(vp1.tensor, vp2.tensor)
# end
# function hilbertschmidt(
#     vp1::VectorSitePEPO{<:TensorMap{S}}, vp2::VectorSitePEPO{<:TensorMap{S}}
# ) where {S<:IndexSpace}
#     return _hilbertschmidt(vp1.tensor, (vp2.tensor)')
# end
# hilbertschmidt(vp::VectorSitePEPO) = _hilbertschmidt(vp.tensor)

##### api for this not yet implemented 
# function sqhsnorm(vp::VectorSitePEPO, D::Int, alg::AbstractAlg)
#     hstensor = hilbertschmidt(vp)

#     sp = hcat(codomain(vp)..., domain(vp)...)
#     A = TensorMap(randn, ComplexF64, sp[9] * ℂ^D, sp[6] * ℂ^D)
#     out = boundary(A, hstensor, alg=alg)
# end
