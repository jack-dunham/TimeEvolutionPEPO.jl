function hsbondenv(vhs::TensorMap{ComplexSpace,6,6}, D::Integer)
    sp = hcat(codomain(vhs)..., domain(vhs)...)
    A = TensorMap(randn, ComplexF64, sp[9] * ℂ^D, sp[6] * ℂ^D)
    λ, AL, C, AR, FL, FR = vumps(A, vhs; maxiter=100, verbose=true)
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
function hilbertschmidt(vp1, vp2)
    @tensoropt begin
        vhs[1 2 3 4 5 6; 7 8 9 10 11 12] := vp1[b k 1 2 3; 7 8 9] * vp2[4 5 6; b k 10 11 12]
    end
    return vhs
end

function hilbertschmidt(
    vp1::VectorPEPOTensor{S}, vp2::VectorPEPOTensor{S}
) where {S<:EuclideanSpace}
    return hilbertschmidt(vp1.tensor, vp2.tensor)
end

hilbertschmidt(vp::TensorMap{<:IndexSpace,3,2}) = hilbertschmidt(vp, vp)
hilbertschmidt(vp::VectorPEPOTensor) = hilbertschmidt(vp.tensor, vp.tensor)
