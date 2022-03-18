#=
    |  ^
    ____
<--|    |<--
<--|    |<--
   |  M |
-->|    |-->
-->|____|-->
    |  ^
Index convention for position in index array, is codomain counterclockwise from most 
west-northerly (west first) positon, then domain in same order. 
=#
function applyH1(AC, FL, FR, M)
    @tensor HAC[l ld; rd r] :=
        FL[l 8 7; 2 1 a] *
        AC[a 12; 6 b] *
        M[1 2 ld 4 5 6; 7 8 rd 10 11 12] *
        FR[11 10 b; 5 4 r]
    return HAC
end

function applyH0(C, FL, FR)
    @tensor HC[L; R] := FL[L 2 3; 4 5 a] * C[a; b] * FR[5 4 b; 3 2 R]
    return H0
end

function _randFL(AL, M)
    spM = hcat(codomain(M)..., domain(M)...)
    spA = hcat(codomain(AL)..., domain(AL)...)
    return TensorMap(randn, ComplexF64, spA[4] * spM[7] * spM[8], spM[1] * spM[2] * spA[1])
end

function _randFR(AR, M)
    spM = hcat(codomain(M)..., domain(M)...)
    spA = hcat(codomain(AR)..., domain(AR)...)
    return TensorMap(
        randn, ComplexF64, spM[11] * spM[10] * spA[4], spM[5] * spmM[4] * spA[1]
    )
end

function leftenv(
    AL::TensorMap{S,2,2}, M::TensorMap{S,6,6}, FL::TensorMap{S,3,3}=_randFL(AL, M)
) where {S<:IndexSpace}
    λs, FLs, info = eigsolve(FL, 1, :LM; eager=true, ishermitian=false) do FL
        @tensor FL[-1 -2 -3; -4 -5 -6] :=
            AL[u 12; 6 -6] *
            FL[d 8 7; 2 1 u] *
            M[1 2 3 -2 -3 6; 7 8 9 -4 -5 12] *
            (AL')[-1 9; d 3]
    end
    return FLs[1], real(λs[1]), info
end

function rightenv(
    AR::TensorMap{S,2,2}, M::TensorMap{S,6,6}, FR::TensorMap{S,3}=_randFR(AR, M)
) where {S<:IndexSpace}
    λs, FRs, info = eigsolve(FR, 1, :LM; eager=true, ishermitian=false) do FR
        @tensor FR[-1 -2 -3; -4 -5 -6] :=
            AR[-3 12; 6 u] *
            M[-1 -2 3 4 5 6; -4 -5 9 10 11 12] *
            FR[11 10 u; 5 4 d] *
            (AR')[d 9; -6 3]
    end
    return FRs[1], real(λs[1]), info
end
