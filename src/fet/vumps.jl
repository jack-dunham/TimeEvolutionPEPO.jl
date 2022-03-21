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
    return FLs[1], λs[1], info
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
    return FRs[1], λs[1], info
end

"""
    leftgauge(A::TensorMap, [C]; kwargs...)

Given a MPS tensor `A`, return a left-canonical MPS tensor `AL`, a gauge transform `C` and
a scalar factor `λ` such that ``λ AL^s C = C A^s``, where an initial guess for `C` can be
provided.
"""
function leftgauge(A, Li, tol = 1e-12, maxiter = 100)
    AL, C, R = _leftgaugeinitial(A, Li)
    numiter = 1
    while norm(C-R) > tol && numiter < maxiter
        λs, Cs, info = eigsolve(R, 1, :LM; ishermitian = false, tol = tol, eager=true, maxiter = 1) do X
            @tensor X[d ; u]:=A[1 2 ; 3 u]*(AL')[d 3 ; b 2]*X[b ; a]
        end
        # _, C = leftorth(Cs[1])

        # @tensor CA[1 2 ; 3 4]:= C[1;a]*A[a 2; 3 4]
        # Q, R = leftorth(CA, (1,2,3), (4,))
        # AL = permute(Q, (1,2), (3,4))

        # R = R/norm(R)

        AL, C, R = _leftgaugefactor(Cs[1], A)
        numiter += 1
    end
    C = R
    return AL, C, λ
end

function _leftgaugefactor(C, A)
    _, C = leftorth(C)
    @tensor CA[1 2 ; 3 4]:= C[1;a]*A[a 2; 3 4]
    Q, R = leftorth(CA, (1,2,3), (4,))
    AL = permute(Q, (1,2), (3,4))

    R = R/norm(R)
    return AL, C, R 
end

function _leftgaugeinitial(A, Li=TensorMap(randn, ComplexF64, codomain(A)[1], codomain(A)[1]))
    λs, Ls, info = eigsolve(Li'*Li, 1, :LM; ishermitian = false, tol = tol, maxiter = 1, eager=true) do L
        @tensor L[d ; u]:=A[1 2 ; 3 u]*(A')[d 3 ; b 2]*L[b ; a]
    end

    L = Ls[1]
    U, Σ, Vt = tsvd(L)
    l = sqrt(Σ)*Vt

    AL, l, R = _leftgaugefactor(l, A)

    # _, l = leftorth(l)

    # @tensor lA[1 2 ; 3 4]:= l[1;a]*A[a 2; 3 4]
    # Q, R = leftorth(lA, (1,2,3), (4,))
    # AL = permute(Q, (1,2), (3,4))

    return AL, l, R
end

function _vumpsstep(AL, C, AR, M, FL, FR; kwargs...)
    @tensor AC[-1 -2 ; -3 -4]:=AL[-1 -2 ; -3 4]*C[4, -4]
    μ1s, ACs, info1 = eigsolve(x->applyH1(x, FL, FR, M), AC, 1, :LM; ishermitian = false, eager=true, maxiter = 1, kwargs...)
    μ0s, Cs, info0 = eigsolve(x->applyH0(x, FL, FR), C, 1; ishermitian = false, eager=true, maxiter = 1, kwargs...)

    λ = real(μ1s[1]/μ0s[1])
    AC = ACs[1]
    C = Cs[1]
    
    QAC, RAC = leftorth(AC, (1,2,3), (4,))
    QC, RC = leftorth(C)
    AL = permute(QAC*QC', (1,2),(3,4))
    errL = norm(RAC-RC)

    LAC, QAC = rightorth(AC, (1,), (2,3,4))
    LC, QC = rightorth(C)
    AL = permute(QC'*QAC, (1,2),(3,4))
    errR = norm(LAC-LC)

    return λ, AL, C, AR, errL, errR
end
