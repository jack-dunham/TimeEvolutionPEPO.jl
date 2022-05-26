#=
             12  6
              │  ▲
              ▼  │
            ┌────┴──┐
      1◄────┤       │◄────11
      2◄────┤       │◄────10
            │   M   │
      7────►│       ├────►5
      8────►│       ├────►4
            └─┬─────┘
              │  ▲
              ▼  │
              3  9

Index convention for position in index array, is codomain counterclockwise from most 
west-northerly (west first) positon, then domain in same order.

                           ┌───┐
     v ^               1◄──┤ a │◄──4
     2 3                   └┬──┘ 
--------------              │ ▲  
     4 1                    ▼ │
     v ^                    2 3

3 ->[ a']-> 2               4 1
                            │ ▲
      3                     ▼ │
      v                    ┌──┴┐      
2 ->[ a']->1           3──►│ a'│──►2 
                           └───┘     
=#

function applyH1(
    AC::TensorMap{S,2,2}, FL::TensorMap{S,3,3}, FR::TensorMap{S,3,3}, M::TensorMap{S,6,6}
) where {S<:IndexSpace}
    @tensor HAC[l ld; rd r] :=
        FL[l 8 7; 2 1 a] *
        AC[a 12; 6 b] *
        M[1 2 ld 4 5 6; 7 8 rd 10 11 12] *
        FR[11 10 b; 5 4 r]
    return HAC
end
function applyH1(
    AC::TensorMap{S,2,1}, FL::TensorMap{S,1,2}, FR::TensorMap{S,2,1}, M::TensorMap{S,2,2}
) where {S<:IndexSpace}
    @tensor HAC[l s; r] := FL[l; 1 a] * AC[a 4; b] * M[1 s; 3 4] * FR[3 b; r]
    return HAC
end
function applyH0(
    C::TensorMap{S,1,1}, FL::TensorMap{S,3,3}, FR::TensorMap{S,3,3}
) where {S<:IndexSpace}
    @tensor H0[L; R] := FL[L 2 3; 4 5 a] * C[a; b] * FR[5 4 b; 3 2 R]
    return H0
end
function applyH0(
    C::TensorMap{S,1,1}, FL::TensorMap{S,1,2}, FR::TensorMap{S,2,1}
) where {S<:IndexSpace}
    @tensor H0[L; R] := FL[L; m a] * C[a; b] * FR[m b; R]
    return H0
end

function _randFL(AL::TensorMap{S,2,2}, M::TensorMap{S,6,6}) where {S<:IndexSpace}
    spM = hcat(codomain(M)..., domain(M)...)
    spA = hcat(codomain(AL)..., domain(AL)...)
    return TensorMap(randn, ComplexF64, spA[1] * spM[8] * spM[7], spM[2] * spM[1] * spA[1])
end
function _randFL(AL::TensorMap{S,2,1}, M::TensorMap{S,2,2}) where {S<:IndexSpace}
    spM = hcat(codomain(M)..., domain(M)...)
    spA = hcat(codomain(AL)..., domain(AL)...)
    return TensorMap(randn, ComplexF64, spA[1], spM[1] * spA[1])
end

function _randFR(AR::TensorMap{S,2,2}, M::TensorMap{S,6,6}) where {S<:IndexSpace}
    spM = hcat(codomain(M)..., domain(M)...)
    spA = hcat(codomain(AR)..., domain(AR)...)
    return TensorMap(
        randn, ComplexF64, spM[11] * spM[10] * spA[4], spM[5] * spM[4] * spA[1]
    )
end
function _randFR(AR::TensorMap{S,2,1}, M::TensorMap{S,2,2}) where {S<:IndexSpace}
    spM = hcat(codomain(M)..., domain(M)...)
    spA = hcat(codomain(AR)..., domain(AR)...)
    return TensorMap(randn, ComplexF64, spM[3] * spA[3], spA[3])
end

function leftenv(
    AL::TensorMap{S}, M::TensorMap{S}; FL::TensorMap{S}=_randFL(AL, M)
) where {S<:IndexSpace}
    λs, FLs, info = eigsolve(
        x -> _flsolve(AL, M, x), FL, 1, :LM; eager=true, ishermitian=false
    )
    return FLs[1], λs[1], info
end

function _flsolve(
    AL::TensorMap{S,2,2}, M::TensorMap{S,6,6}, FL::TensorMap{S,3,3}
) where {S<:IndexSpace}
    @tensor FLu[-1 -2 -3; -4 -5 -6] :=
        AL[u 12; 6 -6] *
        FL[d 8 7; 2 1 u] *
        M[1 2 3 -2 -3 6; 7 8 9 -4 -5 12] *
        (AL')[9 -1; d 3]
    return FLu
end
function _flsolve(
    AL::TensorMap{S,2,1}, M::TensorMap{S,2,2}, FL::TensorMap{S,1,2}
) where {S<:IndexSpace}
    @tensor FLu[-1; -2 -3] := AL[u 4; -3] * FL[d; 1 u] * M[1 2; -2 4] * (AL')[-1; d 2]
    return FLu
end

function rightenv(
    AR::TensorMap{S}, M::TensorMap{S}; FR::TensorMap{S}=_randFR(AR, M)
) where {S<:IndexSpace}
    λs, FRs, info = eigsolve(
        x -> _frsolve(AR, M, x), FR, 1, :LM; eager=true, ishermitian=false
    )
    return FRs[1], λs[1], info
end

function _frsolve(
    AR::TensorMap{S,2,2}, M::TensorMap{S,6,6}, FR::TensorMap{S,3,3}
) where {S<:IndexSpace}
    @tensor FRu[-1 -2 -3; -4 -5 -6] :=
        AR[-3 12; 6 u] *
        M[-1 -2 3 4 5 6; -4 -5 9 10 11 12] *
        FR[11 10 u; 5 4 d] *
        (AR')[9 d; -6 3]
    return FRu
end
function _frsolve(
    AR::TensorMap{S,2,1}, M::TensorMap{S,2,2}, FR::TensorMap{S,2,1}
) where {S<:IndexSpace}
    @tensor FRu[-1 -2; -3] := AR[-2 4; u] * M[-1 2; 3 4] * FR[3 u; d] * (AR')[d; -3 2]
    return FRu
end

@doc raw"""
    leftgauge(A::TensorMap, [C]; kwargs...)

Given a MPS tensor `A`, return a left-canonical MPS tensor `AL`, a gauge transform `C` and
a scalar factor `λ` such that ``λ AL^s C = C A^s``, where an initial guess for `C` can be
provided.

                   ┌────┐   ┌───┐          ┌───┐   ┌───┐
            λ * ◄──┤ AL │◄──┤ C │◄─── = ◄──┤ C │◄──┤ A │◄──
                   └─┬──┘   └─┬─┘          └─┬─┘   └─┬─┘
                     ▼        ▼              ▼       ▼
                     s        s              s       s

"""
function leftgauge(
    A,
    C=TensorMap(randn, ComplexF64, codomain(A)[1], codomain(A)[1]);
    tol=1e-12,
    maxiter=100,
)
    AL, C, R, λ = _leftgaugeinitial(A, C; tol=tol)
    numiter = 1
    while norm(C - R) > tol && numiter < maxiter
        λs, Cs, info = eigsolve(
            x -> _lgsolve(A, AL, x),
            R,
            1,
            :LM;
            ishermitian=false,
            tol=tol,
            eager=true,
            maxiter=1,
        )
        AL, C, R, λ = _leftgaugefactor(Cs[1], A)
        numiter += 1
    end
    C = R
    return AL, C, λ
end

function _leftgaugefactor(C, A)
    _, C = leftorth(C)
    CA = _CA(C, A)

    n1 = numout(CA)
    inds = Tuple(1:numind(CA))

    Q, R = leftorth(CA, inds[1:(end - 1)], inds[end:end])
    # Q, R = leftorth(CA, (1, 2, 3), (4,))

    AL = permute(Q, inds[1:n1], inds[(n1 + 1):end])
    # AL = permute(Q, (1, 2), (3, 4))

    λ = norm(R)
    R = R * 1 / λ
    return AL, C, R, λ
end

@doc raw"""

    _CA(C::TensorMap{S,1,1}, A::TensorMap{S,2,2}) where {S<:IndexSpace}

Apply a matrix `C` to the left bond of a MPS tensor `A`

    ┌─┐   ┌───┐
 ◄──│C│◄──┤ A │◄──
    └─┘   └─┬─┘
            ▼

"""
function _CA(C::TensorMap{S,1,1}, A::TensorMap{S,2,2}) where {S<:IndexSpace}
    return @tensor CA[1 2; 3 4] := C[1; a] * A[a 2; 3 4]
end
function _CA(C::TensorMap{S,1,1}, A::TensorMap{S,2,1}) where {S<:IndexSpace}
    return @tensor CA[1 2; 3] := C[1; a] * A[a 2; 3]
end

# Apply a matrix C to the right bond of a MPS Tensor
# assumes left bond is first index, and right bond is last index, with an arb number 
# of physical indicies.
function _AC(A::TensorMap{S}, C::TensorMap{S,1,1}) where {S<:IndexSpace}
    N = numind(A)
    n = numout(A)
    pind = (N, Tuple(2:(N - 1))..., 1)
    CA = _CA(permute(C, (2,), (1,)), permute(A, pind[1:n], pind[(n + 1):N]))
    return permute(CA, pind[1:n], pind[(n + 1):N])
end

function _leftgaugeinitial(A, C; tol=1e-12)
    λs, Ls, info = eigsolve(
        x -> _lgsolveinit(A, x),
        C' * C,
        1,
        :LM;
        ishermitian=false,
        tol=tol,
        maxiter=1,
        eager=true,
    )

    L = Ls[1]
    L = Ls[1] + Ls[1]'
    L = L / tr(L)
    _, Σ, Vt = tsvd(L)
    l = sqrt(Σ) * Vt

    AL, l, R, λ = _leftgaugefactor(l, A)
    return AL, l, R, λ
end

function _lgsolveinit(A::TensorMap{S,2,2}, L::TensorMap{S,1,1}) where {S<:IndexSpace}
    @tensor Lu[d; u] := A[a 2; 3 u] * (A')[3 d; b 2] * L[b; a]
    return Lu
end
function _lgsolveinit(A::TensorMap{S,2,1}, L::TensorMap{S,1,1}) where {S<:IndexSpace}
    @tensor Lu[d; u] := A[a 2; u] * (A')[d; b 2] * L[b; a]
    return Lu
end

function _lgsolve(
    A::TensorMap{S,2,2}, AL::TensorMap{S,2,2}, X::TensorMap{S,1,1}
) where {S<:IndexSpace}
    @tensor Y[d; u] := A[a 2; 3 u] * (AL')[3 d; b 2] * X[b; a]
    return Y
end
function _lgsolve(
    A::TensorMap{S,2,1}, AL::TensorMap{S,2,1}, X::TensorMap{S,1,1}
) where {S<:IndexSpace}
    @tensor Y[d; u] := A[a 2; u] * (AL')[d; b 2] * X[b; a]
    return Y
end

function rightgauge(
    A;
    C=TensorMap(randn, ComplexF64, codomain(A)[1], codomain(A)[1]),
    tol=1e-12,
    maxiter=100,
)
    N = numind(A)
    n = numout(A)
    pind = (N, Tuple(2:(N - 1))..., 1)
    AR, C, λ = leftgauge(permute(A, pind[1:n], pind[(n + 1):N]), permute(C, (2,), (1,)))
    return permute(AR, pind[1:n], pind[(n + 1):N]), permute(C, (2,), (1,)), λ
end

function _vumpsstep(AL, C, AR, M, FL, FR)
    # @tensor AC[-1 -2; -3 -4] := AL[-1 -2; -3 4] * C[4, -4]
    AC = _AC(AL, C)
    μ1s, ACs, info1 = eigsolve(
        x -> applyH1(x, FL, FR, M), AC, 1, :LM; ishermitian=false, eager=true, maxiter=1
    )
    μ0s, Cs, info0 = eigsolve(
        x -> applyH0(x, FL, FR), C, 1, :LM; ishermitian=false, eager=true, maxiter=1
    )

    λ = real(μ1s[1] / μ0s[1])
    AC = ACs[1]
    C = Cs[1]

    N = numind(AC)
    n = numout(AC)

    QAC, RAC = leftorth(AC, Tuple(1:(N - 1)), (N,))
    QC, RC = leftorth(C)
    AL = permute(QAC * QC', Tuple(1:n), Tuple((n + 1):N))
    errL = norm(RAC - RC)

    LAC, QAC = rightorth(AC, (1,), Tuple(2:N))
    LC, QC = rightorth(C)
    AR = permute(QC' * QAC, Tuple(1:n), Tuple((1 + n):N))
    errR = norm(LAC - LC)

    return λ, AL, C, AR, errL, errR
end

function _vumpsupdate(AL, C, AR, M, FL, FR)
    λ, AL, C, AR, errL, errR = _vumpsstep(AL, C, AR, M, FL, FR)
    AL, C = leftgauge(AR, C) # Regauge MPS: not really required
    FL, λL = leftenv(AL, M; FL=FL)
    FR, λR = rightenv(AR, M; FR=FR)

    # Convergence measure: norm of the projection of the residual onto the tangent space
    # @tensor AC[1 2; 3 4] := AL[1 2; 3 a] * C[a; 4]
    AC = _AC(AL, C)
    MAC = applyH1(AC, FL, FR, M)
    MC = applyH0(C, FL, FR)
    # @tensor MAC[1 2; 3 4] -= λL * AL[1 2; 3 a] * MC[a; 4]
    MAC = MAC - _AC(AL, MC) * λL
    err = norm(MAC)

    λs = (λ, λL, λR)
    mps = (AL, C, AR)
    env = (FL, FR)
    info = (err, errL, errR)

    return λs, mps, env, info
end

function vumps(
    A::TensorMap{S}, M::TensorMap{S}; verbose=false, tol=1e-8, maxiter=100
) where {S<:IndexSpace}
    AL, C, _ = leftgauge(A)
    AR, C, _ = rightgauge(AL)

    FL, λL, _ = leftenv(AL, M)
    FR, λR, _ = rightenv(AR, M)

    verbose && @info "Starting point has λ ≈ $λL ≈ $λR"

    λs, mps, env, info = _vumpsupdate(AL, C, AR, M, FL, FR)
    λ = λs[1]
    λL = real(λs[2])
    λR = real(λs[3])
    err = info[1]
    numiter = 1
    verbose && @info "Step $numiter: λ ≈ $(λ) ≈ $(λL) ≈ $(λR), err ≈ $(err)"

    while err > tol && numiter < maxiter
        λs, mps, env, info = _vumpsupdate(mps..., M, env...)
        λ = λs[1]
        λL = real(λs[2])
        λR = real(λs[3])
        err = info[1]
        numiter += 1
        verbose && @info "Step $numiter: λ ≈ $(λ) ≈ $(λL) ≈ $(λR), err ≈ $(err)"
    end
    return λ, AL, C, AR, FL, FR
end
