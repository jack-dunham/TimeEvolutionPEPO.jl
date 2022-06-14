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
better convention is below (not implemented)
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

             ┌───────6
             ▼
           ┌───┐
           │   │◄───4
           │   │◄───5
           │ L │
           │   ├───►1
           │   ├───►2
           └─┬─┘
             └──────►3

      3◄─────┐
           ┌─┴─┐
      1◄───┤   │
      2◄───┤   │
           │ R │
      4───►│   │
      5───►│   │
           └───┘
             ▲
      6──────┘

             ┌───┐
         2◄──┤ a │◄──4
             └┬──┘ 
              │ ▲  
              ▼ │
              1 3

              3 1
              │ ▲
              ▼ │
             ┌──┴┐      
         4──►│ a'│──►2 
             └───┘     
TODO: rewrite in above convention
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

# function applyH1(
#     AC::TensorMap{S,2,2}, FL::TensorMap{S,3,3}, FR::TensorMap{S,3,3}, M::TensorMap{S,6,6}
# ) where {S<:IndexSpace}
#     @tensor HAC[l ld; rd r] :=
#         FL[l 8 7; 2 1 a] *
#         AC[a 12; 6 b] *
#         M[1 2 ld 4 5 6; 7 8 rd 10 11 12] *
#         FR[11 10 b; 5 4 r]
#     return HAC
# end

function applyhac(
    ac::AbstractTensorMap{S,2,2},
    fl::AbstractTensorMap{S,3,3},
    fr::AbstractTensorMap{S,3,3},
    m::AbstractTensorMap{S,6,6},
) where {S<:IndexSpace}
    @tensoropt begin
        hac[-1 -2; -3 -4] :=
            ac[9 a; 6 b] *
            fl[10 11 -2; 1 2 a] *
            m[1 2 -1 4 5 6 7 8; 9 10 11 -3 13 14 15 16] *
            fr[7 8 b; 4 5 -4]
    end
    return hac
end
function applyhac(
    ac::AbstractTensorMap{S,2,1},
    fl::AbstractTensorMap{S,1,2},
    fr::AbstractTensorMap{S,2,1},
    m::AbstractTensorMap{S,2,2},
) where {S<:IndexSpace}
    @tensoropt begin
        hac[-1 -2; -3] := ac[4 a; b] * fl[-2; 1 a] * m[1 -1; 3 4] * fr[3 b; -3]
    end
    return hac
end

function applyhc(
    c::AbstractTensorMap{S,2,2}, fl::AbstractTensorMap{S,3,3}, fr::AbstractTensorMap{S,3,3}
) where {S<:IndexSpace}
    @tensoropt begin
        hc[a; b] := c[x; y] * fl[1 2 a; 4 5 x] * fr[4 5 y; 1 2 b]
    end
    return hc
end
function applyhc(
    c::AbstractTensorMap{S,1,1}, fl::AbstractTensorMap{S,1,2}, fr::AbstractTensorMap{S,2,1}
) where {S<:IndexSpace}
    @tensoropt hc[a; b] := c[x; y] * fl[a; m x] * fr[m y; b]
    return hc
end
# function applyH1(
#     AC::TensorMap{S,2,1}, FL::TensorMap{S,1,2}, FR::TensorMap{S,2,1}, M::TensorMap{S,2,2}
# ) where {S<:IndexSpace}
#     @tensor HAC[l s; r] := FL[l; 1 a] * AC[a 4; b] * M[1 s; 3 4] * FR[3 b; r]
#     return HAC
# end
# function applyH0(
#     C::TensorMap{S,1,1}, FL::TensorMap{S,3,3}, FR::TensorMap{S,3,3}
# ) where {S<:IndexSpace}
#     @tensor H0[L; R] := FL[L 2 3; 4 5 a] * C[a; b] * FR[5 4 b; 3 2 R]
#     return H0
# end
# function applyH0(
#     C::TensorMap{S,1,1}, FL::TensorMap{S,1,2}, FR::TensorMap{S,2,1}
# ) where {S<:IndexSpace}
#     @tensor H0[L; R] := FL[L; m a] * C[a; b] * FR[m b; R]
#     return H0
# end

function randFL(AL::TensorMap{S,2,2}, M::TensorMap{S,6,6}) where {S<:IndexSpace}
    spM = hcat(codomain(M)..., domain(M)...)
    spA = hcat(codomain(AL)..., domain(AL)...)
    return TensorMap(
        randn, ComplexF64, spM[10] * spM[11] * spA[2], spM[1] * spM[2] * spA[2]
    )
end
function randFL(AL::TensorMap{S,2,1}, M::TensorMap{S,2,2}) where {S<:IndexSpace}
    spM = hcat(codomain(M)..., domain(M)...)
    spA = hcat(codomain(AL)..., domain(AL)...)
    return TensorMap(randn, ComplexF64, spA[2], spM[1] * spA[2])
end

function randFR(AR::TensorMap{S,2,2}, M::TensorMap{S,6,6}) where {S<:IndexSpace}
    spM = hcat(codomain(M)..., domain(M)...)
    spA = hcat(codomain(AR)..., domain(AR)...)
    return TensorMap(randn, ComplexF64, spM[7] * spM[8] * spA[4], spM[4] * spM[5] * spA[4])
end
function randFR(AR::TensorMap{S,2,1}, M::TensorMap{S,2,2}) where {S<:IndexSpace}
    spM = hcat(codomain(M)..., domain(M)...)
    spA = hcat(codomain(AR)..., domain(AR)...)
    return TensorMap(randn, ComplexF64, spM[3] * spA[3], spA[3])
end

function leftenv(
    AL::TensorMap{S}, M::TensorMap{S}; FL::TensorMap{S}=randFL(AL, M)
) where {S<:IndexSpace}
    λs, FLs, info = eigsolve(
        x -> flsolve(AL, M, x), FL, 1, :LM; eager=true, ishermitian=false
    )
    return FLs[1], λs[1], info
end

function flsolve(
    al::TensorMap{S,2,2}, m::TensorMap{S,6,6}, fl::TensorMap{S,3,3}
) where {S<:IndexSpace}
    @tensoropt begin
        flu[4 5 b2; 7 8 a2] :=
            al[9 a1; 6 a2] *
            fl[10 11 b1; 1 2 a1] *
            m[1 2 3 4 5 6; 7 8 9 10 11 12] *
            (al')[12 b2; 3 b1]
    end
    return flu
end
function flsolve(
    al::TensorMap{S,2,1}, m::TensorMap{S,2,2}, fl::TensorMap{S,1,2}
) where {S<:IndexSpace}
    @tensoropt begin
        flu[-1; -2 -3] := al[4 a; -3] * fl[b; 1 a] * m[1 2; -2 4] * (al)'[-1; 2 b]
    end
    return flu
end

# function flsolve(
#     AL::TensorMap{S,2,2}, M::TensorMap{S,6,6}, FL::TensorMap{S,3,3}
# ) where {S<:IndexSpace}
#     @tensor FLu[-1 -2 -3; -4 -5 -6] :=
#         AL[u 12; 6 -6] *
#         FL[d 8 7; 2 1 u] *
#         M[1 2 3 -2 -3 6; 7 8 9 -4 -5 12] *
#         (AL')[9 -1; d 3]
#     return FLu
# end

# function flsolve(
#     AL::TensorMap{S,2,1}, M::TensorMap{S,2,2}, FL::TensorMap{S,1,2}
# ) where {S<:IndexSpace}
#     @tensor FLu[-1; -2 -3] := AL[u 4; -3] * FL[d; 1 u] * M[1 2; -2 4] * (AL')[-1; d 2]
#     return FLu
# end

function rightenv(
    AR::TensorMap{S}, M::TensorMap{S}; FR::TensorMap{S}=randFR(AR, M)
) where {S<:IndexSpace}
    λs, FRs, info = eigsolve(
        x -> frsolve(AR, M, x), FR, 1, :LM; eager=true, ishermitian=false
    )
    return FRs[1], λs[1], info
end

# function frsolve(
#     AR::TensorMap{S,2,2}, M::TensorMap{S,6,6}, FR::TensorMap{S,3,3}
# ) where {S<:IndexSpace}
#     @tensor FRu[-1 -2 -3; -4 -5 -6] :=
#         AR[-3 12; 6 u] *
#         M[-1 -2 3 4 5 6; -4 -5 9 10 11 12] *
#         FR[11 10 u; 5 4 d] *
#         (AR')[9 d; -6 3]
#     return FRu
# end

function frsolve(ar, m, fr)
    @tensoropt begin
        fru[1 2 a1; 10 11 b1] :=
            ar[9 a1; 6 a2] *
            m[1 2 3 4 5 6; 7 8 9 10 11 12] *
            fr[7 8 a2; 4 5 b2] *
            (ar')[12 b2; 3 b1]
    end
    return fru
end
function frsolve(
    ar::TensorMap{S,2,1}, m::TensorMap{S,2,2}, fr::TensorMap{S,2,1}
) where {S<:IndexSpace}
    @tensoropt fru[-1 -2;-3]:=
        ar[4 -2; a]*
        m[-1 2; 3 4]*
        (ar)'[b;2 -3]*
        fr[3 a;b]
end
# function frsolve(
#     AR::TensorMap{S,2,1}, M::TensorMap{S,2,2}, FR::TensorMap{S,2,1}
# ) where {S<:IndexSpace}
#     @tensor FRu[-1 -2; -3] := AR[-2 4; u] * M[-1 2; 3 4] * FR[3 u; d] * (AR')[d; -3 2]
#     return FRu
# end

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
    C=TensorMap(randn, ComplexF64, codomain(A)[2], codomain(A)[2]);
    tol=1e-12,
    maxiter=100,
)
    AL, C, R, λ = leftgaugeinitial(A, C; tol=tol)
    numiter = 1
    while norm(C - R) > tol && numiter < maxiter
        λs, Cs, info = eigsolve(
            x -> lgsolve(A, AL, x),
            R,
            1,
            :LM;
            ishermitian=false,
            tol=tol,
            eager=true,
            maxiter=1,
        )
        AL, C, R, λ = leftgaugefactor(Cs[1], A)
        numiter += 1
    end
    C = R
    return AL, C, λ
end

function leftgaugefactor(C, A)
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
                                          
    ┌─┐ a ┌───┐     |      ┌─┐ a ┌───┐    
2◄──│C│◄──┤ A │◄──3 |  2◄──│C│◄──┤ A │◄──4
    └─┘   └─┬─┘     |      └─┘   └┬──┘    
            ▼                     │ ▲    
            1                     ▼ │
                                  1 3
"""
function _CA(C::AbstractTensorMap{S,1,1}, A::AbstractTensorMap{S,2,2}) where {S<:IndexSpace}
    return @tensor CA[1 2; 3 4] := C[2; a] * A[1 a; 3 4]
end
function _CA(C::AbstractTensorMap{S,1,1}, A::AbstractTensorMap{S,2,1}) where {S<:IndexSpace}
    return @tensor CA[1 2; 3] := C[2; a] * A[1 a; 3]
end

# Apply a matrix C to the right bond of a MPS Tensor
# assumes left bond is last cod index, and right bond is last dom index
function _AC(A::AbstractTensorMap{S}, C::AbstractTensorMap{S,1,1}) where {S<:IndexSpace}
    N = numout(A)
    M = numind(A)
    p = swap(N, M, M) # e.g. (1,4,3,2) or (1 4; 3 2)
    CA = _CA(_permute(C, (2, 1)), _permute(A, p))
    return _permute(CA, invperm(p))
end

function leftgaugeinitial(A, C; tol=1e-12)
    λs, Ls, info = eigsolve(
        x -> lgsolveinit(A, x),
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

    AL, l, R, λ = leftgaugefactor(l, A)
    return AL, l, R, λ
end

function lgsolveinit(a::TensorMap{S,2,2}, l::TensorMap{S,1,1}) where {S<:IndexSpace}
    @tensor lu[d; u] := a[a 2; b u] * (a')[b d; a 1] * l[1; 2]
    return lu
end
function lgsolveinit(a::TensorMap{S,2,1}, l::TensorMap{S,1,1}) where {S<:IndexSpace}
    @tensor lu[d; u] := a[a 2; u] * (a')[d; a 1] * l[1; 2]
    return lu
end

function lgsolve(
    a::TensorMap{S,2,2}, al::TensorMap{S,2,2}, x::TensorMap{S,1,1}
) where {S<:IndexSpace}
    @tensor y[d; u] := a[a 2; b u] * (al')[b d; a 1] * x[1; 2]
    return y
end
function lgsolve(
    a::TensorMap{S,2,1}, al::TensorMap{S,2,1}, x::TensorMap{S,1,1}
) where {S<:IndexSpace}
    @tensor y[d; u] := a[a 2; u] * (al')[d; a 1] * x[1; 2]
    return y
end

function rightgauge(
    A;
    C=TensorMap(randn, ComplexF64, codomain(A)[2], codomain(A)[2]),
    tol=1e-12,
    maxiter=100,
)
    N = numout(A)
    M = numind(A)
    p = swap(N, M, M)
    AR, C, λ = leftgauge(_permute(A, p), _permute(C, (2, 1)))
    return _permute(AR, invperm(p)), _permute(C, (2, 1)), λ
end

function vumpsstep(AL, C, AR, M, FL, FR)
    # @tensor AC[-1 -2; -3 -4] := AL[-1 -2; -3 4] * C[4, -4]
    AC = _AC(AL, C)
    μ1s, ACs, info1 = eigsolve(
        x -> applyhac(x, FL, FR, M), AC, 1, :LM; ishermitian=false, eager=true, maxiter=1
    )
    μ0s, Cs, info0 = eigsolve(
        x -> applyhc(x, FL, FR), C, 1; ishermitian=false, eager=true, maxiter=1
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

    LAC, QAC = rightorth(_permute(AC, swap(1,2,N)), (1,), Tuple(2:N))
    LC, QC = rightorth(C)
    AR = _permute(permute(QC' * QAC, Tuple(1:n), Tuple((1 + n):N)), swap(1,2,N))
    errR = norm(LAC - LC)

    return λ, AL, C, AR, errL, errR
end

function vumpsupdate(AL, C, AR, M, FL, FR)
    λ, AL, C, AR, errL, errR = vumpsstep(AL, C, AR, M, FL, FR)
    AL, C = leftgauge(AR, C) # Regauge MPS: not really required
    FL, λL = leftenv(AL, M; FL=FL)
    FR, λR = rightenv(AR, M; FR=FR)

    # Convergence measure: norm of the projection of the residual onto the tangent space
    # @tensor AC[1 2; 3 4] := AL[1 2; 3 a] * C[a; 4]
    AC = _AC(AL, C)
    MAC = applyhac(AC, FL, FR, M)
    MC = applyhc(C, FL, FR)
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

    λs, mps, env, info = vumpsupdate(AL, C, AR, M, FL, FR)
    λ = λs[1]
    λL = real(λs[2])
    λR = real(λs[3])
    err = info[1]
    numiter = 1
    verbose && @info "Step $numiter: λ ≈ $(λ) ≈ $(λL) ≈ $(λR), err ≈ $(err)"

    while err > tol && numiter < maxiter
        λs, mps, env, info = vumpsupdate(mps..., M, env...)
        λ = λs[1]
        λL = real(λs[2])
        λR = real(λs[3])
        err = info[1]
        numiter += 1
        verbose && @info "Step $numiter: λ ≈ $(λ) ≈ $(λL) ≈ $(λR), err ≈ $(err)"
    end
    (AL, C, AR) = mps
    (FL, FR) = env
    return λ, AL, C, AR, FL, FR
end
