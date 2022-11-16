function leftgauge_multi(
    A,
    L=[
        TensorMap(randn, eltype(A[1]), codomain(A[1])[2], codomain(A[1])[2]) for
        i in 1:length(A)
    ];
    tol=1e-12,
    maxiter=100,
)
    normalize!(L[end])
    numsites = length(A)
    r = 1:numsites

    ϵ = 2 * tol
    numiter = 1

    AL = similar(A)
    R = similar(L)
    λ = zeros(numsites)

    ϵ = zeros(numsites)
    ϵ[end] = 2 * tol

    T = multi_transfer(A, A)

    λs, Ls, info = KrylovKit.eigsolve(
        y -> PEPOKit.lgsolve(y, T),
        L[end],
        1,
        :LM;
        ishermitian=false,
        tol=tol,
        eager=true,
        maxiter=1,
    )

    R[end] = Ls[1]# + Ls[1]'

    R[end] = R[end] / tr(R[end])
    _, Σ, Vt = tsvd(R[end])
    R[end] = sqrt(Σ) * Vt

    # _,R[end] = leftorth(Ls[1])

    for x in r
        AL[x], _, R[x], λ[x] = PEPOKit.leftgaugefactor(R[mod(x - 1, r)], A[x])
        # ϵ[x] = norm(L[x] - R[x])
    end

    # for x in r
    #     AL[x] = _CA(L[mod(x - 1, r)], A[x])
    #     AL[x], L[x] = leftorth!(AL[x])

    #     λ[x] = norm(L[x])

    #     L[x] = L[x] / λ[x]

    #     # ϵ[x] = norm(Lold - L[x])
    # end
    # λ = 1

    while numiter < maxiter && ϵ[end] > tol
        #update all Ls to aid convergence
        if numiter > 10
            T = multi_transfer(A, AL)

            λs, Ls, info = eigsolve(
                y -> lgsolve(y, T),
                R[end],
                1,
                :LM;
                ishermitian=false,
                tol=tol,
                eager=true,
                maxiter=1,
            )
            # λ = λs[1]
            _, R[end] = leftorth(Ls[1])
        end
        # for x in r
        #     AL[x], L[mod(x-1,r)], R[x], λ[x] = PEPOKit.leftgaugefactor(L[mod(x - 1, r)], A[x])
        #     ϵ[x] = norm(L[x] - R[x])
        # end
        for x in r
            AL[x], _, R[x], λ[x] = PEPOKit.leftgaugefactor(R[mod(x - 1, r)], A[x])
            # ϵ[x] = norm(L[x] - R[x])
        end

        for x in r
            AC = _AC(AL[x], R[x])
            CA = _CA(R[mod(x - 1, r)], A[x])
            normalize!(AC)
            normalize!(CA)
            ϵ[x] = norm(AC - CA)
        end
        # for x in r
        #     AL[x] = _CA(L[mod(x - 1, r)], A[x])
        #     AL[x], L[x] = leftorth!(AL[x])

        #     λ[x] = norm(L[x])

        #     L[x] = L[x] / λ[x]

        #     # ϵ[x] = norm(Lold - L[x])
        # end
        numiter += 1
        # @info "Convergence $ϵ"
    end
    return AL, R, λ
end

function leftgauge(
    A::SizedVector{Nx,<:MPSTensor}, C::SizedVector{Nx,<:BondTensor}; tol=1e-12, maxiter=100
) where {Nx}
    AL, R, λ = leftgauge_multi(totensor.(A), totensor.(C); tol=tol, maxiter=maxiter)
    return MPSTensor.(AL), BondTensor.(R), λ
end

#TODO: check the lbspace of it all 
function leftgauge(A::SizedVector{Nx,M}) where {Nx,M<:MPSTensor}
    χ = lbspace(A[begin])
    C = [BondTensor(randn, eltype(M), χ) for _ in 1:Nx]
    return leftgauge(A, C)
end
function rightgauge(
    A::SizedVector{Nx,<:MPSTensor}, C::SizedVector{Nx,<:BondTensor}; tol=1e-12, maxiter=100
) where {Nx}
    AL, R, λ = rightgauge_multi(totensor.(A), totensor.(C); tol=tol, maxiter=maxiter)
    return MPSTensor.(AL), BondTensor.(R), λ
end

#TODO: check the lbspace of it all
function rightgauge(A::SizedVector{Nx,M}) where {Nx,M<:MPSTensor}
    χ = lbspace(A[begin])
    C = SizedVector{Nx}((BondTensor(randn, eltype(M), χ) for _ in 1:Nx))
    return rightgauge(A, C)
end

function mixedgauge(
    A::SizedVector{Nx,<:MPSTensor}, C::SizedVector{Nx,<:BondTensor}; tol=1e-12, maxiter=100
) where {Nx}
    AL, C, AR = mixedgauge(totensor.(A), totensor.(C))
    return MPSTensor.(AL), BondTensor.(C), MPSTensor.(AR)
end

# function mixedgauge!(AL, AR, A::SizedVector{Nx,<:MPSTensor}, C::SizedVector{Nx,<:BondTensor}; tol=1e-12, maxiter=100
# ) where {Nx}
#     ALt, Ct, ARt = mixedgauge(totensor.(A), totensor.(C))
#     AL .= MPSTensor.(ALt)
#     C .= BondTensor.(Ct)
#     AR .= MPSTensor.(ARt)
#     return AL, C, AR
# end

#TODO: check the lbspace of it all
function mixedgauge(A::SizedVector{Nx,M}) where {Nx,M<:MPSTensor}
    χ = lbspace(A[begin])
    C = SizedVector{Nx}((BondTensor(randn, eltype(M), χ) for _ in 1:Nx))
    return mixedgauge(A, C)
end

function mixedgauge(A::InfMPS)
    AL, C, AR = mixedgauge(A.AL, A.C)
    return InfMPS(AL, C, AR)
end

function rightgauge_multi(
    A,
    R=[
        TensorMap(randn, eltype(A[1]), codomain(A[1])[2], codomain(A[1])[2]) for
        i in 1:length(A)
    ];
    tol=1e-12,
    maxiter=100,
)
    N = numout(A[1])
    M = numind(A[1])

    p = PEPOKit.swap(N, M, NTuple{M,Int}(1:M))

    AR, R, λ = leftgauge_multi(
        broadcast(x -> _permute(x, p), A), broadcast(x -> _permute(x, (2, 1)), R)
    )
    AR = broadcast(x -> _permute(x, invperm(p)), AR)
    R = broadcast(x -> _permute(x, (2, 1)), R)
    return AR, R, λ
end

function left2right(v::Vector{<:TensorMap})
    N = numout(v[1])
    M = numind(v[1])

    p = swap(N, M, NTuple{M,Int}(1:M))

    return broadcast(x -> _permute(x, p), v)
end

function mixedgauge(
    A,
    C=[
        TensorMap(randn, eltype(A[1]), codomain(A[1])[2], codomain(A[1])[2]) for
        i in 1:length(A)
    ];
    diagonal=false,
    tol=1e-12,
    maxiter=100,
)
    kwargs = (tol=tol, maxiter=maxiter)
    AR, _, _ = rightgauge_multi(A, C; kwargs...)
    AL, C, _ = leftgauge_multi(AR, C; kwargs...)

    r = 1:length(A)

    # if diagonal==true
    #     U = similar(C)
    #     V = similar(C)
    #     for x in r
    #         U[x], C[x], V[x] = tsvd(C[x])
    #     end
    #     for x in r
    #         AL[x] = _CA(U[x]',_AC(AL[x], U[x]))
    #         AR[x] = _AC(_CA(V[mod(x-1, r)], AR[x]), V[mod(x-1,r)]')
    #     end
    # end

    return AL, C, AR
end

function mps_transfer(a::TensorMap{S,2,1}, al::TensorMap{S,2,1}) where {S<:IndexSpace}
    @tensoropt T[lu rd; ru ld] := a[lu a; ru] * (al')[rd; ld a]
    return T
end

function compose_transfer!(T1, T2)
    Tn = deepcopy(T1)
    @tensoropt T1[lu rd; ru ld] = Tn[lu md; mu ld] * T2[mu rd; ru md]
    return T1
end

function multi_transfer(A::AbstractArray, AL::AbstractArray)
    T = mps_transfer(A[1], A[1])
    for i in 2:length(A)
        Ti = mps_transfer(A[i], AL[i])
        compose_transfer!(T, Ti)
    end
    # T = T / norm(T)
    return T
end

function lgsolve(x, T)
    @tensoropt y[rd; ru] := x[ld; lu] * T[lu rd; ru ld]
    return y
end
# function mps_transfer(
#     a::TensorMap{S,2,2}, al::TensorMap{S,2,2}, x::TensorMap{S,1,1}
# ) where {S<:IndexSpace}
#     @tensor y[d; u] := a[a 2; b u] * (al')[b d; a 1] * x[1; 2]
#     return y
# end
function isgauged(A::InfMPS{Nx}) where {Nx}
    r = 1:Nx
    for x in r
        if (A.AL[x] × A.C[x] ≈ A.C[mod(x - 1, r)] × A.AR[x] ≈ A.AC[x]) == false
            return false
        else
            return true
        end
    end
end
