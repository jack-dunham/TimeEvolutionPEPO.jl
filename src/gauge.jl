#TODO: Clean this mess
function leftgauge_multi(
    A::OnUnitCell{MPS,Nx,1,<:AbsTen{2,1}},
    L::OnUnitCell{BondMatrix,Nx,1,<:AbsTen{1,1}};
    tol=1e-12,
    maxiter=100,
) where {Nx}
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
        AL[x], _, R[x], λ[x] = leftgaugefactor(R[x - 1], A[x])
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

        for x in r
            AL[x], _, R[x], λ[x] = leftgaugefactor(R[x - 1], A[x])
        end

        for x in r
            AC = _AC(AL[x], R[x])
            CA = _CA(R[x - 1], A[x])
            normalize!(AC)
            normalize!(CA)
            ϵ[x] = norm(AC - CA)
        end

        numiter += 1
    end
    return AL, R, λ
end

function leftgauge(
    A::UngaugedInfiniteMPS{Nx}, C::OnUnitCell{BondMatrix,Nx}; tol=1e-12, maxiter=100
) where {Nx}
    AL, R, λ = leftgauge_multi(A, C; tol=tol, maxiter=maxiter)
    return AL, R, λ
end

function leftgauge(A::UngaugedInfiniteMPS)
    # Get the eastbond. 
    χs = eastbond.(A)
    # Construct a vector of random bond matrices. 
    data = [TensorMap(rand, numbertype(A), χs[i], χs[i + 1]) for i in length(A)]
    # Apply similar lattice structure. 
    C = OnLattice(BondMatrix(), latticestyle(A), data)
    return leftgauge(A, C)
end

function rightgauge_multi(
    A::OnUnitCell{MPS,Nx,1,<:AbsTen{2,1}},
    R::OnUnitCell{BondMatrix,Nx,1,<:AbsTen{1,1}};
    kwargs...,
) where {Nx}
    p = (3, 2, 1)

    AR, R, λ = leftgauge_multi(
        broadcast(x -> _permute(x, p), A), broadcast(x -> _permute(x, (2, 1)), R); kwargs...
    )
    AR = broadcast(x -> _permute(x, invperm(p)), AR)
    R = broadcast(x -> _permute(x, (2, 1)), R)
    return AR, R, λ
end

function mixedgauge(A::UngaugedInfiniteMPS; tol=1e-12, maxiter=100)
    # Get the eastbond. 
    χs = eastbond.(A)
    # Construct a vector of random bond matrices. 
    data = [TensorMap(rand, numbertype(A), χs[i], χs[i + 1]) for i in 1:length(A)]
    # Apply similar lattice structure. 
    C = OnLattice(BondMatrix(), latticestyle(A), data)
    return mixedgauge(A, C; tol=tol, maxiter=maxiter)
end

# function mixedgauge(A::InfiniteMPS; kwargs...)
#     AL, C, AR = mixedgauge(A.AL, A.C)
#     return InfiniteMPS(AL, C, AR)
# end

function mixedgauge(
    A::OnUnitCell{MPS,Nx,1,<:AbsTen{2,1}},
    C::OnUnitCell{BondMatrix,Nx,1,<:AbsTen{1,1}};
    kwargs...,
) where {Nx}
    AR, _, _ = rightgauge_multi(A, C; kwargs...)
    AL, C, _ = leftgauge_multi(AR, C; kwargs...)

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

function mps_transfer(
    a::AbstractTensorMap{S,2,1}, al::AbstractTensorMap{S,2,1}
) where {S<:IndexSpace}
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
function isgauged(A::InfiniteMPS{Nx}) where {Nx}
    r = 1:Nx
    AL, C, AR, AC = unpack(A)
    for x in r
        if (_AC(AL[x], C[x]) ≈ _CA(C[x - 1], AR[x]) ≈ AC[x]) == false
            return false
        else
            return true
        end
    end
end
