function _randtensors(S)
    AL = TensorMap(randn, ComplexF64, S * S, S * S)
    M = TensorMap(randn, ComplexF64, S * S * S * S * S * S, S * S * S * S * S * S)
    return AL, M
    # return PEPOKit.leftenv(AL, M)
end

function leftgauge(
    A::Array{T,4},
    C::Matrix{T}=Matrix{eltype(A)}(I, size(A, 1), size(A, 1));
    tol=1e-12,
    maxiter=100,
    kwargs...,
) where {T}
    D = size(A)[1]
    d = size(A)[2]
    AL, C, λ = leftgauge(reshape(A, (D, d * d, D)), C; tol=tol, maxiter=maxiter, kwargs...)
    return reshape(AL, (D, d, d, D)), C, λ
end
function leftgauge(
    A::Array{T,3},
    C::Matrix{T}=Matrix{eltype(A)}(I, size(A, 1), size(A, 1));
    tol=1e-12,
    maxiter=100,
    kwargs...,
) where {T}
    λ2s, ρs, info =
        eigsolve(C' * C, 1, :LM; ishermitian=false, tol=tol, maxiter=1, eager=true) do ρ
            @tensor ρE[a, b] := ρ[a', b'] * A[b', s, b] * conj(A[a', s, a])
            return ρE
        end
    ρ = ρs[1] + ρs[1]'
    ρ ./= tr(ρ)
    # C = cholesky!(ρ).U
    # If ρ is not exactly positive definite, cholesky will fail
    F = svd!(ρ)
    C = lmul!(Diagonal(sqrt.(F.S)), F.Vt)
    _, C = qrpos!(C)

    D, d, = size(A)
    Q, R = qrpos!(reshape(C * reshape(A, D, d * D), D * d, D))
    AL = reshape(Q, D, d, D)
    λ = norm(R)
    rmul!(R, 1 / λ)
    numiter = 1
    while norm(C - R) > tol && numiter < maxiter
        # C = R
        λs, Cs, info =
            eigsolve(R, 1, :LM; ishermitian=false, tol=tol, eager=true, maxiter=1) do X
                @tensor Y[a, b] := X[a', b'] * A[b', s, b] * conj(AL[a', s, a])
                return Y
            end
        _, C = qrpos!(Cs[1])
        # The previous lines can speed up the process when C is still very far from the correct
        # gauge transform, it finds an improved value of C by finding the fixed point of a
        # 'mixed' transfer matrix composed of `A` and `AL`, even though `AL` is also still not
        # entirely correct. Therefore, we restrict the number of iterations to be 1 and don't
        # check for convergence
        Q, R = qrpos!(reshape(C * reshape(A, D, d * D), D * d, D))
        AL = reshape(Q, D, d, D)
        λ = norm(R)
        rmul!(R, 1 / λ)
        numiter += 1
    end
    C = R
    return AL, C, λ
end

function rightgauge(
    A::Array{T,4},
    C::Matrix{T}=Matrix{eltype(A)}(I, size(A, 1), size(A, 1));
    tol=1e-12,
    maxiter=100,
    kwargs...,
) where {T}
    D = size(A)[1]
    d = size(A)[2]
    AL, C, λ = rightgauge(reshape(A, (D, d * d, D)), C; tol=tol, maxiter=maxiter, kwargs...)
    return reshape(AL, (D, d, d, D)), C, λ
end
function rightgauge(
    A::Array{T,3},
    C::Matrix{T}=Matrix{eltype(A)}(I, size(A, 1), size(A, 1));
    tol=1e-12,
    maxiter=100,
    kwargs...,
) where {T}
    AL, C, λ = leftgauge(
        permutedims(A, (3, 2, 1)), permutedims(C, (2, 1)); tol=tol, maxiter=maxiter, kwargs
    )
    return permutedims(AL, (3, 2, 1)), permutedims(C, (2, 1)), λ
end

function applyH1(
    AC::Array{Complex{T},3},
    FL::Array{Complex{T},3},
    FR::Array{Complex{T},3},
    M::Union{Array{T,4},Array{Complex{T},4}},
) where {T}
    @tensor HAC[α, s, β] := FL[α, a, α'] * AC[α', s', β'] * M[a, s, b, s'] * FR[β', b, β]
end

function applyH0(C::Matrix{T}, FL::Array{T,3}, FR::Array{T,3}) where {T}
    @tensor HC[α, β] := FL[α, a, α'] * C[α', β'] * FR[β', a, β]
end

function leftenv(
    A::Array{Complex{T},3},
    M::Union{Array{T,4},Array{Complex{T},4}},
    FL::Array{Complex{T},3}=randn(eltype(A), size(A, 1), size(M, 1), size(A, 1));
    eager=true,
    kwargs...,
) where {T}
    λs, FLs, info = eigsolve(FL, 1, :LM; eager=true, ishermitian=false) do FL
        @tensor FL[α, a, β] :=
        FL[α', a', β'] * A[β', s', β] * M[a', s, a, s'] * conj(A)[α', s, α] # try a a'
    end
    return FLs[1], λs[1], info
end

function rightenv(
    A::Array{Complex{T},3},
    M::Union{Array{T,4},Array{Complex{T},4}},
    FR::Array{Complex{T},3}=randn(eltype(A), size(A, 1), size(M, 1), size(A, 1));
    eager=true,
    kwargs...,
) where {T}
    λs, FRs, info = eigsolve(FR, 1, :LM; eager=true, ishermitian=false) do FR
        @tensor FR[α, a, β] :=
            A[α, s', α'] * FR[α', a', β'] * M[a, s, a', s'] * conj(A[β, s, β'])
    end
    return FRs[1], real(λs[1]), info
end

safesign(x::Number) = iszero(x) ? one(x) : sign(x)

qrpos(A) = qrpos!(copy(A))
function qrpos!(A::Matrix{T}) where {T}
    F = qr!(A)
    Q = Matrix(F.Q)
    R = F.R
    phases = safesign.(diag(R))
    rmul!(Q, Diagonal(phases))
    lmul!(Diagonal(conj!(phases)), R)
    return Q, R
end

lqpos(A) = lqpos!(copy(A))
function lqpos!(A::Matrix{T}) where {T}
    F = qr!(Matrix(A'))
    Q = Matrix(Matrix(F.Q)')
    L = Matrix(F.R')
    phases = safesign.(diag(L))
    lmul!(Diagonal(phases), Q)
    rmul!(L, Diagonal(conj!(phases)))
    return L, Q
end

function vumps(
    A::Array{Complex{T},3},
    M::Union{Array{T,4},Array{Complex{T},4}};
    verbose=false,
    tol=1e-8,
    maxvumpsiters=100,
    kwargs...,
) where {T}
    AL, = leftgauge(A)
    AR, C = rightgauge(AL)

    FL, λL = leftenv(AL, M; kwargs...)
    FR, λR = rightenv(AR, M; kwargs...)

    verbose && println("Starting point has λ ≈ $λL ≈ $λR")

    λ, AL, C, AR, = vumpsstep(AL, C, AR, M, FL, FR; tol=tol / 10)
    AL, C, = leftgauge(AR, C; tol=tol / 10, kwargs...) # regauge MPS: not really necessary
    FL, λL = leftenv(AL, M, FL; tol=tol / 10, kwargs...)
    FR, λR = rightenv(AR, M, FR; tol=tol / 10, kwargs...)
    FR ./= @tensor scalar(FL[c, b, a] * C[a, a'] * conj(C[c, c']) * FR[a', b, c']) # normalize FL and FR: not really necessary

    # Convergence measure: norm of the projection of the residual onto the tangent space
    @tensor AC[a, s, b] := AL[a, s, b'] * C[b', b]
    MAC = applyH1(AC, FL, FR, M)
    @tensor MAC[a, s, b] -= AL[a, s, b'] * (conj(AL[a', s', b']) * MAC[a', s', b])
    err = norm(MAC)
    i = 1
    verbose && println("Step $i: λ ≈ $λ ≈ $λL ≈ $λR, err ≈ $err")

    while err > tol && i < maxvumpsiters
        λ, AL, C, AR, = vumpsstep(AL, C, AR, M, FL, FR; tol=tol / 10)
        AL, C, = leftgauge(AR, C; tol=tol / 10, kwargs...) # regauge MPS: not really necessary
        FL, λL = leftenv(AL, M, FL; tol=tol / 10, kwargs...)
        FR, λR = rightenv(AR, M, FR; tol=tol / 10, kwargs...)
        FR ./= @tensor scalar(FL[c, b, a] * C[a, a'] * conj(C[c, c']) * FR[a', b, c']) # normalize FL and FR: not really necessary

        # Convergence measure: norm of the projection of the residual onto the tangent space
        @tensor AC[a, s, b] := AL[a, s, b'] * C[b', b]
        MAC = applyH1(AC, FL, FR, M)
        @tensor MAC[a, s, b] -= AL[a, s, b'] * (conj(AL[a', s', b']) * MAC[a', s', b])
        err = norm(MAC)
        i += 1
        verbose && println("Step $i: λ ≈ $λ ≈ $λL ≈ $λR, err ≈ $err")
    end
    return λ, AL, C, AR, FL, FR
end
function vumpsstep(
    AL::Array{Complex{T},3},
    C::Matrix{Complex{T}},
    AR::Array{Complex{T},3},
    M::Union{Array{T,4},Array{Complex{T},4}},
    FL::Array{Complex{T},3},
    FR::Array{Complex{T},3};
    kwargs...,
) where {T}
    D, d, = size(AL)
    @tensor AC[a, s, b] := AL[a, s, b'] * C[b', b]
    μ1s, ACs, info1 = eigsolve(
        x -> applyH1(x, FL, FR, M),
        AC,
        1,
        :LM;
        ishermitian=false,
        eager=true,
        maxiter=1,
        kwargs...,
    )
    μ0s, Cs, info0 = eigsolve(
        x -> applyH0(x, FL, FR), C, 1; ishermitian=false, eager=true, maxiter=1, kwargs...
    )
    λ = real(μ1s[1] / μ0s[1])
    AC = ACs[1]
    C = Cs[1]

    QAC, RAC = qrpos(reshape(AC, (D * d, D)))
    QC, RC = qrpos(C)
    AL = reshape(QAC * QC', (D, d, D))
    errL = norm(RAC - RC)

    LAC, QAC = lqpos(reshape(AC, (D, d * D)))
    LC, QC = lqpos(C)

    AR = reshape(QC' * QAC, (D, d, D))
    errR = norm(LAC - LC)

    return λ, AL, C, AR, errL, errR
end

function vumpsstep(
    AL::Array{Complex{T},4},
    C::Array{Complex{T},2},
    AR::Array{Complex{T},4},
    M::Array{Complex{T},12},
    FL::Array{Complex{T},6},
    FR::Array{Complex{T},6};
    kwargs...,) where {T}
    
    return vumpsstep(
    )
end

function _getsize(a, order)
    dims = size(a)
    shape = Array{Integer,1}() 
    shapeinv = Array{Integer,1}()
    perm = Array{Integer,1}()
    for group in order
        D = 1
        for index in group
            D = dims[index]*D
            push!(shapeinv, dims[index])
            push!(perm, index)
        end
        push!(shape, D)
    end
    return Tuple(shape), Tuple(shapeinv), Tuple(perm)
end

function _merge(a, order)
    shape, _, perm = _getsize(a, order)
    aa = permutedims(a, perm)
    return reshape(aa, shape)
end

# broken
function _unmerge(a, order)
    _, shapeinv, perm = _getsize(a, order)
    aa = reshape(a, shapeinv)
    return permutedims(aa, invperm(perm))
end

#----------

function _leftenvapply(FL, AL, M)
    @tensor FLnew[-1 -2 -3; -4 -5 -6] :=
        AL[u 12; 6 -6] *
        FL[d 8 7; 2 1 u] *
        M[1 2 3 -2 -3 6; 7 8 9 -4 -5 12] *
        (AL')[9 -1; d 3]
    return FLnew
end

@testset "func: leftenv()" begin
    @testset "test ComplexSpace" begin
        S = ℂ^2
        AL, M = _randtensors(S)
        FL, λ, info = PEPOKit.leftenv(AL, M)
        @test λ * FL ≈ _leftenvapply(FL, AL, M)
        λs = []
        for i in 1:3
            _, λ, _ = PEPOKit.leftenv(AL, M)
            push!(λs, λ)
        end
        @test all(x -> x ≈ λs[1], λs)
    end
    @testset "test GradedSpace" begin
        S = Z2Space(0 => 1, 1 => 1)
        AL, M = _randtensors(S)
        FL, λ, info = PEPOKit.leftenv(AL, M)
        @test λ * FL ≈ _leftenvapply(FL, AL, M)
    end
end

@testset "trace vumps" begin
    S = ℂ 
    tA = TensorMap(randn, ComplexF64, S^4 * S^4, S^4)
    tM = TensorMap(randn, ComplexF64, S^4 * S^4, S^4 * S^4)

    aA = convert(Array, tA)
    aM = convert(Array, tM)

    tAL, tC, tλ = PEPOKit.leftgauge(tA)
    aAL, aC, aλ = leftgauge(aA)

    tAR, tC, tλ = PEPOKit.rightgauge(tAL)
    aAR, aC, aλ = rightgauge(aAL)

    @testset "gauging" begin
        @testset "left" begin
            @test aAL ≈ convert(Array, tAL) # Test AL correct
            @test aC ≈ convert(Array, tC) # Test C is correct
            @test aλ ≈ tλ # Test λ is correct
        end
        @testset "right" begin
            @test aAR ≈ convert(Array, tAR) # Test AL correct
            @test aC ≈ convert(Array, tC) # Test C is correct
            @test aλ ≈ tλ # Test λ is correct
        end
    end
    tFL, tλ, _ = PEPOKit.leftenv(tAL, tM) 
    aFL, aλ, _ = leftenv(aAL, aM)
    @testset "environments" begin
        @test aFL ≈ convert(Array, tFL) 
        @info "tλ = $(tλ) \n aλ = $(aλ)"
        @test tλ ≈ aλ
    end
end

@testset "vumps" begin
    S = ℂ
    tA = TensorMap(randn, ComplexF64, S^4 * S^2, S^2 * S^4)
    tM = TensorMap(
        randn,
        ComplexF64,
        S^2 * S^2 * S^2 * S^2 * S^2 * S^2,
        S^2 * S^2 * S^2 * S^2 * S^2 * S^2,
    )
    aA = convert(Array, tA)
    aM = convert(Array, tM)

    tAL, tC, tλ = PEPOKit.leftgauge(tA)
    aAL, aC, aλ = leftgauge(aA)

    tAR, tC, tλ = PEPOKit.rightgauge(tAL)
    aAR, aC, aλ = rightgauge(aAL)
    @testset "gauging" begin
        @testset "left" begin
            @test aAL ≈ convert(Array, tAL) # Test AL correct
            @test aC ≈ convert(Array, tC) # Test C is correct
            @test aλ ≈ tλ # Test λ is correct
        end
        @testset "right" begin
            @test aAR ≈ convert(Array, tAR) # Test AL correct
            @test aC ≈ convert(Array, tC) # Test C is correct
            @test aλ ≈ tλ # Test λ is correct
        end
    end
    order_A = ((1,),(2,3),(4,))
    order_FL = ((1,), (2,3,4,5), (6,))
    order_M = ((1,2,7,8),(3,9),(4,5,10,11), (6,12)) 
    tFL, tλ, _ = PEPOKit.leftenv(tAL, tM) # this is (2,2) (6,6)
    aFL, aλ, _ = leftenv(_merge(aAL,order_A), _merge(aM, order_M)) # this is rank 3, rank 4
    @testset "environments" begin 
        tFL_arr = _merge(convert(Array, tFL), order_FL)
        @test real(tλ) ≈ aλ
        # @test aFL ≈ tFL_arr
    end
end
