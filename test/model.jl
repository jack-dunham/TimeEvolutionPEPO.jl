import PEPOKit: lazypower, curly, dynamicalmpo

function xyzlindblad(; Jx=0.5, Jy=1.2, Jz=1.0, Γ=1.0)
    X, Y, Z = PAULI
    M, _ = LADDER

    eqn = Jx * ham(X, 0.2, X) + Jy * ham(Y, 0.4, Y) + Jz * ham(Z, 0.6, Z) + sqrt(Γ) * lin(M)

    return eqn
end

@testset verbose = true "Time evolution MPO" begin
    H = MPO(xyzlindblad())

    cur = (args...) -> curly(H, args...)

    A = H.A
    B = H.B
    C = H.C
    D = H.D
    Id = H[1, 1]

    dt = 1 / 3

    WT = Matrix{Matrix{Matrix{ComplexF64}}}

    WI = WT(undef, 2, 2)
    WII = WT(undef, 2, 2)

    WI[1, 1] = Id + dt * D
    WI[1, 2] = C
    WI[2, 1] = dt * B
    WI[2, 2] = A

    WII[1, 1] = WI[1, 1] + dt^2 / 2 * kron(D, D)
    WII[1, 2] = WI[1, 2] + dt / 2 * cur(:C, :D)
    WII[2, 1] = WI[2, 1] + dt^2 / 2 * cur(:B, :D)
    WII[2, 2] = WI[2, 2] + dt / 2 * (cur(:A, :D) + cur(:B, :C))

    @testset "Curly" begin
        @test cur(:B, :D) == kron(B, D) + kron(D, B)
        @test cur(:A, :B, :B) == kron(A, B, B) + kron(B, A, B) + kron(B, B, A)
        @test cur(:A, :B, :C) ≈
            kron(A, B, C) +
              kron(A, C, B) +
              kron(B, A, C) +
              kron(B, C, A) +
              kron(C, A, B) +
              kron(C, B, A)
    end

    @testset "MPO powers" begin
        for ind in CartesianIndices(H.data)
            i = ind[1]
            j = ind[2]
            @test lazypower(H, (i,), (j,)) == H[i, j]
        end

        @test lazypower(H, (1, 1), (1, 1)) == Id
        @test lazypower(H, (1, 1), (1, 2)) == H.C
        @test lazypower(H, (1, 1), (1, 3)) == H.D

        @test lazypower(H, (1, 1), (2, 1)) == H.C
        @test lazypower(H, (1, 1), (2, 2)) == kron(H.C, H.C)
        @test lazypower(H, (1, 1), (2, 3)) == kron(H.C, H.D)

        @test lazypower(H, (1, 1), (3, 1)) == H.D
        @test lazypower(H, (1, 1), (3, 2)) == kron(H.D, H.C)
        @test lazypower(H, (1, 1), (3, 3)) == kron(H.D, H.D)
    end

    @testset verbose = true "Loopback" begin
        @testset "First order" begin
            M = dynamicalmpo(H, dt, 1; extend=false, compress=false, approx=false)

            @test M ≈ WI
        end

        @testset "Equation (32)" begin
            M = dynamicalmpo(H, dt, 2; extend=false, compress=false, approx=false)

            @test M[1, 1] ≈ Id + dt * D + dt^2 / 2 * kron(D, D)
            @test M[2, 1] ≈ dt / 2 * B + dt^2 / 2 * kron(D, B)
            @test M[3, 1] ≈ dt / 2 * B + dt^2 / 2 * kron(B, D)
            @test M[4, 1] ≈ dt^2 / 2 * kron(B, B)
            @test M[5, 1] ≈ dt^2 / 2 * B
            @test M[6, 1] ≈ dt^2 / 2 * B

            @test M[1, 2] ≈ C
            @test M[2, 2] ≈ A
            @test M[3, 2] ≈ zero.(A)
            @test M[4, 2] ≈ zero.(kron(A, B))
            @test M[5, 2] ≈ zero.(A)
            @test M[6, 2] ≈ zero.(A)

            @test M[1, 3] ≈ C
            @test M[2, 3] ≈ zero.(A)
            @test M[3, 3] ≈ A
            @test M[4, 3] ≈ zero.(kron(A, B))
            @test M[5, 3] ≈ zero.(A)
            @test M[6, 3] ≈ zero.(A)

            @test M[1, 4] ≈ kron(C, C)
            @test M[2, 4] ≈ kron(C, A)
            @test M[3, 4] ≈ kron(A, C)
            @test M[4, 4] ≈ kron(A, A)
            @test M[5, 4] ≈ zero.(kron(A, C))
            @test M[6, 4] ≈ zero.(kron(A, C))

            @test M[1, 5] ≈ kron(C, D)
            @test M[2, 5] ≈ kron(C, B)
            @test M[3, 5] ≈ kron(A, D)
            @test M[4, 5] ≈ kron(A, B)
            @test M[5, 5] ≈ A
            @test M[6, 5] ≈ zero.(A)

            @test M[1, 6] ≈ kron(D, C)
            @test M[2, 6] ≈ kron(D, A)
            @test M[3, 6] ≈ kron(B, C)
            @test M[4, 6] ≈ kron(B, A)
            @test M[5, 6] ≈ zero.(A)
            @test M[6, 6] ≈ A
        end
    end
    @testset verbose = true "Exact compression" begin
        @testset "First order" begin
            M = dynamicalmpo(H, dt, 1; extend=false, compress=true, approx=false)

            @test M ≈ WI
        end

        @testset "Equation (39)" begin
            M = dynamicalmpo(H, dt, 2; extend=false, compress=true, approx=false)

            @test M[1, 1] ≈ Id + dt * D + dt^2 / 2 * kron(D, D)
            @test M[2, 1] ≈ dt * B + dt^2 / 2 * (kron(D, B) + kron(B, D))
            @test M[3, 1] ≈ dt^2 / 2 * kron(B, B)
            @test M[4, 1] ≈ dt^2 / 2 * B

            @test M[1, 2] ≈ C
            @test M[2, 2] ≈ A

            @test M[1, 3] ≈ kron(C, C)
            @test M[2, 3] ≈ kron(C, A) + kron(A, C)
            @test M[3, 3] ≈ kron(A, A)

            @test M[1, 4] ≈ kron(C, D) + kron(D, C)
            @test M[2, 4] ≈ kron(C, B) + kron(A, D) + kron(D, A) + kron(B, C)
            @test M[3, 4] ≈ kron(A, B) + kron(B, A)
            @test M[4, 4] ≈ A
        end
    end
    @testset verbose = true "Extension and exact compression" begin
        @testset "First order" begin
            M = dynamicalmpo(H, dt, 1; extend=true, compress=true, approx=false)

            @test M ≈ WII
        end
        @testset "Equation (43)" begin
            M = dynamicalmpo(H, dt, 2; extend=true, compress=true, approx=false)

            @test M[1, 1] ≈ Id + dt * D + dt^2 / 2 * kron(D, D) + dt^3 / 6 * kron(D, D, D)
            @test M[2, 1] ≈ dt * B + dt^2 / 2 * cur(:D, :B) + dt^3 / 6 * cur(:B, :D, :D)
            @test M[3, 1] ≈ dt^2 / 2 * kron(B, B) + dt^3 / 6 * cur(:B, :B, :D)
            @test M[4, 1] ≈ dt^2 / 2 * B + dt^3 / 6 * (cur(:B, :D) + kron(B, D))
            #

            @test M[1, 2] ≈ C
            @test M[2, 2] ≈ A

            @test M[1, 3] ≈ kron(C, C) + dt / 3 * cur(:C, :C, :D)
            @test M[2, 3] ≈ cur(:A, :C) + dt / 3 * (cur(:A, :C, :D) + cur(:B, :C, :C))
            @test M[3, 3] ≈ kron(A, A) + dt / 3 * (cur(:A, :B, :C) + cur(:A, :A, :D))
            @test M[4, 3] ≈ dt / 3 * (cur(:A, :C) + kron(A, C))
            #
            @test M[1, 4] ≈ cur(:C, :D) + dt / 3 * cur(:C, :D, :D)
            @test M[2, 4] ≈
                cur(:B, :C) + cur(:A, :D) + dt / 3 * (cur(:B, :C, :D) + cur(:A, :D, :D))
            @test M[3, 4] ≈ cur(:A, :B) + dt / 3 * (cur(:A, :B, :D) + cur(:B, :B, :C))
            @test M[4, 4] ≈
                A + dt / 3 * (cur(:A, :D) + kron(A, D) + cur(:B, :C) + kron(B, C))
        end
    end
    @testset verbose = true "Optimal second order" begin
        @testset "Equation (45)" begin
            M = dynamicalmpo(H, dt, 2; extend=true, compress=true, approx=true)

            @test M[1, 1] ≈ Id + dt * D + dt^2 / 2 * kron(D, D) + dt^3 / 6 * kron(D, D, D)
            @test M[2, 1] ≈ dt * B + dt^2 / 2 * cur(:B, :D) + dt^3 / 6 * cur(:B, :D, :D)
            @test M[3, 1] ≈ dt^2 / 2 * kron(B, B) + dt^3 / 6 * cur(:B, :B, :D)
            #

            @test M[1, 2] ≈ C + dt / 2 * cur(:C, :D) + dt^2 / 6 * cur(:C, :D, :D)
            @test M[2, 2] ≈
                A +
                  dt / 2 * (cur(:B, :C) + cur(:A, :D)) +
                  dt^2 / 6 * (cur(:C, :B, :D) + cur(:A, :D, :D))
            @test M[3, 2] ≈
                dt / 2 * cur(:A, :B) + dt^2 / 6 * (cur(:A, :B, :D) + cur(:B, :B, :C))

            @test M[1, 3] ≈ kron(C, C) + dt / 3 * cur(:C, :C, :D)
            @test M[2, 3] ≈ cur(:A, :C) + dt / 3 * (cur(:A, :C, :D) + cur(:B, :C, :C))
            @test M[3, 3] ≈ kron(A, A) + dt / 3 * (cur(:A, :B, :C) + cur(:A, :A, :D))
            #
        end
    end
end
