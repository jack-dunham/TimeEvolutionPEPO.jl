@testsetup module Imports
using Reexport
@reexport using TimeEvolutionPEPO
@reexport import TimeEvolutionPEPO: combineaxial, TrotterLayer, Combine
@reexport import TimeEvolutionPEPO: GlobalTEBD, GlobalClusterMPO, changeaxis, terms
@reexport import TimeEvolutionPEPO: trotterlayers, seperateaxis, isexpandable
@reexport import TimeEvolutionPEPO: PowOp, LocalOp, GaussOp2d, oneterms, distributelocal
end

@testitem "Trotterisation" setup = [Imports] begin
    X, Y, Z = PAULI

    @testset "`TrotterLayer`" begin
        op = LocalOp(X)
        op2 = LocalOp(X, X)

        h = TrotterLayer(op)
        hx = TrotterLayer(op, 1)

        H0 = zero(Hamiltonian)

        @test changeaxis(hx, 2).axis == 2
    end

    @testset "`combineaxial`" begin
        let out = combineaxial((hx, hy))
            @test length(out) == 1

            el = only(out)

            @test el isa Combine
            @test el.model[1] === hx
            @test el.model[2] === hy
        end

        let steps = [hx, hy, h], out = combineaxial(steps)
            @test length(out) == 2

            el1 = first(out)
            el2 = last(out)

            @test el1 isa Combine
            @test el2 isa Combine

            @test el1.model[1] === steps[1]
            @test el1.model[2] === steps[2]

            @test el2.model[1].axis == 1
            @test el2.model[2].axis == 2
        end

        let steps = [hx, hy, hx], out = combineaxial(steps)
            @test length(out) == 2

            el1 = first(out)
            el2 = last(out)

            @test el1 isa Combine
            @test el2 isa Combine

            @test el1.model[1] === steps[1]
            @test el1.model[2] === steps[2]

            @test el2.model[1] === steps[3]
        end
        let steps = [hx, hx, hy], out = combineaxial(steps)
            @test length(out) == 2

            el1 = first(out)
            el2 = last(out)

            @test el1 isa Combine
            @test el2 isa Combine

            @test el1.model[1] === steps[1]

            @test el2.model[1] === steps[2]
            @test el2.model[2] === steps[3]
        end

        @test length(combineaxial([hx, hy, hy, hx])) == 2
        @test length(combineaxial([h, h, h, h, hx, hy])) == 5
    end

    @testset "`seperateaxis`" begin
        let out = seperateaxis([h, h])
            @test length(out) == 4
            @test eltype(out) <: TrotterLayer

            @test out[1].axis == 1
            @test out[2].axis == 1

            @test out[3].axis == 2
            @test out[4].axis == 2
        end

        let out = seperateaxis([h, hx, h])
            @test length(out) == 5
            @test eltype(out) <: TrotterLayer

            @test out[1].axis == 1
            @test out[2].axis == 2

            @test out[3].axis == 1

            @test out[4].axis == 1
            @test out[5].axis == 2
        end
    end

    @testset "`powop2d`" begin
        @test only(PowOp2d(X, X, 1.0)).α == 1.0
        @test only(PowOp2d(X, X, 0.5)).α == 0.5

        @test only(PowOp2d(X, X, 1.0)).a == 1.0
        @test only(2 * PowOp2d(X, X, 0.5)).a == 2

        pf = n -> PowOp2d(X, X, 1.0; n=n, expand=false)
        p0f = n -> PowOp2d(X, X, 1.0; n=n, expand=true)

        p = pf(10)
        p0 = p0f(10)

        @test p isa Hamiltonian
        @test terms(powop2d(X, X, 1.0)) isa Tuple
        @test length(p) == 1
        @test length(p0) == 1
        @test only(p) isa PowOp{GaussOp2d}

        @test isexpandable(first(p)) == false
        @test isexpandable(first(p0)) == true

        for n in [1, 5, 10]
            @test length(first(pf(n))) == n
            @test length(first(p0f(n))) == n

            @test length(trotterlayers(H0, [pf(n)])) == n
            @test length(trotterlayers(H0, [p0f(n)])) == 1
        end
    end
end

@testitem "Model Interface" setup = [Imports] begin
    @testset "`Hamiltonian`" begin
        X, Y, Z = PAULI

        @test (LocalOp(Z) + LocalOp(Z)) isa Hamiltonian
        @test (LocalOp(Z) + LocalOp(X, X)) isa Hamiltonian

        @test length(LocalOp(Z) + LocalOp(Z)) == 0
        @test length(LocalOp(Z) + LocalOp(Y) + LocalOp(Z, Z)) == 1
        @test length(LocalOp(X, X) + LocalOp(Y, Y)) == 2
        @test length(LocalOp(Z, Z) + LocalOp(Z, Z)) == 2
        @test length(LocalOp(X, X) + LocalOp(Y, Y) + LocalOp(Z, Z) + LocalOp(X)) == 3

        H = 0.5 * LocalOp(Z) + 3.0 * LocalOp(X, X) - 0.5 * LocalOp(Y, Y)

        @test length(H + LocalOp(X, X)) == length(H) + 1
        @test length(LocalOp(X, X) + H) == length(H) + 1
        @test length(LocalOp(X) + H) == length(H)

        let (H0, state) = iterate(H)
            @test H0 == 3.0 * LocalOp(X, X)
            @test state == 2
        end

        @test isexpandable(H) == false
        @test oneterms(H) == [0.5 * LocalOp(Z)]
        @test isaxial(H) == true

        @test zero(H) + H == H
        @test H + zero(H) == H

        @test (zero(H) + LocalOp(X)) isa Hamiltonian
        @test (zero(H) + LocalOp(X, X)) isa Hamiltonian

        @test (zero(H) + LocalOp(X, X)) == Hamiltonian(LocalOp(X, X))

        let Hi = 2 * H
            @test length(oneterms(Hi)) == 1
            @test length(oneterms(H)) == 1

            for (hi, h) in (Hi, H)
                @test hi == 2 * h
            end

            @test only(oneterms(Hi)) == 2 * only(oneterms(H))
        end
    end

    @testset "`Liouvillian`" begin
        X, Y, Z = PAULI

        H = 0.5 * LocalOp(X) + 3.0 * LocalOp(Z, Z)
        D = Dissipator(X, Y)

        @test first((1 / 4) * D) == 1 / 2 * X
        @test last(D * (1 / 9)) == 1 / 3 * Y

        L = H + D

        @test L isa Liouvillian

        @test L.hamiltonian === H
        @test X ∈ dissipator(L)
        @test Y ∈ dissipator(L)

        @test length(dissipator(L)) == 2
    end

    @testset "`Model`" begin
        X, Y, Z = PAULI

        H = 0.5 * LocalOp(Z) + 3.0 * LocalOp(X, X) - 0.5 * LocalOp(Y, Y)

        D = Dissipator(X, Y)

        @test length(Model(H)) == 1
        @test length(Model(H, H, H)) == 3
        @test length(Model([H, H])) == 2

        @test Model(H) isa Model{Hamiltonian}
        @test Model(H + D) isa Model{Liouvillian}
        @test length(Model(H + D)) == 1
        @test length(Model(H + D, H + D)) == 2

        @test_throws MethodError push!(Model(H), D)
        @test length(push!(Model(H + D), D)) == 1

        @test length(push!(Model(H), H)) == 2
        @test length(push!(Model(H), LocalOp(X))) == 1
        @test length(push!(Model(H), zero(H) + LocalOp(X))) == 2
        @test length(push!(Model(H), LocalOp(X, X))) == 2
        @test length(push!(Model(H), H)) == 2

        @test length(push!(Model(H), PowOp(X, X, 1.0; n=10, expand=true))) == 11
        @test length(push!(Model(H), PowOp(X, X, 1.0; n=10, expand=false))) == 2

        @test length(append!(Model(H), Model(H, H))) == 3
        @test length(append!(Model(H, H), Model(H, H))) == 4
    end
end
