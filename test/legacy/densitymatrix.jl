@testsetup module DensityMatrix
using Reexport
@reexport using TimeEvolutionPEPO
@reexport using TensorKit

export DMZ, DMX

function getdm()
    obs = VUMPS(; bonddim=4)

    ρz = PEPO((0, 0, -1), 2, 2)
    ρx = PEPO((1, 0, 0), 2, 2)

    return map((ρz, ρx)) do ρ
        return DensityMatrix(ρ, obs)
    end
end

const DMZ, DMX = getdm()
end # DensityMatrix

@testitem "Density Matrix" setup = [DensityMatrix] begin
    X, Y, Z = PAULI

    id = one(Z)

    @testset "`expval`" begin
        rdmz = partialtrace(DMZ, (1, 1))
        rdmx = partialtrace(DMX, (1, 1))

        @test expval(rdmz, X) ≈ 0
        @test expval(rdmz, Z) ≈ -1
        @test expval(rdmx, X) ≈ 1
        @test expval(rdmx, Z) ≈ 0

        let A = rand(2, 2), s = domain(rdmz)
            @test expval(rdmz, A) ≈ tr(TensorMap(A, s, s) * rdmz)
        end

        let rdm12 = partialtrace(DMZ, (1, 1), (2, 1))
            @test expval(rdm12, Z, Z) ≈ 1
            @test expval(rdm12, Z, id) ≈ -1
            @test expval(rdm12, id, Z) ≈ -1
            @test expval(rdm12, kron(Z, Z)) ≈ 1
            @test expval(rdm12, Z) ≈ -1
            let A = rand(2, 2)
                @test expval(rdm12, A) ≈ expval(rdm12, kron(A, one(A)))
                @test expval(rdm12, A) ≈ expval(rdm12, A, one(A))
            end
        end

        @test correlator(DMZ, Z, (1, 1), Z, (2, 1)) ≈ 1
    end
end
