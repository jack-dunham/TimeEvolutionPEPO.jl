import PEPOKit: partialtruncate, TruncationIsometries

randpepo(d, DH, DV=DH) = TensorMap(randn, d * d', DH * DV * DH' * DV')

@testset verbose = true "Truncation" begin
    @testset "Partial truncation" begin
        d = ℂ^2
        s = U1Space(-1 => 1, 1 => 1)

        D1 = ℂ^4
        D2 = ℂ^8

        uD, sD, vD = tsvd(id(D2); trunc=truncspace(D1))
        isoD = TruncationIsometries(uD, sD, vD)

        S1 = U1Space(-1 => 1, 0 => 2, 1 => 1)
        S2 = U1Space(-1 => 2, 0 => 4, 1 => 2)
        uS, sS, vS = tsvd(id(S2); trunc=truncspace(S1))
        isoS = TruncationIsometries(uS, sS, vS)

        tD = randpepo(d, D2)
        tS = randpepo(s, S2)

        @testset "Correct codomain/domain" begin
            for (t, b, iso) in zip((tD, tS), (D1, S1), (isoD, isoS))
                te = partialtruncate(t, iso)
                tw = partialtruncate(iso, t)

                @test codomain(te) == codomain(t)
                @test codomain(tw) == codomain(t)

                for i in 1:4
                    if i == 1
                        @test domain(te, 1) == b
                    elseif i == 3
                        @test domain(tw, 3) == b'
                    else
                        @test domain(te, i) == domain(t, i)
                        @test domain(tw, i) == domain(t, i)
                    end
                end
            end
        end

        @testset "Invariant under identity" begin
            idisoD = TruncationIsometries(id(D2), id(D2), id(D2))
            idisoS = TruncationIsometries(id(S2), id(S2), id(S2))

            @test partialtruncate(tD, idisoD) == tD
            @test partialtruncate(tS, idisoS) == tS
        end

        @testset "Correct east/west truncation" begin
            data1 = rand(2, 2, 3, 3, 3, 3)
            datae = rand(2, 2, 6, 3, 3, 3)
            dataw = rand(2, 2, 3, 3, 6, 3)

            inds = CartesianIndices(data1)

            copyto!(datae, inds, data1, inds)
            copyto!(dataw, inds, data1, inds)

            t1 = TensorMap(data1, ℂ^2 * ℂ^2', ℂ^3 * ℂ^3 * ℂ^3' * ℂ^3')
            te = TensorMap(datae, ℂ^2 * ℂ^2', ℂ^6 * ℂ^3 * ℂ^3' * ℂ^3')
            tw = TensorMap(dataw, ℂ^2 * ℂ^2', ℂ^3 * ℂ^3 * ℂ^6' * ℂ^3')

            u, s, v = tsvd!(id(ℂ^6); trunc=truncspace(ℂ^3))

            @test partialtruncate(te, TruncationIsometries(u, s, v)) ≈ t1
            @test partialtruncate(TruncationIsometries(u, s, v), tw) ≈ t1
        end
    end
end
