randpepo(d, DH, DV=DH) = TensorMap(randn, d * d', DH * DV * DH' * DV')

@testset verbose = true "Isometries" begin
    uc = Lattice()   [randpepo(ℂ^2, ℂ^3) for i in 1:2, j in 1:2]
    @testset "Rank reduction" begin
    end
end
