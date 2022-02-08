@testset "func: compose()" begin
    t0 = TensorMap(randnormal, ℂ^2, ℂ^2)
    t1 = TensorMap(randnormal, ℂ^2, ℂ^2)
    t2 = TensorMap(randnormal, ℂ^2, ℂ^2)
    t1_mat = t1.data
    t2_mat = t2.data
    t3 = TensorMap(randnormal, ℂ^2, ℂ^3)
    t4 = TensorMap(randnormal, ℂ^3, ℂ^2)
    t5 = TensorMap(randnormal, ℂ^2 * ℂ^2, ℂ^2 * ℂ^2)
    t3_mat = t3.data
    t4_mat = t4.data
    @test PEPOKit.compose(t1, t2) ≈ TensorMap(t1_mat * t2_mat, ℂ^2, ℂ^2)
    @test PEPOKit.compose(t2, t1) ≈ TensorMap(t2_mat * t1_mat, ℂ^2, ℂ^2)
    @test PEPOKit.compose(t3, t4) ≈ TensorMap(t3_mat * t4_mat, ℂ^2, ℂ^2)
    @test_throws SpaceMismatch PEPOKit.compose(t3, t1)
    @test_throws DimensionMismatch PEPOKit.compose(t1, t5)
    @testset "Associativity" begin
        @test PEPOKit.compose(PEPSKit.compose(t0, t1), t2) ≈
            PEPOKit.compose(t0, PEPSKit.compose(t1, t2))
    end
    @testset "Distributivity" begin
        @test PEPOKit.compose(t0, t1 + t2) ≈
            PEPOKit.compose(t0, t1) + PEPSKit.compose(t0, t2)
        @test PEPOKit.compose(t0 + t1, t2) ≈
            PEPOKit.compose(t0, t2) + PEPSKit.compose(t1, t2)
    end
end

@testset "func: vec()" begin
    A = TensorMap(randn, ComplexF64, ℂ^2 * ℂ^2 * ℂ^2, ℂ^2 * ℂ^2 * ℂ^2)
    Aarr = convert(Array, A)
    vec_Aarr = permutedims(
        reshape(permutedims(Aarr, (1, 2, 4, 5, 3, 6)), (2, 2, 2, 2, 4)), (1, 2, 5, 3, 4)
    )
    @test convert(Array, PEPOKit._vec(A)[1]) ≈ vec_Aarr
end
