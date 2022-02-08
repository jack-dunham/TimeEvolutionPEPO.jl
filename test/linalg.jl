@testset "func: vec()" begin
    A = TensorMap(randn, ComplexF64, ℂ^2 * ℂ^2 * ℂ^2, ℂ^2 * ℂ^2 * ℂ^2)
    Aarr = convert(Array, A)
    vec_Aarr = permutedims(
        reshape(permutedims(Aarr, (1, 2, 4, 5, 3, 6)), (2, 2, 2, 2, 4)), (1, 2, 5, 3, 4)
    )
    @test convert(Array, PEPOKit._vec(A)[1]) ≈ vec_Aarr
end
