@testset "struct: VectorPEPOTensor" begin
    S1 = Z2Space(0 => 1, 1 => 1)
    A1 = PEPOTensor(TensorMap(randn, ComplexF64, S1 * S1 * S1, S1 * S1 * S1))
    @test A1 ≈ PEPOKit.unvec(vec(A1))
    S2 = U1Space(-1 => 1, 0 => 3, 1 => 1)
    A2 = PEPOTensor(TensorMap(randn, ComplexF64, S2 * S2 * S2, S2 * S2 * S2))
    @test A2 ≈ PEPOKit.unvec(vec(A2))
    S3 = U1Space(-1 => 1, 1 => 1)
    A3 = PEPOTensor(TensorMap(randn, ComplexF64, S3 * S3 * S2, S3 * S3 * S2))
    @test A3 ≈ PEPOKit.unvec(vec(A3))
end
