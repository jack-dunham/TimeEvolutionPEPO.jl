@testset "struct: VectorPEPOTensor" begin
    S1 = Z2Space(0 => 1, 1 => 1)
    A1 = PEPOTensor(TensorMap(randn, ComplexF64, S1 * S1 * S1, S1 * S1 * S1))
    @test A1 ≈ unvec(vec(A1))
    S2 = U1Space(-1 => 1, 0 => 3, 1 => 1)
    A2 = PEPOTensor(TensorMap(randn, ComplexF64, S2 * S2 * S2, S2 * S2 * S2))
    @test A2 ≈ unvec(vec(A2))
end
