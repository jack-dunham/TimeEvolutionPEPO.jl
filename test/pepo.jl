@testset "func: bloch_pepo_data()" begin
    @testset "Bad arguments" begin
        @test_throws ArgumentError PEPSKit.bloch_pepo_data(2, 1.0, 2.0, 3.0)
        @test_throws ArgumentError PEPSKit.bloch_pepo_data(2, 2, 0, 0)
        @test_throws ArgumentError PEPSKit.bloch_pepo_data(0, 1.0, 0.0, 0.0)
        @test_throws ArgumentError PEPSKit.bloch_pepo_data(-1, 1.0, 0.0, 0.0)
    end
    @testset "Good arguments" begin
        @test PEPSKit.bloch_pepo_data(2, 1, 0, 0) ≈
            PEPSKit.bloch_pepo_data(2, 1.0, 0.0, 0.0)
    end
end

@testset "struct: PEPOTensor" begin
    a = TensorMap(randn, ℂ^2*ℂ^2*ℂ^2, ℂ^2*ℂ^2*ℂ^2)
    @test PEPOTensor{ComplexSpace}(a).tensor == a
    @test_throws MethodError PEPOTensor{GradedSpace}(a)
end
