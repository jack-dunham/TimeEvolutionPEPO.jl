@testset "func: symmetrise" begin
    g = σ3 ⊗ one(σ3) - one(σ3) ⊗ σ3
    vg = (g ⊗ one(g) - one(g) ⊗ g)
    basis = U1Space(-1=>1, 1=>1)
    SG = PEPOKit.SymmetryGroup(vg, basis)
    H = σ1 ⊗ σ1 - σ2 ⊗ σ2
    L = PEPOKit.StandardLiouvillian(PEPOKit.coherent(H) + PEPOKit.dissipator(2), nothing)
    symL = PEPOKit.symmetrise(SG,L)

    @test PEPOKit.unsymmetrise(symL) ≈ L 
end
