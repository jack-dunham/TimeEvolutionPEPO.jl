@testset "func: symmetrise" begin
    g = σ3 ⊗ one(σ3) - one(σ3) ⊗ σ3
    vg = (g ⊗ one(g) - one(g) ⊗ g)
    basis = U1Space(-1=>1, 1=>1)
    SG = PEPOKit.SymmetryGroup(vg, basis)
    H = σ1 ⊗ σ1 - σ2 ⊗ σ2
    L = PEPOKit.coherent(H) + PEPOKit.dissipator(2) 
    LL, mapping = PEPOKit.symmetrise(SG,L)
    
    symL = PEPOKit.SymmetricLiouvillian{U₁}(LL, mapping)

    @test PEPOKit.unsymmetrise(symL) ≈ L 
end
