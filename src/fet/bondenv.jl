function hilbertschmidt(vpw::VectorPEPOTensor)
    vp = vpw.tensor
    @tensor vphs[wu wd su sd; nu nd eu ed] := vp[wu su ph; nu eu ] * (vp')[wd sd; nd ed ph]
    cod = codomain(vphs)
    dom = domain(vphs)
    # codomain 
    wu = unitary(fuse(cod[1] ⊗ cod[2]), cod[1] ⊗ cod[2])
    su = unitary(fuse(cod[3] ⊗ cod[4]), cod[3] ⊗ cod[4])
    # domain
    nu = unitary(dom[1] ⊗ dom[2], fuse(dom[1] ⊗ dom[2]))
    eu = unitary(dom[3] ⊗ dom[4], fuse(dom[3] ⊗ dom[4]))
    @tensor rv[ w s ; n e]:= wu[w ; wu wd] * su[s ; su sd] * vphs[wu wd su sd; nu nd eu ed] * nu[nu nd; n] * eu[eu ed ; e]
    return rv
end
