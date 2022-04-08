@doc raw"""
    hsbondenv(p::PEPOTensor)

Returns the Hilbert-Schmidt bond evironment of a tensor
"""
function hsbondenv(vp::VectorPEPOTensor{S}, bond::S) where {S<:IndexSpace}
    hs, wu, eu = hilbertschmidt(vp)
    guess = InfiniteMPS([codomain(hs)[2]], [bond])
    ts, envs, _ = leading_boundary(
        guess, InfiniteMPO(hs), Vumps(; maxiter=100, verbose=true)
    )
    FL = permute(leftenv(envs, 1, 1, convert(MPSMultiline, ts)), Tuple(1), (2, 3))
    FR = rightenv(envs, 1, 1, convert(MPSMultiline, ts))
    AL = ts.AL[1]
    AR = ts.AR[1]
    C = ts.CR[1]
    # original γ[w2 ; e1]
    # without flipping after split: 
    # 3. e1u |-<-  -<-| w2u 1.
    # 2. e1d |->-  ->-| w2d 4.
    γ = _construct_hsbondenv(hs, wu, eu, ts, envs)
    return γ
end

function _construct_hsbondenv(hs, wu, eu, ts, envs)
    wui = inv(wu)
    eui = inv(eu)
    γ = TensorMap(
        undef,
        ComplexF64,
        codomain(wui)[1] * domain(eui)[2]',
        domain(eui)[1] * codomain(wui)[2]',
    )
    FL = permute(leftenv(envs, 1, 1, convert(MPSMultiline, ts)), Tuple(1), (2, 3))
    FR = rightenv(envs, 1, 1, convert(MPSMultiline, ts))
    AL = ts.AL[1]
    AR = ts.AR[1]
    C = ts.CR[1]
    @time begin
        # use @tensoropt macro, otherwise uses GiB of memory
        @tensoropt γ[w2u e1d; e1u w2d] =
            FL[8; w1 1] *
            AL[1 n1; 2] *
            C[2; 3] *
            AR[3 n2; 4] *
            FR[4 e2; 5] *
            (AR')[5; 6 s2] *
            (C')[6; 7] *
            (AL')[7; 8 s1] *
            hs[w1 s1; n1 e1] *
            eui[e1; e1u e1d] *
            hs[w2 s2; n2 e2] *
            wui[w2u w2d; w2]
    end
    return γ
end

function _construct_hsbondenv(
    M::TensorMap{S,6,6},
    AL::TensorMap{S,2,2},
    C::TensorMap{S,1,1},
    AR::TensorMap{S,2,2},
    FL::TensorMap{S,3,3},
    FR::TensorMap{S,3,3},
) where {S<:IndexSpace}
    @info "Constructing bond environment..."
    @time begin
    @tensoropt γ[-1 -2 -3 -4; -5 -6 -7 -8] :=
        M[l1 l2 l3 -2 -1 l6; l7 l8 l9 -6 -5 l12] *
        M[-4 -3 r3 r4 r5 r6; -8 -7 r9 r10 r11 r12] *
        FL[sw l8 l7; l2 l1 nw] *
        FR[r11 r10 ne; r5 r4 se] *
        AL[nw l12; l6 au] *
        C[au; bu] *
        AR[bu r12; r6 ne] *
        (AL')[l9 ad; sw l3] *
        (C')[bd; ad] *
        (AR')[r9 se; bd r3]
    end
    return γ
end

# PERF: this uses loads of memory. Probably need to come up with a better way of reshaping
# that avoids tensor contractions.
function hilbertschmidt(vp1::TensorMap{S,3,2}, vp2::TensorMap{S,3,2}) where {S<:IndexSpace}
    # @tensor vphs[wu wd su sd; nu nd eu ed] := vp1[wu su ph; nu eu] * (vp2')[wd sd; nd ed ph] old
    @info "Constructing HS tensor..."
    @time begin
        @tensoropt vphs[wu wd su sd; nu nd eu ed] :=
            vp1[wu su ph; nu eu] * (vp2')[nd ed; wd sd ph]
        @info "Size of vphs: $(Base.summarysize(vphs)/1024/1024) MiB"
    end
    cod = codomain(vphs)
    dom = domain(vphs)
    # codomain 
    wu = unitary(fuse(cod[1] ⊗ cod[2]), cod[1] ⊗ cod[2])
    su = unitary(fuse(cod[3] ⊗ cod[4]), cod[3] ⊗ cod[4])
    # domain
    nu = unitary(dom[1] ⊗ dom[2], fuse(dom[1] ⊗ dom[2]))
    eu = unitary(dom[3] ⊗ dom[4], fuse(dom[3] ⊗ dom[4]))
    @info "fusing indices..."
    @time begin
        @tensoropt rv[w s; n e] :=
            wu[w; wu wd] *
            su[s; su sd] *
            vphs[wu wd su sd; nu nd eu ed] *
            nu[nu nd; n] *
            eu[eu ed; e]
    end

    # @time @tensoropt rv1[w su sd; nu nd eu ed]:=vphs[wu wd su sd; nu nd eu ed]*wu[w; wu wd]
    # @time @tensoropt rv2[w s; nu nd eu ed]:=rv1[w su sd; nu nd eu ed]*su[s; su sd]
    # @time @tensoropt rv3[w s; n eu ed]:=rv2[w s; nu nd eu ed]*nu[nu nd ; n]
    # @time @tensoropt rv[w s; n e]:=rv3[w s; n eu ed]*eu[eu ed; e]
    return rv, wu, eu
end

function hilbertschmidt(vp1::TensorMap{S,4,3}, vp2::TensorMap{S,4,3}) where {S<:IndexSpace}
    @tensoropt vhs[1 2 3 4 5 6; 7 8 9 10 11 12] :=
        vp1[1 2 3 s; 10 11 12] * (vp2')[4 5 6; 7 8 9 s]
    return vhs
end

function hsbondenv(vhs::TensorMap{ComplexSpace,6,6}, D::Integer)
    sp = hcat(codomain(vhs)..., domain(vhs)...)
    A = TensorMap(randn, ComplexF64, ℂ^D * sp[6], sp[12] * ℂ^D)
    λ, AL, C, AR, FL, FR = vumps(A, vhs; maxiter=400,verbose=true)
    return _construct_hsbondenv(vhs, AL, C, AR, FL, FR)
end

function hilbertschmidt(
    vp1::VectorPEPOTensor{S}, vp2::VectorPEPOTensor{S}
) where {S<:EuclideanSpace}
    return hilbertschmidt(vp1.tensor, vp2.tensor)
end

hilbertschmidt(vp::TensorMap{<:IndexSpace,3,2}) = hilbertschmidt(vp, vp)
hilbertschmidt(vp::VectorPEPOTensor) = hilbertschmidt(vp.tensor, vp.tensor)

@doc raw"""
    merge(tensor) 

TODO: docstring
"""
function merge(tensor, i1, i2)
    # merge two indices of a tensor 
    cod = codomain(tensor)
    dom = domain(tensor)
    codD = length(cod)
    domD = length(dom)
    if ((i1 <= codD) && (i2 <= codD)) # fuse two indices in codomain
    # do stuff
    elseif ((i1 > codD) && (i2 > codD)) # fuse two indices in domain
    #  TODO: WIP needs to use TO.jl API :(
    # do stuff
    else
        throw(ArgumentError("Not valid tensor indices for merging"))
    end
end
