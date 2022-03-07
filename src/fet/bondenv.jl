@doc raw"""
    hsbondenv(p::PEPOTensor)

Returns the Hilbert-Schmidt bond evironment of a tensor
"""
function hsbondenv(vp::VectorPEPOTensor{S}, bond::S) where {S<:IndexSpace}
    hs = hilbertschmidt(vp)
    guess = InfiniteMPS([codomain(hs)[2]], [bond])
    ts,envs,_ = leading_boundary(guess, InfiniteMPO(hs), Vumps(; maxiter=400, verbose=true))
    FL =  permute(leftenv(envs,1,1,convert(MPSMultiline,ts)), Tuple(1), (2,3))
    FR =  rightenv(envs,1,1,convert(MPSMultiline,ts))
    AL = ts.AL[1]
    AR = ts.AR[1]
    C = ts.CR[1]
    @tensor γ[w2 ; e1] := 
        FL[8 ;w1 1]*AL[1 n1 ; 2]*C[2 ; 3] * AR[3 n2; 4] * FR[4 e2; 5] *
        (AR')[5 ; 6 s2 ]*(C')[6 ; 7]*(AL')[7 ; 8 s1]*hs[w1 s1 ; n1 e1]*hs[w2 s2 ; n2 e2] 
    return γ
end


function hilbertschmidt(vp1::TensorMap{S,3,2}, vp2::TensorMap{S,3,2}) where {S<:IndexSpace}
    @tensor vphs[wu wd su sd; nu nd eu ed] := vp1[wu su ph; nu eu] * (vp2')[wd sd; nd ed ph]
    cod = codomain(vphs)
    dom = domain(vphs)
    # codomain 
    wu = unitary(fuse(cod[1] ⊗ cod[2]), cod[1] ⊗ cod[2])
    su = unitary(fuse(cod[3] ⊗ cod[4]), cod[3] ⊗ cod[4])
    # domain
    nu = unitary(dom[1] ⊗ dom[2], fuse(dom[1] ⊗ dom[2]))
    eu = unitary(dom[3] ⊗ dom[4], fuse(dom[3] ⊗ dom[4]))
    @tensor rv[w s; n e] :=
        wu[w; wu wd] *
        su[s; su sd] *
        vphs[wu wd su sd; nu nd eu ed] *
        nu[nu nd; n] *
        eu[eu ed; e]
    return rv
end

function hilbertschmidt(
    vp1::VectorPEPOTensor{S}, vp2::VectorPEPOTensor{S}
) where {S<:EuclideanSpace}
    return hilbertschmidt(vp1.tensor, vp2.tensor)
end

hilbertschmidt(vp::TensorMap{<:IndexSpace,3,2}) = hilbertschmidt(vp,vp)
hilbertschmidt(vp::VectorPEPOTensor) = hilbertschmidt(vp.tensor,vp.tensor)

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
