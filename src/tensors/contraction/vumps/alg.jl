#=
             12  6
              │  ▲
              ▼  │
            ┌────┴──┐
      1◄────┤       │◄────11
      2◄────┤       │◄────10
            │   M   │
      7────►│       ├────►5
      8────►│       ├────►4
            └─┬─────┘
              │  ▲
              ▼  │
              3  9
better convention is below (not implemented)
              9  6
              │  ▲
              ▼  │
            ┌────┴──┐
      1◄────┤       │◄────7
      2◄────┤       │◄────8
            │   M   │
     10────►│       ├────►4
     11────►│       ├────►5
            └─┬─────┘
              │  ▲
              ▼  │
              3  12

             ┌───────6
             ▼
           ┌───┐
           │   │◄───4
           │   │◄───5
           │ L │
           │   ├───►1
           │   ├───►2
           └─┬─┘
             └──────►3

      3◄─────┐
           ┌─┴─┐
      1◄───┤   │
      2◄───┤   │
           │ R │
      4───►│   │
      5───►│   │
           └───┘
             ▲
      6──────┘

             ┌───┐
         2◄──┤ a │◄──4
             └┬──┘ 
              │ ▲  
              ▼ │
              1 3

              3 1
              │ ▲
              ▼ │
             ┌──┴┐      
         4──►│ a'│──►2 
             └───┘     
TODO: rewrite in above convention
Index convention for position in index array, is codomain counterclockwise from most 
west-northerly (west first) positon, then domain in same order.

                           ┌───┐
     v ^               1◄──┤ a │◄──4
     2 3                   └┬──┘ 
--------------              │ ▲  
     4 1                    ▼ │
     v ^                    2 3

3 ->[ a']-> 2               4 1
                            │ ▲
      3                     ▼ │
      v                    ┌──┴┐      
2 ->[ a']->1           3──►│ a'│──►2 
                           └───┘     
=#

# function applyH1(
#     AC::TensorMap{S,2,2}, FL::TensorMap{S,3,3}, FR::TensorMap{S,3,3}, M::TensorMap{S,6,6}
# ) where {S<:IndexSpace}
#     @tensor HAC[l ld; rd r] :=
#         FL[l 8 7; 2 1 a] *
#         AC[a 12; 6 b] *
#         M[1 2 ld 4 5 6; 7 8 rd 10 11 12] *
#         FR[11 10 b; 5 4 r]
#     return HAC
# end

# yet unused

function applyhac(ac::MPSTensor, fl::LeftEnvTensor, fr::RightEnvTensor, m::MPOTensor)
    act = applyhac(totensor(ac), totensor(fl), totensor(fr), totensor(m))
    return MPSTensor(act)
end
function applyhac(
    ac::AbstractTensorMap{S,2,2},
    fl::AbstractTensorMap{S,3,3},
    fr::AbstractTensorMap{S,3,3},
    m::AbstractTensorMap{S,6,6},
) where {S<:IndexSpace}
    @tensoropt begin
        hac[-1 -2; -3 -4] :=
            ac[9 a; 6 b] *
            fl[10 11 -2; 1 2 a] *
            m[1 2 -1 4 5 6; 7 8 9 10 11 -3] *
            fr[7 8 b; 4 5 -4]
    end
    return hac
end
function applyhac(
    ac::AbstractTensorMap{S,2,1},
    fl::AbstractTensorMap{S,1,2},
    fr::AbstractTensorMap{S,2,1},
    m::AbstractTensorMap{S,2,2},
) where {S<:IndexSpace}
    @tensoropt begin
        hac[-1 -2; -3] := ac[4 a; b] * fl[-2; 1 a] * m[1 -1; 3 4] * fr[3 b; -3]
    end
    return hac
end
# multi-site unit cell (bad convention)
function applyhac(
    ac::AbstractTensorMap{S,3,2},
    fl::AbstractTensorMap{S,2,3},
    fr::AbstractTensorMap{S,3,2},
    m::AbstractTensorMap{S,3,3},
) where {S<:IndexSpace}
    s = codomain(ac)[3]
    R = raise(ComplexF64, s)
    L = lower(ComplexF64, s)

    δtx = TensorMap(delta(ComplexF64, 2, 5), s, s * s * s * s)
    δty = TensorMap(delta(ComplexF64, 2, 5), s * s * s * s, s)

    @tensoropt begin
        hac[s 2 x; 4 y] :=
            fl[2 x1i; w α y1] *
            ac[n α x2; β y2] *
            m[w s x3; e n y3] *
            fr[e β x4i; 4 y4] *
            δtx[x; x1 x2 x3 x4] *
            R[x1; x1i] *
            R[x4; x4i] *
            δty[y1 y2 y3 y4; yi] *
            L[yi; y]
    end
    return hac
end
# multi-site unit cell (good convention)
function applyhac(
    ac::AbstractTensorMap{S,2,3},
    fl::AbstractTensorMap{S,1,4},
    fr::AbstractTensorMap{S,4,1},
    m::AbstractTensorMap{S,2,4},
) where {S<:IndexSpace}
    spx = domain(ac)[2]
    spy = domain(ac)[3]
    r = raise(ComplexF64, spx)
    l = lower(ComplexF64, spy)

    δx = TensorMap(delta(ComplexF64, dim(spx), 5), spx * spx * spx, spx * spx)
    δy = TensorMap(delta(ComplexF64, dim(spy), 5), spy * spy * spy, spy * spy)

    hac = similar(ac)

    @tensoropt begin
        hac[1 2; 3 x y] =
            fl[2; w α xl yl] *
            ac[n α; β xa ya] *
            m[w 1; e n xm ym] *
            fr[e β xr yr; 3] *
            r[xl; x1] *
            δx[x1 xa xm; x2 x] *
            r[x2; xr] *
            δy[yl ya ym; yy yr] *
            l[yy; y]
    end
end

function applyhc(c::BondTensor, fl::LeftEnvTensor, fr::RightEnvTensor)
    ct = applyhc(totensor(c), totensor(fl), totensor(fr))
    return BondTensor(ct)
end
function applyhc(
    c::AbstractTensorMap{S,1,1}, fl::AbstractTensorMap{S,3,3}, fr::AbstractTensorMap{S,3,3}
) where {S<:IndexSpace}
    @tensoropt begin
        hc[a; b] := c[x; y] * fl[1 2 a; 4 5 x] * fr[4 5 y; 1 2 b]
    end
    return hc
end
function applyhc(
    c::AbstractTensorMap{S,1,1}, fl::AbstractTensorMap{S,1,2}, fr::AbstractTensorMap{S,2,1}
) where {S<:IndexSpace}
    @tensoropt hc[a; b] := c[x; y] * fl[a; m x] * fr[m y; b]
    return hc
end
# multi-site unit cell (good convention)
function applyhc(
    c::AbstractTensorMap{S,1,3}, fl::AbstractTensorMap{S,1,4}, fr::AbstractTensorMap{S,4,1}
) where {S<:IndexSpace}
    s = domain(c)[3]
    δ = TensorMap(delta(ComplexF64, dim(s), 4), s * s, s * s)
    r = raise(ComplexF64, s)
    l = lower(ComplexF64, s)
    @tensoropt begin
        hc[a; b x y] :=
            fl[a; m α xl yl] *
            c[α; β xc yc] *
            fr[m β xr yr; b] *
            δ[xc xl; xx x] *
            r[xx; xr] *
            δ[yl yc; yr yy] *
            l[yy; y]
    end
    return hc
end

# function applyH1(
#     AC::TensorMap{S,2,1}, FL::TensorMap{S,1,2}, FR::TensorMap{S,2,1}, M::TensorMap{S,2,2}
# ) where {S<:IndexSpace}
#     @tensor HAC[l s; r] := FL[l; 1 a] * AC[a 4; b] * M[1 s; 3 4] * FR[3 b; r]
#     return HAC
# end
# function applyH0(
#     C::TensorMap{S,1,1}, FL::TensorMap{S,3,3}, FR::TensorMap{S,3,3}
# ) where {S<:IndexSpace}
#     @tensor H0[L; R] := FL[L 2 3; 4 5 a] * C[a; b] * FR[5 4 b; 3 2 R]
#     return H0
# end
# function applyH0(
#     C::TensorMap{S,1,1}, FL::TensorMap{S,1,2}, FR::TensorMap{S,2,1}
# ) where {S<:IndexSpace}
#     @tensor H0[L; R] := FL[L; m a] * C[a; b] * FR[m b; R]
#     return H0
# end

function randFL(AL::TensorMap{S,2,2}, M::TensorMap{S,6,6}) where {S<:IndexSpace}
    spM = hcat(codomain(M)..., domain(M)...)
    spA = hcat(codomain(AL)..., domain(AL)...)
    return TensorMap(
        randn, ComplexF64, spM[10] * spM[11] * spA[2], spM[1] * spM[2] * spA[2]
    )
end
function randFL(AL::TensorMap{S,2,1}, M::TensorMap{S,2,2}) where {S<:IndexSpace}
    spM = hcat(codomain(M)..., domain(M)...)
    spA = hcat(codomain(AL)..., domain(AL)...)
    return TensorMap(randn, ComplexF64, spA[2], spM[1] * spA[2])
end

function randFR(AR::TensorMap{S,2,2}, M::TensorMap{S,6,6}) where {S<:IndexSpace}
    spM = hcat(codomain(M)..., domain(M)...)
    spA = hcat(codomain(AR)..., domain(AR)...)
    return TensorMap(randn, ComplexF64, spM[7] * spM[8] * spA[4], spM[4] * spM[5] * spA[4])
end
function randFR(AR::TensorMap{S,2,1}, M::TensorMap{S,2,2}) where {S<:IndexSpace}
    spM = hcat(codomain(M)..., domain(M)...)
    spA = hcat(codomain(AR)..., domain(AR)...)
    return TensorMap(randn, ComplexF64, spM[3] * spA[3], spA[3])
end

function leftenv(
    AL::TensorMap{S}, M::TensorMap{S}; FL::TensorMap{S}=randFL(AL, M)
) where {S<:IndexSpace}
    λs, FLs, info = eigsolve(
        x -> flsolve(AL, M, x), FL, 1, :LM; eager=true, ishermitian=false
    )
    return FLs[1], λs[1], info
end

function flsolve(
    al::TensorMap{S,2,2}, m::TensorMap{S,6,6}, fl::TensorMap{S,3,3}
) where {S<:IndexSpace}
    @tensoropt begin
        flu[4 5 b2; 7 8 a2] :=
            al[9 a1; 6 a2] *
            fl[10 11 b1; 1 2 a1] *
            m[1 2 3 4 5 6; 7 8 9 10 11 12] *
            (al')[12 b2; 3 b1]
    end
    return flu
end
function flsolve(
    al::TensorMap{S,2,1}, m::TensorMap{S,2,2}, fl::TensorMap{S,1,2}
) where {S<:IndexSpace}
    @tensoropt begin
        flu[-1; -2 -3] := al[4 a; -3] * fl[b; 1 a] * m[1 2; -2 4] * (al)'[-1; 2 b]
    end
    return flu
end
# multi-site trace (bad convention)
function flsolve(
    al::TensorMap{S,3,2}, m::TensorMap{S,3,3}, fl::TensorMap{S,2,3}
) where {S<:IndexSpace}
    s = codomain(al)[3]
    r = raise(ComplexF64, s)
    δh = TensorMap(delta(ComplexF64, 2, 4), s * s, s * s)
    δbv = TensorMap(delta(ComplexF64, 2, 3), s, s * s)
    δrv = TensorMap(delta(ComplexF64, 2, 3), s * s, s)
    @tensoropt begin
        fl[d3 x; e2 u1 y] =
            fl[d1 xl; w1 u1 yl] *
            m[w1 s1 xm1; c n1 ym1] *
            m[c s2 xm2; e2 n2 ym2] *
            al[n1 u1 xa1; u2 ya1] *
            al[n2 u2 xa2; u3 ya2] *
            (al)'[d2 yb1; s1 d1 xb1] *
            (al)'[d3 yb2; s2 d2 xb2] *
            δh[xr1 xv1; xl xa1] *
            r[xr2; xr1] *
            δh[x xv2; xr2 xa2] *
            δbv[xb1; xm1 xv1] *
            δbv[xb2; xm2 xv2] *
            δh[yl ya1; yy yv1] *
            δh[yy ya2; y yv2] *
            δrv[yv1 ym1; yr1] *
            r[yr1; yb1] *
            δrv[yv2 ym2; yr2] *
            r[yr2; yb2]
    end
end
# multi-site trace (good convention)
function uc2_flsolve(
    al::TensorMap{S,2,4}, m::TensorMap{S,6,8}, fl::TensorMap{S,3,5}
) where {S<:IndexSpace}
    s = domain(al)[3]
    r = raise(ComplexF64, s)
    δh = TensorMap(delta(ComplexF64, 2, 4), s * s, s * s)
    δv = TensorMap(delta(ComplexF64, 2, 3), s * s, s)
    @tensoropt begin
        fl[24 25 26; 22 23 21 x y] =
            fl[4 5 6; 2 3 1 xl yl] *
            al[7 1; 9 11 xa1 ya1] *
            m[2 3 8 14 15 9; 12 13 7 4 5 10 xm1 ym1] *
            (al')[10 16 xb1 yb1; 8 6] *
            al[17 11; 19 21 xa2 ya2] *
            m[12 13 18 24 25 19; 22 23 17 14 15 20 xm2 ym2] *
            (al')[20 26 xb2 yb2; 18 16] *
            δh[xl xa1; x1 x2] *
            r[x1; x3] *
            δh[x3 xa2; x x4] *
            δv[x2 xm1; xb1] *
            δv[x4 xm2; xb2] *
            δh[yl ya1; y1 yy] *
            δh[yy ya2; y3 y] *
            δv[y1 ym1; y2] *
            r[y2; yb1] *
            δv[y3 ym2; y4] *
            r[y4; yb2]
    end
    return fl
end

function uc2_flsolve(
    al::TensorMap{S,2,3}, m::TensorMap{S,2,4}, fl::TensorMap{S,1,4}
) where {S<:IndexSpace}
    s = domain(al)[3]
    r = raise(ComplexF64, s)
    println(r.data)
    δ = Tensor([1, 0], s)#temp
    δh = TensorMap(delta(ComplexF64, 2, 4), s * s, s * s)
    # δhotemp = TensorMap(delta(ComplexF64, 2, 5), s * s, s * s *s)
    # @tensoropt δho[1 2; 3 4] := δhotemp[1 2; 3 4 5]*δ[5] 

    # δvotemp = TensorMap(delta(ComplexF64, 2, 4), s * s, s * s )
    # @tensoropt δvo[1 2; 3 ] := δvotemp[1 2; 3 4 ]*δ[4] 

    δv = TensorMap(delta(ComplexF64, 2, 3), s * s, s)
    @tensoropt fln[13; 12 11 x y] :=
        fl[3; 2 1 xl yl] *
        al[4 1; 6 xa1 ya1] *
        m[2 5; 7 4 xm1 ym1] *
        (al')[8 xb1 yb1; 5 3] *
        al[9 6; 11 xa2 ya2] *
        m[7 10; 12 9 xm2 ym2] *
        (al')[13 xb2 yb2; 10 8] *
        δh[xl xa1; x1 x2] *
        r[x1; x3] *
        δh[x3 xa2; x x4] *
        δv[x2 xm1; xb1] *
        δv[x4 xm2; xb2] *
        δh[yl ya1; y1 yy] *
        δh[yy ya2; y3 y] *
        δv[y1 ym1; y2] *
        r[y2; yb1] *
        δv[y3 ym2; y4] *
        r[y4; yb2]
end
# Construct the transfer matrix
function transfer(a::TensorMap{S,2,3}, m::TensorMap{S,2,4}) where {S<:IndexSpace}
    D = eltype(a)
    s = domain(a)[3]
    n = dim(s)
    r = raise(ComplexF64, s)
    δh = TensorMap(delta(ComplexF64, n, 4)::Array{D,4}, s * s, s * s)
    δv = TensorMap(delta(ComplexF64, n, 3)::Array{D,3}, s * s, s)
    @tensoropt begin
        T[1 2 3 xl yl; 6 7 8 xr yr] :=
            m[2 5; 7 4 xm ym] *
            a[4 1; 6 xa ya] *
            (a')[3 xb yb; 5 8] *
            δh[xl xa; xr x1] *
            δv[x1 xm; xb] *
            δh[yl ya; yr y1] *
            δv[y1 ym; y2] *
            r[y2; yb]
    end
    return T
end

# TL(x,y) = ...(A(:,y), A(:,y+1), M(:,y)) mod nx,ny
# TR(x,y) = ...(A(:,y), A(:,y+1), M(:,y)) // circshift x+1
#
# note, for a given y, TL(x,y) = TR(x-1,y)
function transfer_redux(
    au::TensorMap{S,2,1}, ad::TensorMap{S,2,1}, m::TensorMap{S,2,2}
) where {S<:IndexSpace}
    @tensoropt begin
        T[1 2 3; 6 7 8] := m[2 5; 7 4] * au[4 1; 6] * (ad')[3; 5 8]
    end
end
function transfer_multi_redux(T1, T2)
    @tensoropt begin
        T3[-1 -2 -3; -6 -7 -8] := T1[-1 -2 3; 6 7 -8] * T2[6 7 -3; -6 -7 3]
    end
    return T3 / norm(T3)
end
function flsolve_multi_redux(
    fl::TensorMap{S,1,2}, T::TensorMap{S,3,3}
) where {S<:IndexSpace}
    @tensoropt begin
        flu[3; 7 6] := fl[8; 2 1] * T[1 2 3; 6 7 8]
    end
    return flu
end

function leftenv_multi_redux(ALu::Vector, ALd::Vector, M::Vector) end

function transfer_multi(T::TensorMap, N::Int)
    s = domain(T)[5]
    r = raise(ComplexF64, s)
    T0 = deepcopy(T)
    for s in 2:N
        @tensoropt begin
            T0[-1 -2 -3 xl yl; -6 -7 -8 xr yr] =
                T[-1 -2 3 xl yl; 6 7 -8 x1 y] * r[x1; x2] * T0[6 7 -3 x2 y; -6 -7 3 xr yr]
        end
    end
    return T0 / norm(T0)
end

function flsolve_multi(fl::TensorMap{S,1,4}, T::TensorMap{S,5,5}) where {S<:IndexSpace}
    @tensoropt begin
        flu[3; 7 6 x y] := fl[8; 2 1 x1 y1] * T[1 2 3 x1 y1; 6 7 8 x y]
    end
    return flu
end
# can maybe do this with permutations on flsolve but idk how
function frsolve_multi(fr::TensorMap{S,4,1}, T::TensorMap{S,5,5}) where {S<:IndexSpace}
    @tensoropt begin
        fru[2 1 x y; 8] := T[1 2 3 x y; 6 7 8 x1 y1] * fr[7 6 x1 y1; 3]
    end
    return fru
end

function leftenv_multi(al, m, fl)
    N = dim(domain(al)[3])

    T = transfer(al, m)
    T = transfer_multi(T, N)
    λs, FLs, info = eigsolve(
        x -> flsolve_multi(x, T), fl, 1, :LM; eager=true, ishermitian=false
    )
    return FLs[1], λs[1], info
end
function rightenv_multi(ar, m, fr)
    N = dim(domain(ar)[3])

    T = transfer(ar, m)
    T = transfer_multi(T, N)
    λs, FRs, info = eigsolve(
        x -> frsolve_multi(x, T), fr, 1, :LM; eager=true, ishermitian=false
    )
    return FRs[1], λs[1], info
end

# function flsolve(
#     AL::TensorMap{S,2,2}, M::TensorMap{S,6,6}, FL::TensorMap{S,3,3}
# ) where {S<:IndexSpace}
#     @tensor FLu[-1 -2 -3; -4 -5 -6] :=
#         AL[u 12; 6 -6] *
#         FL[d 8 7; 2 1 u] *
#         M[1 2 3 -2 -3 6; 7 8 9 -4 -5 12] *
#         (AL')[9 -1; d 3]
#     return FLu
# end

# function flsolve(
#     AL::TensorMap{S,2,1}, M::TensorMap{S,2,2}, FL::TensorMap{S,1,2}
# ) where {S<:IndexSpace}
#     @tensor FLu[-1; -2 -3] := AL[u 4; -3] * FL[d; 1 u] * M[1 2; -2 4] * (AL')[-1; d 2]
#     return FLu
# end

function rightenv(
    AR::TensorMap{S}, M::TensorMap{S}; FR::TensorMap{S}=randFR(AR, M)
) where {S<:IndexSpace}
    λs, FRs, info = eigsolve(
        x -> frsolve(AR, M, x), FR, 1, :LM; eager=true, ishermitian=false
    )
    return FRs[1], λs[1], info
end

# function frsolve(
#     AR::TensorMap{S,2,2}, M::TensorMap{S,6,6}, FR::TensorMap{S,3,3}
# ) where {S<:IndexSpace}
#     @tensor FRu[-1 -2 -3; -4 -5 -6] :=
#         AR[-3 12; 6 u] *
#         M[-1 -2 3 4 5 6; -4 -5 9 10 11 12] *
#         FR[11 10 u; 5 4 d] *
#         (AR')[9 d; -6 3]
#     return FRu
# end

function frsolve(ar, m, fr)
    @tensoropt begin
        fru[1 2 a1; 10 11 b1] :=
            ar[9 a1; 6 a2] *
            m[1 2 3 4 5 6; 7 8 9 10 11 12] *
            fr[7 8 a2; 4 5 b2] *
            (ar')[12 b2; 3 b1]
    end
    return fru
end
function frsolve(
    ar::TensorMap{S,2,1}, m::TensorMap{S,2,2}, fr::TensorMap{S,2,1}
) where {S<:IndexSpace}
    @tensoropt fru[-1 -2; -3] := ar[4 -2; a] * m[-1 2; 3 4] * (ar)'[b; 2 -3] * fr[3 a; b]
end
# function frsolve(
#     AR::TensorMap{S,2,1}, M::TensorMap{S,2,2}, FR::TensorMap{S,2,1}
# ) where {S<:IndexSpace}
#     @tensor FRu[-1 -2; -3] := AR[-2 4; u] * M[-1 2; 3 4] * FR[3 u; d] * (AR')[d; -3 2]
#     return FRu
# end

@doc raw"""
    leftgauge(A::TensorMap, [C]; kwargs...)

Given a MPS tensor `A`, return a left-canonical MPS tensor `AL`, a gauge transform `C` and
a scalar factor `λ` such that ``λ AL^s C = C A^s``, where an initial guess for `C` can be
provided.

                   ┌────┐   ┌───┐          ┌───┐   ┌───┐
            λ * ◄──┤ AL │◄──┤ C │◄─── = ◄──┤ C │◄──┤ A │◄──
                   └─┬──┘   └─┬─┘          └─┬─┘   └─┬─┘
                     ▼        ▼              ▼       ▼
                     s        s              s       s

"""
function leftgauge(
    A,
    C=TensorMap(randn, ComplexF64, codomain(A)[2], codomain(A)[2]);
    tol=1e-12,
    maxiter=100,
)
    AL, C, R, λ = leftgaugeinitial(A, C; tol=tol)
    numiter = 1
    while norm(C - R) > tol && numiter < maxiter
        λs, Cs, info = eigsolve(
            x -> lgsolve(A, AL, x),
            R,
            1,
            :LM;
            ishermitian=false,
            tol=tol,
            eager=true,
            maxiter=1,
        )
        AL, C, R, λ = leftgaugefactor(Cs[1], A)
        numiter += 1
    end
    C = R
    return AL, C, λ
end

function leftgaugefactor(C, A)
    # _, C = leftorth(C)
    CA = _CA(C, A)

    n1 = numout(CA)
    inds = Tuple(1:numind(CA))

    Q, R = leftorth(CA, inds[1:(end - 1)], inds[end:end])
    # Q, R = leftorth(CA, (1, 2, 3), (4,))

    AL = permute(Q, inds[1:n1], inds[(n1 + 1):end])
    # AL = permute(Q, (1, 2), (3, 4))

    λ = norm(R)
    R = R * 1 / λ
    return AL, C, R, λ
end

@doc raw"""

    _CA(C::TensorMap{S,1,1}, A::TensorMap{S,2,2}) where {S<:IndexSpace}

Apply a matrix `C` to the left bond of a MPS tensor `A`
                                          
    ┌─┐ a ┌───┐     |      ┌─┐ a ┌───┐    
2◄──│C│◄──┤ A │◄──3 |  2◄──│C│◄──┤ A │◄──4
    └─┘   └─┬─┘     |      └─┘   └┬──┘    
            ▼                     │ ▲    
            1                     ▼ │
                                  1 3
"""
function _CA(C::AbstractTensorMap{S,1,1}, A::AbstractTensorMap{S,2,2}) where {S<:IndexSpace}
    return @tensor CA[1 2; 3 4] := C[2; a] * A[1 a; 3 4]
end
function _CA(C::AbstractTensorMap{S,1,1}, A::AbstractTensorMap{S,2,1}) where {S<:IndexSpace}
    return @tensor CA[1 2; 3] := C[2; a] * A[1 a; 3]
end

# Apply a matrix C to the right bond of a MPS Tensor
# assumes left bond is last cod index, and right bond is last dom index
function _AC(A::AbstractTensorMap{S}, C::AbstractTensorMap{S,1,1}) where {S<:IndexSpace}
    N = numout(A)
    M = numind(A)
    p = swap(N, M, M) # e.g. (1,4,3,2) or (1 4; 3 2)
    CA = _CA(_permute(C, (2, 1)), _permute(A, p))
    return _permute(CA, invperm(p))
end

function leftgaugeinitial(A, C; tol=1e-12)
    λs, Ls, info = eigsolve(
        x -> lgsolveinit(A, x),
        C' * C,
        1,
        :LM;
        ishermitian=false,
        tol=tol,
        maxiter=1,
        eager=true,
    )

    L = Ls[1]
    L = Ls[1] + Ls[1]'
    L = L / tr(L)
    _, Σ, Vt = tsvd(L)
    l = sqrt(Σ) * Vt

    AL, l, R, λ = leftgaugefactor(l, A)
    return AL, l, R, λ
end

function lgsolveinit(a::TensorMap{S,2,2}, l::TensorMap{S,1,1}) where {S<:IndexSpace}
    @tensor lu[d; u] := a[a 2; b u] * (a')[b d; a 1] * l[1; 2]
    return lu
end
function lgsolveinit(a::TensorMap{S,2,1}, l::TensorMap{S,1,1}) where {S<:IndexSpace}
    @tensor lu[d; u] := a[a 2; u] * (a')[d; a 1] * l[1; 2]
    return lu
end

function lgsolve(
    a::TensorMap{S,2,2}, al::TensorMap{S,2,2}, x::TensorMap{S,1,1}
) where {S<:IndexSpace}
    @tensor y[d; u] := a[a 2; b u] * (al')[b d; a 1] * x[1; 2]
    return y
end
function lgsolve(
    a::TensorMap{S,2,1}, al::TensorMap{S,2,1}, x::TensorMap{S,1,1}
) where {S<:IndexSpace}
    @tensor y[d; u] := a[a 2; u] * (al')[d; a 1] * x[1; 2]
    return y
end

function rightgauge(
    A,
    C=TensorMap(randn, ComplexF64, codomain(A)[2], codomain(A)[2]);
    tol=1e-12,
    maxiter=100,
)
    N = numout(A)
    M = numind(A)
    p = swap(N, M, M)
    AR, C, λ = leftgauge(_permute(A, p), _permute(C, (2, 1)))
    return _permute(AR, invperm(p)), _permute(C, (2, 1)), λ
end

function leftgauge_check(al::TensorMap{<:IndexSpace,2,1})
    return al' * al
end
function rightgauge_check(ar::TensorMap{<:IndexSpace,2,1})
    arr = permute(ar, (2,), (1, 3))
    return arr * arr'
end

function vumpsstep(AL, C, AR, M, FL, FR)
    # @tensor AC[-1 -2; -3 -4] := AL[-1 -2; -3 4] * C[4, -4]
    AC = _AC(AL, C)
    μ1s, ACs, info1 = eigsolve(
        x -> applyhac(x, FL, FR, M), AC, 1, :LM; ishermitian=false, eager=true, maxiter=1
    )
    μ0s, Cs, info0 = eigsolve(
        x -> applyhc(x, FL, FR), C, 1, :LM; ishermitian=false, eager=true, maxiter=1
    )

    λ = real(μ1s[1] / μ0s[1])
    AC = ACs[1]
    C = Cs[1]

    N = numind(AC)
    n = numout(AC)

    QAC, RAC = leftorth(AC, Tuple(1:(N - 1)), (N,))
    QC, RC = leftorth(C)
    AL = permute(QAC * QC', Tuple(1:n), Tuple((n + 1):N))
    errL = norm(RAC - RC)

    LAC, QAC = rightorth(_permute(AC, swap(1, 2, N)), (1,), Tuple(2:N))
    LC, QC = rightorth(C)
    AR = _permute(permute(QC' * QAC, Tuple(1:n), Tuple((1 + n):N)), swap(1, 2, N))
    errR = norm(LAC - LC)

    return λ, AL, C, AR, errL, errR
end

function permuteleft(t::AbstractTensorMap)
    N = numind(t)
    return permute(t, Tuple(1:(N - 1)), (N,))
end

function invpermuteleft!(tdst::AbstractTensorMap, tsrc::AbstractTensorMap)
    N₁ = numout(tdst)
    N = numind(tdst)
    permute!(tdst, tsrc, Tuple(1:N₁), Tuple((1 + N₁):N))
    return tdst
end

function updateleft!(al::AbstractTensorMap, ac::AbstractTensorMap, c::AbstractTensorMap)
    qac, rac = leftorth(permuteleft(ac))
    qc, rc = leftorth(c)

    invpermuteleft!(al, qac * qc') # QAC * QC' -> AL
    errL = norm(rac - rc)

    return al, errL
end
function updateleft!(AL::MPSTensor, AC::MPSTensor, C::BondTensor)
    _, errL = updateleft!(totensor(AL), totensor(AC), totensor(C))
    return AL, errL
end

function updateleft!(A::InfMPS)
    nx = length(A)
    errL = Vector{Float64}(undef, nx)

    for x in 1:nx
        _, errL[x] = updateleft!(A.AL[x], A.AC[x], A.C[x])
    end

    return A, errL
end
function updateleft!(A::MultilineInfMPS)
    nx, ny = size(A)
    errL = zeros(Float64, nx, ny)
    for y in 1:ny
        _, errL[:, y] = updateleft!(A[y])
    end
    return A, errL
end

# function vumps_leftupdate(AC::TensorMap{<:IndexSpace,2,1},C)
#     U,S,V =  tsvd(_AC(AC,C'))
#     print(S)
#     AL = _AC(U,V)
#     ϵL = norm(AC - _AC(AL, C))
#     return AL, ϵL
# end

function permuteright(t::AbstractTensorMap)
    N₁ = numout(t)
    N = numind(t)
    p = nswap(N₁, 1, Tuple(1:N))
    tp = permute(t, (1,), Tuple(2:N)) # reshape so only one index in codomain
    return _permute(tp, p) # move the bond index to the front 
end

function invpermuteright!(tdst::AbstractTensorMap, tsrc::AbstractTensorMap)
    N₁ = numout(tdst)
    N = numind(tdst)
    p = invperm(nswap(N₁, 1, Tuple(1:N)))
    tp = _permute(tsrc, p) # move the bond index back to pos N₁ (from first)
    permute!(tdst, tp, Tuple(1:N₁), Tuple((N₁ + 1):N)) # move remaining indices back to cod
    return tdst
end

function updateright!(ar::AbstractTensorMap, ac::AbstractTensorMap, c::AbstractTensorMap)
    lac, qac = rightorth(permuteright(ac))
    lc, qc = rightorth(c)

    invpermuteright!(ar, qc' * qac)
    errR = norm(lac - lc)

    return ar, errR
end

function updateright!(AR::MPSTensor, AC::MPSTensor, C::BondTensor)
    _, errR = updateright!(totensor(AR), totensor(AC), totensor(C))
    return AR, errR
end

function updateright!(A::InfMPS)
    nx = length(A)
    errR = Vector{Float64}(undef, nx)

    rx = 1:nx
    for x in rx
        _, errR[x] = updateright!(A.AR[x], A.AC[x], A.C[mod(x - 1, rx)])
    end

    return A, errR
end

function updateright!(A::MultilineInfMPS)
    nx, ny = size(A)
    errR = zeros(Float64, nx, ny)
    for y in 1:ny
        _, errR[:, y] = updateright!(A[y])
    end
    return A, errR
end
# function vumps_rightupdate(AC,C)
#     U,S,V = tsvd(_CA(C', AC), (2,),(1,3))
#     AR = _CA(U, permute(V, (2,1),(3,)))
#     ϵR = norm(AC - _CA(C,AR))
#     return AR, ϵR
# end
function updateboth!(A::MultilineInfMPS)
    _, errL = updateleft!(A)
    _, errR = updateright!(A)
    return A, errL, errR
end

function _AC(AL::TensorMap{S,2,3}, C::TensorMap{S,1,3}) where {S<:IndexSpace}
    spx = domain(C)[2]
    Nx = dim(spx)
    spy = domain(C)[3]
    Ny = dim(spy)

    δx = TensorMap(delta(ComplexF64, Nx, 3), spx * spx, spx)
    δy = TensorMap(delta(ComplexF64, Ny, 3), spy * spy, spy)

    AC = similar(AL)

    @tensoropt AC[1 2; 3 x y] =
        AL[1 2; a xa ya] * C[a; 3 xc yc] * δx[xa xc; x] * δy[ya yc; y]
    return AC
end

function applyhac(z::RecursiveVec, FL::AbstractVector, FR::AbstractVector, M::AbstractVector)
    Ny = length(FL)
    rv = Vector{eltype(z.vecs)}(undef, Ny)
    for y in 1:Ny
        rv[y] = applyhac(z[y], FL[y], FR[y], M[y])
    end
    rv = circshift(rv, 1)
    return RecursiveVec(rv...)
end

function applyhc(z::RecursiveVec, FL::AbstractVector, FR::AbstractVector)
    Ny = length(FL)
    rv = Vector{eltype(z.vecs)}(undef, Ny)
    for y in 1:Ny
        rv[y] = applyhc(z[y], FL[y], FR[y])
    end
    rv = circshift(rv, 1)
    return RecursiveVec(rv...)
end

function vumpsstep!(
    A::MultilineInfMPS{Nx,Ny}, M::InfMPO, FL::LeftEnv, FR::RightEnv
) where {Nx,Ny}
    ry = 1:Nx
    rx = 1:Ny

    λ = 1

    AC = centraltensor(A)
    C = bondtensor(A)
    
    for x in rx
        xp = mod(x + 1, rx)

        # take mps[y], get mps[y].AC, send in mps[y].AC[x]
        Threads.@sync begin
            Threads.@spawn begin
                μ1s, ACs, _ = eigsolve(
                z -> applyhac(z, ($FL[x, :]), ($FR[x, :]), ($M[x, :])),
                    RecursiveVec(($AC[x, :])...),
                    1,
                    :LM,
                )
                 
                # @info "μ1 $(μ1s)"
                for y in ry
                    A[y].AC[x] = ACs[1][y]
                end
            end
            # now set mps[y-1, mod].AC[x] to output of above

            # A[mod(y - 1, ry)].AC[x] = ACs[1]
            Threads.@spawn begin
                μ0s, Cs, _ = eigsolve(
                z -> applyhc(z, ($FL[xp, :]), ($FR[x, :])),
                    RecursiveVec(($C[x, :])...),
                    1,
                    :LM,
                )

                # @info "μ0 $(μ0s[1]))"

                for y in ry
                    A[y].C[x] = Cs[1][y]
                end
            end
        end
        # A[mod(y - 1, ry)].C[x] = Cs[1]
        # λ = real(μ1s[1] / μ0s[1])
    end

    #A now as updated AC, C, need to update AL, AR
    _, errL, errR = updateboth!(A)

    errL, _ = findmax(errL)
    errR, _ = findmax(errR)
    @info "errL = $errL \t errR = $errR"
    return A, errL, errR
end

function vumpsupdate(AL, C, AR, M, FL, FR)
    λ, AL, C, AR, errL, errR = vumpsstep(AL, C, AR, M, FL, FR)
    AL, C = leftgauge(AR, C) # Regauge MPS: not really required
    FL, λL = leftenv(AL, M; FL=FL)
    FR, λR = rightenv(AR, M; FR=FR)

    # Convergence measure: norm of the projection of the residual onto the tangent space
    # @tensor AC[1 2; 3 4] := AL[1 2; 3 a] * C[a; 4]
    AC = _AC(AL, C)
    MAC = applyhac(AC, FL, FR, M)
    MC = applyhc(C, FL, FR)
    # @tensor MAC[1 2; 3 4] -= λL * AL[1 2; 3 a] * MC[a; 4]
    MAC = MAC - _AC(AL, MC) * λL
    err = norm(MAC)

    λs = (λ, λL, λR)
    mps = (AL, C, AR)
    env = (FL, FR)
    info = (err, errL, errR)

    return λs, mps, env, info
end

function vumpsupdate!(A::MultilineInfMPS, M::InfMPO, FL::LeftEnv, FR::RightEnv)
    A, errL, errR = vumpsstep!(A, M, FL, FR) # array form
    # _, _, λL, λR = environments!(A, M, FL, FR)
    
    _, _ = environments_alt!(A,M, FL, FR)

    nx, ny = size(A)
    ry = 1:ny
    rx = 1:nx

    err = zeros(Float64, nx, ny)

    AC = centraltensor(A)

    for x in rx, y in ry
        yp = mod(y+1, ry) 

        vl = leftnull(totensor(AC[x,yp]))

        ac = totensor(applyhac(AC[x,y],FL[x,y],FR[x,y],M[x,y]))
        err[x,y] = norm(vl'*ac)
    end

    # println("err: ", err)
    erri = maximum(err)

    println(erri)

    return A, FL, FR, erri
end

function vumps(
    A::TensorMap{S}, M::TensorMap{S}; verbose=false, tol=1e-8, maxiter=100
) where {S<:IndexSpace}
    AL, C, _ = leftgauge(A)
    AR, C, _ = rightgauge(AL)

    FL, λL, _ = leftenv(AL, M)
    FR, λR, _ = rightenv(AR, M)

    verbose && @info "Starting point has λ ≈ $λL ≈ $λR"

    λs, mps, env, info = vumpsupdate(AL, C, AR, M, FL, FR)
    λ = λs[1]
    λL = real(λs[2])
    λR = real(λs[3])
    err = info[1]
    numiter = 1
    verbose && @info "Step $numiter: λ ≈ $(λ) ≈ $(λL) ≈ $(λR), err ≈ $(err)"

    while err > tol && numiter < maxiter
        λs, mps, env, info = vumpsupdate(mps..., M, env...)
        λ = λs[1]
        λL = real(λs[2])
        λR = real(λs[3])
        err = info[1]
        numiter += 1
        verbose && @info "Step $numiter: λ ≈ $(λ) ≈ $(λL) ≈ $(λR), err ≈ $(err)"
    end

    AL, C, AR = mps
    FL, FR = env
    return λ, AL, C, AR, FL, FR
end

function vumps!(A::MultilineInfMPS, M::InfMPO; verbose=false, tol=1e-8, maxiter=20)

    
    FL, FR = environments_alt(A, M)

    # FL, FR, _, _ = environments(A, M)

    A, _, _, err = vumpsupdate!(A, M, FL, FR)
    numiter = 1
    # verbose && @info "Step $numiter: λ ≈ $(λ) ≈ $(λL) ≈ $(λR), err ≈ $(err)"

    while err > tol && numiter < maxiter
        A, _,_, err = vumpsupdate!(A, M, FL, FR)
        # verbose && @info "Step $numiter: λ ≈ $(λ) ≈ $(λL) ≈ $(λR), err ≈ $(err)"
        numiter += 1
    end
    return A, FL, FR
end

function vumps_test(A::MultilineInfMPS)
    βc = log(1 + sqrt(2)) / 2
    β = 1.5 * βc

    println("β = \t", β)

    aM, aM2 = classicalisingmpo_alt(β)
    tM = TensorMap(aM, ℂ^2 * ℂ^2, ℂ^2 * ℂ^2) #mpo
    tM2 = TensorMap(aM2, ℂ^2 * ℂ^2, ℂ^2 * ℂ^2) #mpo

    M = InfMPO{2,2}(MPOTensor(tM))

    _, FL, FR = vumps!(A, M)

    al = lefttensor(A)
    ar = righttensor(A)
    c = bondtensor(A)
    ac = centraltensor(A)

    z2 = expval(tM, tomatrixt(ac), tomatrixt(FL), tomatrixt(FR))
    magn = expval(tM2, tomatrixt(ac), tomatrixt(FL), tomatrixt(FR))

    return magn./z2
end
function vumps_test_dimer(x, A=MultilineInfMPS{2,2}(rand, ComplexF64, ℂ^4, ℂ^10))
    Tc = 0.65

    T = x * Tc

    println("T = \t", T)

    t = dimer(T)

    M = InfMPO{2,2}(MPOTensor.(t))

    _, FL, FR = vumps!(A, M; verbose=true,maxiter=100)

    al = lefttensor(A)
    ar = righttensor(A)
    c = bondtensor(A)
    ac = centraltensor(A)

    z2 = expval_2(t, normalize.(tomatrixt(ac)), tomatrixt(FL), tomatrixt(FR))
    ds = expval_2(dimer_d(t), normalize.(tomatrixt(ac)), tomatrixt(FL), tomatrixt(FR))

    ps = []
    for l in 1:4
        tp = dimer_p(l, t)
        p = expval_2(tp, tomatrixt(ac), tomatrixt(FL), tomatrixt(FR))
        push!(ps, abs.(p ./ z2))
    end

    return ps, abs(sum(ds ./ z2)/4)
end

function expval(m, ac, fl, fr)
    nx, ny = size(ac)
    rx = 1:nx
    ry = 1:ny

    rv = zeros(ComplexF64, nx, ny)

    for x in rx, y in ry
        yp = mod(y + 1, ry)
        @tensoropt rvi = scalar(
            (fl[x, y])[b1; 1 a1] *
            (ac[x, y])[4 a1; a2] *
            m[1 2; 3 4] *
            (ac[x, yp]')[b2; 2 b1] *
            (fr[x, y])[3 a2; b2],
        )

        rv[x, y] = rvi
    end
    return rv
end
function expval_2(m, ac, fl, fr)
    nx, ny = size(ac)
    rx = 1:nx
    ry = 1:ny

    rv = zeros(ComplexF64, nx, ny)

    for x in rx, y in ry
        xp = mod(x + 1, rx)
        xm = mod(x - 1, rx)
        yp = mod(y + 1, ry)
        @tensoropt rvi = scalar(
            (fl[x, y])[b1; 1 a1] *
            (ac[x, y])[4 a1; a2] *
            (m[x, y])[1 2; 3 4] *
            (ac[x, yp]')[b2; 2 b1] *
            (fr[x, y])[3 a2; b2],
        )

        rv[x, y] = rvi
    end
    return rv
end

# AL[x], C[x], AR[x+1]
function get_aac(al, c, ar)
    nx, ny = size(c)
    rx = 1:nx
    ry = 1:ny

    aac = Matrix{TensorMap}(undef, 2, 2)

    for x in rx, y in ry
        xp = mod(x + 1, rx)
        @tensoropt (aac[x, y])[s1 s2 a; b] :=
            (al[x, y])[s1 a; x] * (c[x, y])[x; y] * (ar[xp, y])[s2 y; b]
    end
    return aac
end

function partition_function(M, AAC, FL, FR)
    nx, ny = size(M)
    rx = 1:nx
    ry = 1:ny

    Z2 = zeros(ComplexF64, nx, ny)

    for x in rx, y in ry
        xp = mod(x + 1, rx)
        yp = mod(y + 1, ry)

        @tensoropt begin
            Z2i = scalar(
                (FL[x, y])[b1; 1 a1] *
                (AAC[x, y])[s1 s2 a1; a2] *
                (M[x, y])[1 t1; 4 s1] *
                (M[xp, y])[4 t2; 3 s2] *
                (AAC[x, yp]')[b2; t1 t2 b1] *
                (FR[xp, y])[3 a2; b2],
            )
        end
        Z2[x, y] = Z2i
    end
    return Z2
end
function observable(M2, AAC, FL, FR, Z2)
    nx, ny = size(FL)
    rx = 1:nx
    ry = 1:ny

    obs = zeros(ComplexF64, nx, ny)

    for x in rx, y in ry
        xp = mod(x + 1, rx)
        yp = mod(y + 1, ry)

        @tensoropt obsi = scalar(
            (FL[x, y])[b1; 1 a1] *
            (AAC[x, y])[s1 s2 a1; a2] *
            M2[1 t1 t2; 3 s2 s1] *
            (AAC[x, yp]')[b2; t1 t2 b1] *
            (FR[xp, y])[3 a2; b2],
        )
        obs[x, y] = obsi / Z2[x, y]
    end
    return obs
end

function statmechmpo(β, h, D)
    M = zeros(D, D, D, D)
    for i in 1:D
        M[i, i, i, i] = 1
    end
    X = zeros(D, D)
    for j in 1:D, i in 1:D
        X[i, j] = exp(-β * h(i, j))
    end
    Xsq = sqrt(X)
    @tensor M1[a, b, c, d] :=
        M[a', b', c', d'] * Xsq[c', c] * Xsq[d', d] * Xsq[a, a'] * Xsq[b, b']

    # For computing energy: M2 is a tensor across 2 nearest neighbor sites in the lattice, whose
    # expectation value in the converged fixed point of the transfer matrix represents the energy
    Y = zeros(D, D)
    for j in 1:D, i in 1:D
        Y[i, j] = h(i, j) * exp(-β * h(i, j))
    end
    @tensor M2[a, b1, b2, c, d2, d1] :=
        M[a', b1', c1, d1'] *
        Xsq[a, a'] *
        Xsq[b1, b1'] *
        Xsq[d1', d1] *
        Y[c1, c2] *
        M[c2, b2', c', d2'] *
        Xsq[b2, b2'] *
        Xsq[d2', d2] *
        Xsq[c', c]

    Z0 = [1.0 0.0; 0.0 -1.0]
    Z = zeros(D, D)
    for j in 1:D, i in 1:D
        Z[i, j] = Z0[i, j] * exp(-β * h(i, j))
    end
    @tensor M3[a, b1, b2, c, d2, d1] :=
        M[a', b1', c1, d1'] *
        Xsq[a, a'] *
        Xsq[b1, b1'] *
        Xsq[d1', d1] *
        Z[c1, c2] *
        M[c2, b2', c', d2'] *
        Xsq[b2, b2'] *
        Xsq[d2', d2] *
        Xsq[c', c]

    return M1, M2, M3
end

function statmechmpo_alt(β, h, D)
    δ1 = zeros(D, D, D, D)
    for i in 1:D
        δ1[i, i, i, i] = 1 # kronecker delta
    end

    δ2 = zeros(D, D, D, D, D)
    for i in 1:D
        δ2[i, i, i, i, i] = 1 # kronecker delta
    end

    X = zeros(D, D)
    for j in 1:D, i in 1:D
        X[i, j] = exp(-β * h(i, j))
    end
    Xsq = sqrt(X)

    Z = [1.0, -1.0]

    @tensor M1[a, b, c, d] :=
        δ1[a', b', c', d'] * Xsq[c', c] * Xsq[d', d] * Xsq[a, a'] * Xsq[b, b']

    @tensor M2[a, b, c, d] :=
        δ2[a', b', c', d', e'] * Xsq[c', c] * Xsq[d', d] * Xsq[a, a'] * Xsq[b, b'] * Z[e']

    return M1, M2
end

function classicalisingmpo_alt(β; J=1.0, h=0.0)
    return statmechmpo_alt(
        β, (s1, s2) -> -J * (-1)^(s1 != s2) - h / 2 * (s1 == 1 + s2 == 1), 2
    )
end

function dimer(T)
    r = 1:4

    A = zeros(4, 4, 4, 4)
    for i in r
        A[i, mod(i - 1, r), mod(i - 2, r), mod(i - 3, r)] = 1
    end
    B = zeros(4, 4, 4, 4)
    for i in r
        B[i, mod(i + 1, r), mod(i + 2, r), mod(i + 3, r)] = 1
    end

    h = exp(1 / (T * 2))

    t = [1 0 0 0; 0 h 1 1; 0 1 1 1; 0 1 1 h]

    q = sqrt(t)

    @tensor Ab[a, b, c, d] := A[a', b', c', d'] * q[a', a] * q[b', b] * q[c', c] * q[d', d]
    @tensor Bb[a, b, c, d] := B[a', b', c', d'] * q[a', a] * q[b', b] * q[c', c] * q[d', d]

    sp = ℂ^4 * ℂ^4

    At = TensorMap(Ab, sp, sp)
    Bt = TensorMap(Bb, sp, sp)

    return [At Bt; Bt At]
end

function dimer_p(l, m)
    δa = zeros(4, 4)
    δa[l, l] = 1
    δ = TensorMap(δa, ℂ^4, ℂ^4)
    mo = similar(m)
    for x in 1:2, y in 1:2
        @tensor (mo[x, y])[a b; c d] := δ[a; a'] * (m[x, y])[a' b; c d]
    end
    return mo
end

function dimer_d(m)
    δa = zeros(4, 4)
    δa[1, 1] = δa[3, 3] = 1
    δa[2, 2] = δa[4, 4] = -1
    δ = TensorMap(δa, ℂ^4, ℂ^4)
    mo = similar(m)
    for x in 1:2, y in 1:2
        @tensor (mo[x, y])[a b; c d] := δ[a; a'] * (m[x, y])[a' b; c d]
    end
    return mo
end
