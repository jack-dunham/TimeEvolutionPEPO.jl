using TensorKit
using BenchmarkTools


function fuse1(t::AbstractTensorMap{<:IndexSpace,3,2})
    cod = codomain(t)
    fuser = isomorphism(fuse(cod[1]*cod[2]), cod[1]*cod[2])
    return @tensor fused[1 2; 3 4]:= fuser[1; x y]*t[x y 2; 3 4]
end

function fuse2(t::AbstractTensorMap{<:IndexSpace,3,2})
    cod = codomain(t)
    dom = domain(t)
    return TensorMap(t.data, fuse(cod[1]*cod[2]) * cod[3], dom[1] * dom[2])
end

function flip_and_fuse(t)
    return fuse2(permute(t, (1,3,2),(4,5)))
end

function mpo_fuse_noflip(m)
    @tensoropt mn[1 -1 -2; 3 -3 4] := (m')[-1 -2; -3  x]*m[1 x; 3 4]
end

#orders smartly
function mpo_fuse_flip(m)
    @tensoropt mn[1 -3 -2; 3 -1 4] := ((m)')[-1 -2; -3  x]*(m)[1 x; 3 4]
end

function mpo_fuse_merge(t)
    cod = codomain(t)
    dom = domain(t)
    return TensorMap(t.data, fuse(cod[1]*cod[2])*cod[3], fuse(dom[1]*dom[2])*dom[3])
end

function bench_fuse_dim(d)
    s = SU2Space(0=>d, 1=>d, 2=>d)
    t = TensorMap(rand, ComplexF64, s*s*s,s*s)
    @btime fuse1($t)
    @btime fuse2($t)
end
function bench_fuse_sector(n)
    s = SU2Space((i/2 => 2 for i in 1:n)...)
    t = TensorMap(rand, ComplexF64, s*s*s,s*s)
    @btime fuse1($t)
    @btime fuse2($t)
end

function bench_flip_dim(d)
    s = SU2Space(0=>d, 1=>d, 2=>d)
    t = TensorMap(rand, ComplexF64, s*s,s*s*s)
    @btime flip_and_fuse($t)
end
function bench_flip_sector(n)
    s = SU2Space((i/2 => 2 for i in 1:n)...)
    t = TensorMap(rand, ComplexF64, s*s,s*s*s)
    @btime flip_and_fuse($t)
end

function bench_mpo_dim(d)
    s = SU2Space(0=>d, 1=>d, 2=>d)
    t = TensorMap(rand, ComplexF64, s*s,s*s)
    @btime mpo_fuse_noflip($t)
    @btime mpo_fuse_flip($t)
end
function bench_mpo_sector(n)
    s = SU2Space((i/2 => 2 for i in 1:n)...)
    t = TensorMap(rand, ComplexF64, s*s,s*s)
    @btime mpo_fuse_noflip($t)
    @btime mpo_fuse_flip($t)
end

function bench_full_dim(d)
    s = SU2Space(0=>d, 1=>d, 2=>d)
    m = TensorMap(rand, ComplexF64, s*s,s*s)
    t = mpo_fuse_flip(m)
    @btime mpo_fuse_merge($t)
end
function bench_full_sector(n)
    s = SU2Space((i/2 => 2 for i in 1:n)...)
    m = TensorMap(rand, ComplexF64, s*s,s*s)
    t = mpo_fuse_flip(m)
    @btime mpo_fuse_merge($t)
end


function compare_perm(n)
    s = SU2Space((i/2 => 2 for i in 1:n)...)
    m = TensorMap(rand, ComplexF64, s*s,s*s)
    @btime begin
        t0 = mpo_fuse_noflip($m)
    end
    @btime begin
        t1 = mpo_fuse_flip($m)
        mpo_fuse_merge(t1)
    end
    @btime begin
        t2 = mpo_fuse_noflip($m)
        permute(t2, (1,5,3),(4,2,6))
        mpo_fuse_merge(t2)
    end
end
