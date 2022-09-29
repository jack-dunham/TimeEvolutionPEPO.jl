const PEPO_ORDER = (:ket, :xout, :yout, :sx, :bra, :xin, :yin, :sy)
# const VPEPO_ORDER = (:bra, :ket, :xout, :yout, :sx, :xin, :yin, :sy)

abstract type AbstractSitePEPO end

mutable struct SitePEPO{T<:TensorMap{<:IndexSpace,4,4}} <:AbstractSitePEPO
    tensor::T
    perm::NTuple{8,Int}
end

function Base.permutedims!(p::AbstractSitePEPO, perm::NTuple{8,Int})
    p.tensor = _permute(p.tensor, perm)
    ptmp = Base.permute!(collect(p.perm), collect(perm))
    p.perm = Tuple(ptmp)
    return nothing
end

SitePEPO(T::TensorMap{<:IndexSpace,4,4}) = VectorSitePEPO(T, Tuple(1:8))

Base.getindex(P::SitePEPO, ind::Int) = PEPO_ORDER[P.perm[ind]]
function Base.getindex(P::SitePEPO, ind::Symbol)
    ind_num = findfirst(x -> x === ind, PEPO_ORDER)
    return findfirst(x -> x == ind_num, P.perm)
end

struct AdjointSitePEPO{T<:TensorKit.AdjointTensorMap}
    tensor::T
    perm::NTuple{8,Int}
end

# function LinearAlgebra.vec(p::SitePEPO)
#     t = p.tensor
#     N₁ = numout(t)
#     N = numind(t)
#     perm = ((p.bra, Tuple(2:N₁+1)...), Tuple(N₁+2:N))  
#     return VectorSitePEPO(permute(t, perm...),perm, 1, p.ket )
# end


function _vec(t::TensorMap{<:IndexSpace,N,M}) where {N,M}
    return permute(t, (Tuple((N + 1):(N + M))..., Tuple(1:N)...)),
    (Tuple((M + 1):(N + M)), Tuple(1:M))
end

function bloch(a::Vector)
    σ = pauli
    mat = σ[4].data
    for j in 1:3
        mat = mat + a[j] * σ[j].data
    end
    return 1 / 2 * mat
end

function blochpepo(a, D)
    m = bloch(a)
    to = zeros(ComplexF64, 2, D, 2)
    to[:, 1, :] = m
    ti = zeros(ComplexF64, 2, 2, D)
    ti[:, :, 1] = m
    p2 = TensorMap(to, ℂ^2 * ℂ^D, ℂ^2)
    p3 = TensorMap(to, ℂ^2 * ℂ^D, ℂ^2)
    p5 = TensorMap(ti, ℂ^2, ℂ^2 * ℂ^D)
    p6 = TensorMap(ti, ℂ^2, ℂ^2 * ℂ^D)
    return @tensor p[1 2 3; 4 5 6] :=
        p2[s1 2; 4] * p3[s2 3; s1] * p5[s3; s2 5] * p6[1; s3 6]
end

function test_evolve(p, η)
    id = Matrix{ComplexF64}(I(2))
    arr = zeros(ComplexF64, 2, η, 2, η)
    arr[:, 1, :, 1] = id
    m = TensorMap(arr, ℂ^2 * ℂ^η, ℂ^2 * ℂ^η)
    return @tensor pe[1 2 3 4; 5 6 7 8] := p[s 2 4; 5 6 8] * m[1 3; s 7]
end

function evolve(vp::TensorMap{S,4,2}, m::TensorMap{S,3,3}) where {S<:IndexSpace}
    return @tensor pe[b k 3 4 5; 6 7 8] := vp[mb mk 3 5; 6 8] * m[b k 4; mb mk 7]
end
function evolve(p::TensorMap{S,3,3}, m::TensorMap{S,3,3}) where {S<:IndexSpace}
    return @tensor pe[k 3 4 5;b 6 7 8] := p[mk 3 5; mb 6 8] * m[b k 4; mb mk 7]
end

function test_thermal(τ, D=4, J=1.0)
    βc = log(1 + sqrt(2)) / 2
    σ = pauli
    l = exp(1 / 2 * τ * βc * σ[3] ⊗ σ[3])
    lb = TensorMap(convert(Array, l), (ℂ^2)' * (ℂ^2)', (ℂ^2)' * (ℂ^2)')
    L = permute(lb ⊗ l, (1, 3, 2, 4), (5, 7, 6, 8))
    u, s, v, err = tsvd(L, (1, 2, 5, 6), (3, 4, 7, 8); trunc=truncdim(D))
    print("err: ", err)
    up = permute(u * sqrt(s), (1, 2), (3, 4, 5))
    vp = permute(sqrt(s) * v, (2, 3, 1), (4, 5))

    @tensor m[sk sb l; tk tb r] := up[mk mb; tk tb r] * vp[sk sb l; mk mb]

    return m
end

function test_thermal_sym(τ, D=4, J=1.0)
    S = Z2Space(0 => 1, 1 => 1)

    βc = log(1 + sqrt(2)) / 2
    σ = pauli
    l = TensorMap(convert(Array, exp(1 / 2 * τ * βc * σ[3] ⊗ σ[3])), S * S, S * S)
    lb = TensorMap(convert(Array, l), S' * S', S' * S')
    L = permute(lb ⊗ l, (1, 3, 2, 4), (5, 7, 6, 8))
    u, s, v, err = tsvd(L, (1, 2, 5, 6), (3, 4, 7, 8); trunc=truncdim(D))
    print("err: ", err)
    up = permute(u * sqrt(s), (1, 2), (3, 4, 5))
    vp = permute(sqrt(s) * v, (2, 3, 1), (4, 5))

    @tensor m[sk sb l; tk tb r] := up[mk mb; tk tb r] * vp[sk sb l; mk mb]

    return m
end

# function test_th_ev(t)
#     τ = t / 20
#     m = test_thermal(τ)
#     p = 2 * PEPOKit.blochpepo([0.0, 0.0, 0.0], 2)
#     vp = permute(p, (4, 1, 2, 3), (5, 6))

#     mag = []

#     @time begin
#         for i in 1:20
#             vp = fetstep!(vp, m, :x)
#             _, ptr = wip_normalise(vp)
#             vp = fetstep!(vp, m, :y)
#             _, ptr = wip_normalise(vp)
#             push!(mag, abs(tr(σ3 * ptr)))
#         end
#     end
#     return vp, mag
# end
function test_th_ev(t)
    n = 20
    τ = t / n
    m = test_thermal(τ)
    p = 2 * PEPOKit.blochpepo([0.0, 0.0, 0.0], 2)
    vp = permute(p, (4, 1, 2, 3), (5, 6))

    vp = propagate(vp, m, n, 2)

    return vp
end
function test_th_ev2(t)
    n = 20
    τ = t / n
    m = test_thermal(τ)
    p = 2 * PEPOKit.blochpepo([0.0, 0.0, 0.0], 2)

    p = propagate(p, m, n, 2)

    return vp
end

function test_th_ev_sym(t)
    S = Z2Space(0 => 1, 1 => 1)

    pi = 1 / 2 * Matrix{ComplexF64}([1 0; 0 1])
    to = zeros(ComplexF64, 2, 2, 2)
    to[:, 1, :] = pi
    ti = zeros(ComplexF64, 2, 2, 2)
    ti[:, :, 1] = pi
    p2 = TensorMap(to, S * Z2Space(0 => 2), S)
    p3 = TensorMap(to, S * Z2Space(0 => 2), S)
    p5 = TensorMap(ti, S, S * Z2Space(0 => 2))
    p6 = TensorMap(ti, S, S * Z2Space(0 => 2))
    @tensor p[1 2 3; 4 5 6] := p2[s1 2; 4] * p3[s2 3; s1] * p5[s3; s2 5] * p6[1; s3 6]

    τ = t / 20
    m = test_thermal_sym(τ)
    p = 2 * p
    vp = permute(p, (4, 1, 2, 3), (5, 6))
    mag = []

    Z = TensorMap(σ3.data, S, S)

    @time begin
        for i in 1:20
            vp = fetstep!(vp, m, :x)
            _, ptr = wip_normalise(vp)
            vp = fetstep!(vp, m, :y)
            _, ptr = wip_normalise(vp)
            push!(mag, abs(tr(Z * ptr)))
        end
    end
    return vp, mag
end

function wip_normalise(vp::TensorMap{S}) where {S<:IndexSpace}
    @tensor vtr[2 3; 5 6] := vp[s s 2 3; 5 6]
    p = permute(vp, (2, 3, 4), (1, 5, 6))
    sp = hcat(codomain(vtr)..., domain(vtr)...)

    unitsp = oneunit(S)
    bondsp = ⊕(fill(unitsp, dim(sp[4])^2)...)

    A = TensorMap(randn, ComplexF64, sp[4] * bondsp, bondsp)

    λ, AL, C, AR, FL, FR = vumps(A, vtr; maxiter=20, verbose=false, tol=1e-20)
    @assert isapprox(_AC(AL, C), _CA(C, AR), rtol=1e-4)
    AC = _AC(AL, C)
    @tensoropt begin
        ptr[o; i] :=
            FL[b1; 1 a1] * AC[4 a1; a2] * p[o 1 2; i 3 4] * (AC')[b2; 2 b1] * FR[3 a2; b2]
    end

    return vp / tr(ptr), ptr / tr(ptr)
end
