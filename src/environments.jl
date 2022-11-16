abstract type AbstractEnvTensor{S,N₁,N₂,T} <: AbstractPhysicalTensor{S,N₁,N₂,T} end
abstract type AbstractEnv{L,P<:AbstractEnvTensor} <: AbstractLattice{L,P} end

struct LeftEnvTensor{S,T<:AbstractPhysicalTensor{S,1,2}} <: AbstractEnvTensor{S,1,2,T}
    tensor::T
end
struct RightEnvTensor{S,T<:AbstractPhysicalTensor{S,2,1}} <: AbstractEnvTensor{S,2,1,T}
    tensor::T
end
# LeftEnvTensor{S,Nc,Nd,T}(t) where {S<:IndexSpace,Nc,Nd,T} = LeftEnvTensor(t)
# RightEnvTensor{S,Nc,Nd,T}(t) where {S<:IndexSpace,Nc,Nd,T} = RightEnvTensor(t)

function LeftEnvTensor(f, a::MPSTensor, m::MPOTensor)
    sp_m = codomain(m)[1]
    sp_a = codomain(a)[1]
    return LeftEnvTensor(TensorMap(f, ComplexF64, sp_a, sp_a * sp_m))
end
function RightEnvTensor(f, a::MPSTensor, m::MPOTensor)
    sp_m = domain(m)[1]
    sp_a = domain(a)[1]
    return RightEnvTensor(TensorMap(f, ComplexF64, sp_a * sp_m, sp_a))
end

function westenvspace(m::MPOTensor)
    N = numout(m)
    codi, domi = westind(m)
    cod = slice(codi, codomain(m))
    dom = slice(domi .- N, domain(m))
    return cod, dom
end
function eastenvspace(m::MPOTensor)
    N = numout(m)
    codi, domi = eastind(m)
    cod = slice(codi, codomain(m))
    dom = slice(domi .- N, domain(m))
    return cod, dom
end

struct LeftEnv{La,L<:LeftEnvTensor} <: AbstractEnv{La,L}
    data::Matrix{L}
end
struct RightEnv{La,R<:RightEnvTensor} <: AbstractEnv{La,R}
    data::Matrix{R}
end

function LeftEnv(f, A::MultilineInfMPS, M::InfMPO{L}) where {L}
    nx, ny = size(M)
    mat = [LeftEnvTensor(f,A[x,y],M[x,y]) for x in 1:nx, y in 1:ny]
    return LeftEnv{L}(mat)
end
function RightEnv(f, A::MultilineInfMPS, M::InfMPO{L}) where {L}
    nx, ny = size(M)
    mat = [RightEnvTensor(f,A[x,y],M[x,y]) for x in 1:nx, y in 1:ny]
    return RightEnv{L}(mat)
end

function renorm(cb, ca, fl, fr)
    cbt = totensor(cb)
    cat = totensor(ca)
    return dot(cbt, hcapply(cat, totensor(fl), totensor(fr)))
end

function hcapply(
    c::AbstractTensorMap{S,1,1}, fl::AbstractTensorMap{S,1,2}, fr::AbstractTensorMap{S,2,1}
) where {S<:IndexSpace}
    @tensoropt hc[a; b] := c[x; y] * fl[a; x m] * fr[y m; b]
    return hc
end

function environments(A::MultilineInfMPS, M::InfMPO)
    FL = LeftEnv(rand, A, M)
    FR = RightEnv(rand, A, M)
    return environments!(A, M, FL, FR)
end

#this is now the correct env func
function environments!(FL::LeftEnv, FR::RightEnv, A::MultilineInfMPS, M::InfMPO)
    Nx, Ny = size(M)
    rx = 1:Nx
    ry = 1:Ny

    λ = Matrix{eltype(M)}(undef, Nx, Ny)

    info = 0

    AL = lefttensor(A)
    AR = righttensor(A)
    C = bondtensor(A)

    for y in ry
        xi = 1

        yp = mod(y + 1, ry)

        ls, Ls, infol = eigsolve(z -> envsolve(z, AL, M, 1, y), FL[1, y], 1, :LM)
        rs, Rs, infor = eigsolve(z -> envsolve(z, AR, M, Nx, y), FR[Nx, y], 1, :LM)

        FL[1, y] = Ls[1]

        for x in 2:Nx
            xm = mod(x - 1, rx)
            yp = mod(y + 1, ry)

            FL[x, y] = FL[xm, y] * TransferMatrix(AL[xm, y], M[xm, y], AL[xm, yp])#works
        end

        NN = renorm(C[Nx, yp], C[Nx, y], Ls[1], Rs[1])

        # the eigenvalue problem eqn gives us FL[1,y] and FR[end,y], so normalise them
        FL[1, y] = rmul!(FL[1, y], 1 / sqrt(NN)) #correct NN 
        FR[Nx, y] = rmul!(Rs[1], 1 / sqrt(NN)) #correct NN

        for x in (Nx - 1):-1:1
            xp = mod(x + 1, rx)
            yp = mod(y + 1, ry)

            FR[x, y] = TransferMatrix(AR[xp, y], M[xp, y], AR[xp, yp]) * FR[xp, y]#works

            NN = renorm(C[x, yp], C[x, y], FL[xp, y], FR[x, y])

            rmul!(FL[xp, y], 1 / sqrt(NN))
            rmul!(FR[x, y], 1 / sqrt(NN))
        end
    end
    return FL, FR #mutated 
end

function envsolve(
    FL::LeftEnvTensor, A::AbstractLattice{L,<:MPSTensor}, M::InfMPO{L}, x::Int, y::Int
) where {Nx,L<:LatticeSize{Nx}}
    fl = flsolve(totensor(FL), to_matrixt(A), to_matrixt(M), x, y, Val(Nx))
    return LeftEnvTensor(fl)
end
function envsolve(
    FR::RightEnvTensor, A::AbstractLattice{L,<:MPSTensor}, M::InfMPO{L}, x::Int, y::Int
) where {Nx,L<:LatticeSize{Nx}}
    fr = frsolve(totensor(FR), to_matrixt(A), to_matrixt(M), x, y, Val(Nx))
    return RightEnvTensor(fr)
end

function flsolve(
    fl::AbstractTensorMap{S,1,2},
    a::AbstractMatrix{<:AbstractTensorMap{S,2,1}},
    m::AbstractMatrix{<:AbstractTensorMap{S,2,2}},
    x::Int,
    y::Int,
    ::Val{2},
) where {S<:IndexSpace}
    flu = similar(fl)
    @tensoropt begin
        flu[13; 11 12] =
            fl[3; 1 2] *
            (a[x, y])[1 4; 6] *
            (m[x, y])[2 5; 7 4] *
            (a[x, y + 1]')[8; 3 5] *
            (a[x + 1, y])[6 9; 11] *
            (m[x + 1, y])[7 10; 12 9] *
            (a[x + 1, y + 1]')[13; 8 10]
    end
    return flu
end

function flsolve(
    fl::AbstractTensorMap{S,1,2},
    a::AbstractMatrix{<:AbstractTensorMap{S,2,1}},
    m::AbstractMatrix{<:AbstractTensorMap{S,2,2}},
    x::Int,
    y::Int,
    ::Val{3},
) where {S<:IndexSpace}
    flu = similar(fl)
    @tensoropt begin
        flu[18; 16 17] =
            fl[3; 1 2] *
            (a[x, y])[1 4; 6] *
            (m[x, y])[2 5; 7 4] *
            (a[x, y + 1]')[8; 3 5] *
            (a[x + 1, y])[6 9; 11] *
            (m[x + 1, y])[7 10; 12 9] *
            (a[x + 1, y + 1]')[13; 8 10]*
            (a[x + 2, y])[11 14; 16] *
            (m[x + 2, y])[12 15; 17 14] *
            (a[x + 2, y + 1]')[18; 13 15]
    end
    return flu
end

function frsolve(
    fr::AbstractTensorMap{S,2,1},
    a::AbstractMatrix{<:AbstractTensorMap{S,2,1}},
    m::AbstractMatrix{<:AbstractTensorMap{S,2,2}},
    x::Int,
    y::Int,
    ::Val{2},
) where {S<:IndexSpace}
    fru = similar(fr)
    @tensoropt begin
        fru[1 2; 3] =
            (a[x - 1, y])[1 4; 6] *
            (m[x - 1, y])[2 5; 7 4] *
            (a[x - 1, yp]')[8; 3 5] *
            (a[x, y])[6 9; 11] *
            (m[x, y])[7 10; 12 9] *
            (a[x, y + 1]')[13; 8 10] *
            fr[11 12; 13]
    end
    return fru
end
function frsolve(
    fr::AbstractTensorMap{S,2,1},
    a::AbstractMatrix{<:AbstractTensorMap{S,2,1}},
    m::AbstractMatrix{<:AbstractTensorMap{S,2,2}},
    x::Int,
    y::Int,
    ::Val{3},
) where {S<:IndexSpace}
    fru = similar(fr)
    @tensoropt begin
        fru[1 2; 3] =
            (a[x - 2, y])[1 4; 6] *
            (m[x - 2, y])[2 5; 7 4] *
            (a[x - 2, yp]')[8; 3 5] *
            (a[x - 1, y])[6 9; 11] *
            (m[x - 1, y])[7 10; 12 9] *
            (a[x - 1, y + 1]')[13; 8 10] *
            (a[x, y])[11 14; 16] *
            (m[x, y])[12 15; 17 14] *
            (a[x, y + 1]')[18; 13 15] *
            fr[16 17; 18]
    end
    return fru
end
