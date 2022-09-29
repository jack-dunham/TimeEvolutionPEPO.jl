using MPSKit

abstract type AbstractEnvTensor{S,T} <: AbstractPhysicalTensor{S,T} end
abstract type AbstractEnv{Nx,Ny,P<:AbstractEnvTensor} <: AbstractTensorMatrix{Nx,Ny,P} end

struct LeftEnvTensor{S<:IndexSpace,Nc,Nd,T<:TensorMap{S}} <: AbstractEnvTensor{S,T}
    tensor::T
    function LeftEnvTensor(t::TensorMap{S,N1,N2}) where {S<:IndexSpace,N1,N2}
        Nc = N1 - 1
        Nd = N2 - 1
        return new{S,Nc,Nd,typeof(t)}(t)
    end
end
struct RightEnvTensor{S<:IndexSpace,Nc,Nd,T<:TensorMap{S}} <: AbstractEnvTensor{S,T}
    tensor::T
    function RightEnvTensor(t::TensorMap{S,N1,N2}) where {S<:IndexSpace,N1,N2}
        Nc = N1 - 1
        Nd = N2 - 1
        return new{S,Nc,Nd,typeof(t)}(t)
    end
end
LeftEnvTensor{S,Nc,Nd,T}(t) where {S<:IndexSpace,Nc,Nd,T} = LeftEnvTensor(t)
RightEnvTensor{S,Nc,Nd,T}(t) where {S<:IndexSpace,Nc,Nd,T} = RightEnvTensor(t)

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

@doc raw"""
    Base.rand(::Type{LeftEnvTensor}, a::MPSTensor{S}, m::MPOtensor{S}) where {S<:IndexSpace}

Return a random `LeftEnvTensor` compatible with `a` and `m`. 
"""
function Base.rand(
    ::Type{LeftEnvTensor}, a::MPSTensor{S}, m::MPOTensor{S}
) where {S<:IndexSpace}
    cod, dom = westenvspace(m)
    lb = lbspace(a)
    t = TensorMap(rand, eltype(a), dom * lb, cod * lb)
    return LeftEnvTensor(t)
end

@doc raw"""
    Base.rand(::Type{RightEnvTensor}, a::MPSTensor{S}, m::MPOtensor{S}) where {S<:IndexSpace}

Return a random `RightEnvTensor` compatible with `a` and `m`. 
"""
function Base.rand(
    ::Type{RightEnvTensor}, a::MPSTensor{S}, m::MPOTensor{S}
) where {S<:IndexSpace}
    cod, dom = eastenvspace(m)
    rb = rbspace(a)
    t = TensorMap(rand, eltype(a), dom * rb, cod * rb)
    return RightEnvTensor(t)
end

struct LeftEnv{Nx,Ny,S<:IndexSpace,Kind,L<:LeftEnvTensor{S,Kind}} <:
       AbstractTensorMatrix{Nx,Ny,L}
    matrix::Matrix{L}
    cell::UnitCell{Nx,Ny,S}
end
struct RightEnv{Nx,Ny,S<:IndexSpace,Kind,R<:RightEnvTensor{S,Kind}} <:
       AbstractTensorMatrix{Nx,Ny,R}
    matrix::Matrix{R}
    cell::UnitCell{Nx,Ny,S}
end

function Base.rand(
    ::Type{LeftEnv}, a::MultilineInfMPS{Nx,Ny}, m::InfMPO{Nx,Ny}
) where {Nx,Ny}
    cell = unitcell(m)
    al = lefttensor(a)
    fl = [rand(LeftEnvTensor, al[x, y], m[x, y]) for x in 1:Nx, y in 1:Ny]
    return LeftEnv(fl, cell)
end

function Base.rand(
    ::Type{RightEnv}, a::MultilineInfMPS{Nx,Ny}, m::InfMPO{Nx,Ny}
) where {Nx,Ny}
    cell = unitcell(m)
    ar = righttensor(a)
    fr = [rand(RightEnvTensor, ar[x, y], m[x, y]) for x in 1:Nx, y in 1:Ny]
    return RightEnv(fr, cell)
end

function LeftEnv(a::MultilineInfMPS{Nx,Ny}, m::InfMPO{Nx,Ny}) where {Nx,Ny}
    fl = rand(LeftEnv, a, m)
    al = lefttensor(a)
    _, λl = envupdate!(fl, al, m)
    return fl, λl
end

function RightEnv(a::MultilineInfMPS{Nx,Ny}, m::InfMPO{Nx,Ny}) where {Nx,Ny}
    fr = rand(RightEnv, a, m)
    ar = righttensor(a)
    _, λr = envupdate!(fr, ar, m)
    return fr, λr
end

function environments(a::MultilineInfMPS{Nx,Ny}, m::InfMPO{Nx,Ny}) where {Nx,Ny}
    fl, λl = LeftEnv(a, m)
    fr, λr = RightEnv(a, m)
    return fl, fr, λl, λr
end

function environments!(
    A::MultilineInfMPS{Nx,Ny}, M::InfMPO{Nx,Ny}, FL::LeftEnv{Nx,Ny}, FR::RightEnv{Nx,Ny}
) where {Nx,Ny}
    C = centraltensor(A)

    AL = lefttensor(A)
    _, λL, infoL = envupdate!(FL, AL, M)
    infoL.converged < 1 && @warn "left fixed-point did not converge: $(infoL.normres)"

    AR = righttensor(A)
    _, λR, infoR = envupdate!(FR, AR, M)
    infoR.converged < 1 && @warn "right fixed-point did not converge: $(infoR.normres)"

    return FL, FR, λL, λR
end

function renorm(cb, ca, fl, fr)
    cbt = totensor(cb)
    cat = totensor(ca)
    return dot(cbt, hcapply(cat, totensor(fl), totensor(fr)))
end

function hcapply(
    c::AbstractTensorMap{S,1,1}, fl::AbstractTensorMap{S,1,2}, fr::AbstractTensorMap{S,2,1}
) where {S<:IndexSpace}
    @tensoropt hc[a; b] := c[x; y] * fl[a; m x] * fr[m y; b]
    return hc
end

function envupdate!(
    F::AbstractTensorMatrix{Nx,Ny},
    A::AbstractTensorMatrix{Nx,Ny,<:MPSTensor{S}},
    M::InfMPO{Nx,Ny,S},
) where {Nx,Ny,S<:IndexSpace}
    rx = 1:Nx
    ry = 1:Ny

    λ = Matrix{eltype(F)}(undef, Nx, Ny)

    info = 0

    for y in ry
        for x in rx
            λs, fs, info = eigsolve(z -> envsolve(z, A, M, x, y), F[x, y], 1, :LM)
            λ[x, y] = λs[1]
            F[x, y] = fs[1]
        end
    end
    return F, λ, info #mutated
end

function environments_alt(A::MultilineInfMPS, M)
    FL = rand(LeftEnv, A, M)
    FR = rand(RightEnv, A, M)
    return environments_alt!(A, M, FL, FR)
end

function environments_alt!(A::MultilineInfMPS, M::InfMPO, FL::LeftEnv, FR::RightEnv)
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



        # for x in 1:(Nx - 1)
        #     xp = mod(x + 1, rx)
        #     yp = mod(y + 1, ry)

        #     FL[xp, y] = FL[x, y] * TransferMatrix(AL[x, y], M[x, y], AL[x, yp])#works
        # end

        # for x in Nx:-1:2
        #     xm = mod(x - 1, rx)
        #     yp = mod(y + 1, ry)

        #     FR[xm, y] = TransferMatrix(AR[x, y], M[x, y], AR[x, yp]) * FR[x, y]#works

        #     NN = renorm(C[xm, yp], C[xm, y], FL[x, y], FR[xm, y])

        #     rmul!(FR[xm, y], 1 / sqrt(NN))
        #     rmul!(FL[x, y], 1 / sqrt(NN))
        # end
        
        FL[1,y] = Ls[1]

        for x in 2:Nx
            xm = mod(x - 1, rx)
            yp = mod(y + 1, ry)

            FL[x, y] = FL[xm, y] * TransferMatrix(AL[xm, y], M[xm, y], AL[xm, yp])#works
            
        end

        NN = renorm(C[Nx, yp], C[Nx, y], Ls[1], Rs[1])

        FL[1, y] = rmul!(FL[1,y], 1 / sqrt(NN)) #correct NN i think
        FR[Nx, y] = rmul!(Rs[1], 1 / sqrt(NN)) #correct NN

        for x in (Nx - 1):-1:1
            xp = mod(x + 1, rx)
            yp = mod(y + 1, ry)

            FR[x, y] = TransferMatrix(AR[xp, y], M[xp, y], AR[xp, yp]) * FR[xp, y]#works


            NN = renorm(C[x, yp], C[x, y], FL[xp, y], FR[x, y])

            rmul!(FR[x, y], 1 / sqrt(NN))
            rmul!(FL[xp, y], 1 / sqrt(NN))
        end
    end
    return FL, FR #mutated 
end

function my_fixpoints(
    above::MPSMultiline,
    mpo::MPOMultiline,
    below::MPSMultiline,
    init=MPSKit.gen_init_fps(above, mpo, below);
    solver=MPSKit.Defaults.eigsolver,
)
    T = eltype(above)

    #sanity check
    (numrows, numcols) = size(above)
    @assert size(above) == size(mpo)
    @assert size(below) == size(mpo)

    envtype = eltype(init[1])
    lefties = PeriodicArray{envtype,2}(undef, numrows, numcols)
    righties = PeriodicArray{envtype,2}(undef, numrows, numcols)

    for cr in 1:numrows
        c_above = above[cr]
        c_below = below[cr + 1]

        (L0, R0) = init[cr]

        E_LL = MPSKit.TransferMatrix(c_above.AL, mpo[cr, :], c_below.AL)

        packed_init = L0 isa Vector ? RecursiveVec(L0) : L0
        (valsL, Ls, convhist) = eigsolve(flip(E_LL), packed_init, 1, :LM, solver)
        convhist.converged < 1 &&
            @info "left eigenvalue failed to converge (convhist.normres)"
        L0 = L0 isa Vector ? Ls[1].vecs : Ls[1]

        packed_init = R0 isa Vector ? RecursiveVec(R0) : R0
        E_RR = MPSKit.TransferMatrix(c_above.AR, mpo[cr, :], c_below.AR)
        (valsR, Rs, convhist) = eigsolve(E_RR, packed_init, 1, :LM, solver)
        convhist.converged < 1 &&
            @info "right eigenvalue failed to converge (convhist.normres)"
        R0 = R0 isa Vector ? Rs[1].vecs : Rs[1]

        lefties[cr, 1] = L0
        

        for loc in 2:numcols
            lefties[cr, loc] =
                lefties[cr, loc - 1] * MPSKit.TransferMatrix(
                    c_above.AL[loc - 1], mpo[cr, loc - 1], c_below.AL[loc - 1]
                )
           
            println(norm(lefties[cr,loc]))

            # println(norm(c_above.AL[loc-1]))
            # println(norm(c_below.AL[loc-1]))
            # println(norm(mpo[cr,loc-1]))
        end

        renormfact::eltype(T) = dot(c_below.CR[0], MPSKit.MPO_∂∂C(L0, R0) * c_above.CR[0])


        righties[cr, end] = R0 / sqrt(renormfact)
        lefties[cr, 1] /= sqrt(renormfact)
        


        for loc in (numcols - 1):-1:1
            righties[cr, loc] =
                MPSKit.TransferMatrix(
                    c_above.AR[loc + 1], mpo[cr, loc + 1], c_below.AR[loc + 1]
                ) * righties[cr, loc + 1]


            renormfact = dot(
                c_below.CR[loc],
                MPSKit.MPO_∂∂C(lefties[cr, loc + 1], righties[cr, loc]) * c_above.CR[loc],
            )
            
            righties[cr, loc] /= sqrt(renormfact)
            lefties[cr, loc + 1] /= sqrt(renormfact)
        end
    end

    return (lefties, righties)
end

function envsolve(
    FL::LeftEnvTensor{S},
    A::AbstractTensorMatrix{Nx,Ny,<:MPSTensor{S}},
    M::InfMPO{Nx,Ny,S},
    x::Int,
    y::Int,
) where {Nx,Ny,S<:IndexSpace}
    fl = flsolve(totensor(FL), tomatrixt(A), tomatrixt(M), x, y, Val(Nx))
    return LeftEnvTensor(fl)
end
function envsolve(
    FR::RightEnvTensor{S},
    A::AbstractTensorMatrix{Nx,Ny,<:MPSTensor{S}},
    M::InfMPO{Nx,Ny,S},
    x::Int,
    y::Int,
) where {Nx,Ny,S<:IndexSpace}
    fr = frsolve(totensor(FR), tomatrixt(A), tomatrixt(M), x, y, Val(Nx))
    return RightEnvTensor(fr)
end

function flsolve(
    fl::TensorMap{S,1,2},
    a::Array{<:TensorMap{S,2,1}},
    m::Array{<:TensorMap{S,2,2}},
    x::Int,
    y::Int,
    ::Val{2},
) where {S<:IndexSpace}
    flu = similar(fl)
    nx, ny = size(a)
    rx = 1:nx
    ry = 1:ny
    xp = mod(x + 1, rx)
    yp = mod(y + 1, ry)
    @tensoropt begin
        flu[13; 12 11] =
            fl[3; 2 1] *
            (a[x, y])[4 1; 6] *
            (m[x, y])[2 5; 7 4] *
            (a[x, yp]')[8; 5 3] *
            (a[xp, y])[9 6; 11] *
            (m[xp, y])[7 10; 12 9] *
            (a[xp, yp]')[13; 10 8]
    end
    return flu
end

function flsolve(
    fl::TensorMap{S,1,2},
    a::Array{<:TensorMap{S,2,1}},
    m::Array{<:TensorMap{S,2,2}},
    x::Int,
    y::Int,
    ::Val{3},
) where {S<:IndexSpace}
    flu = similar(fl)
    nx, ny = size(a)
    rx = 1:nx
    ry = 1:ny
    xp = mod(x + 1, rx)
    xpp = mod(x + 2, rx)
    yp = mod(y + 1, ry)
    @tensoropt begin
        flu[18; 17 16] =
            fl[3; 2 1] *
            (a[x, y])[4 1; 6] *
            (m[x, y])[2 5; 7 4] *
            (a[x, yp]')[8; 5 3] *
            (a[xp, y])[9 6; 11] *
            (m[xp, y])[7 10; 12 9] *
            (a[xp, yp]')[13; 10 8] *
            (a[xpp, y])[14 11; 16] *
            (m[xpp, y])[12 15; 17 14] *
            (a[xpp, yp]')[18; 15 13]
    end
    return flu
end

function frsolve(
    fr::TensorMap{S,2,1},
    a::Array{<:TensorMap{S,2,1}},
    m::Array{<:TensorMap{S,2,2}},
    x::Int,
    y::Int,
    ::Val{2},
) where {S<:IndexSpace}
    fru = similar(fr)
    nx, ny = size(a)
    rx = 1:nx
    ry = 1:ny

    xm = mod(x - 1, rx)
    yp = mod(y + 1, ry)
    @tensoropt begin
        fru[2 1; 3] =
            (a[xm, y])[4 1; 6] *
            (m[xm, y])[2 5; 7 4] *
            (a[xm, yp]')[8; 5 3] *
            (a[x, y])[9 6; 11] *
            (m[x, y])[7 10; 12 9] *
            (a[x, yp]')[13; 10 8] *
            fr[12 11; 13]
    end
    return fru
end
function frsolve(
    fr::TensorMap{S,2,1},
    a::Array{<:TensorMap{S,2,1}},
    m::Array{<:TensorMap{S,2,2}},
    x::Int,
    y::Int,
    ::Val{3},
) where {S<:IndexSpace}
    fru = similar(fr)
    nx, ny = size(a)
    rx = 1:nx
    ry = 1:ny

    xm = mod(x - 1, rx)
    xmm = mod(x - 2, rx)
    yp = mod(y + 1, ry)
    @tensoropt begin
        fru[2 1; 3] =
            (a[xmm, y])[4 1; 6] *
            (m[xmm, y])[2 5; 7 4] *
            (a[xmm, yp]')[8; 5 3] *
            (a[xm, y])[9 6; 11] *
            (m[xm, y])[7 10; 12 9] *
            (a[xm, yp]')[13; 10 8] *
            (a[x, y])[14 11; 16] *
            (m[x, y])[12 15; 17 14] *
            (a[x, yp]')[18; 15 13] *
            fr[17 16; 18]
    end
    return fru
end

#TODO: Deprec?
function tensorise(r::RightEnv)
    rm = tomatrixt(r)
    rt = _tensorize(rm; conj=true)
    return permute(rt, (1, 2, 4, 5), (3,))
end

function get_ltransfer(A, M::InfMPO, y=Int)
    Nx, Ny = size(M)
    Ts = []
    yp = mod(y + 1, 1:Ny)
    for x in 1:Nx
        push!(
            Ts, transfer_matrix(totensor(A[x, y]), totensor(M[x, y]), totensor(A[x, yp])')
        )
    end
    return Ts
end

function transfer(fl, T::Vector)
    flu = foldl(*, T; init=fl)
    return flu
end

function Base.convert(::Type{MPSMultiline}, a::MultilineInfMPS)
    AL = tomatrixt(lefttensor(a))
    AR = tomatrixt(righttensor(a))
    AC = tomatrixt(centraltensor(a))
    C = tomatrixt(bondtensor(a))

    AL = broadcast(z -> permute(z, (2, 1), (3,)), AL)
    AR = broadcast(z -> permute(z, (2, 1), (3,)), AR)
    AC = broadcast(z -> permute(z, (2, 1), (3,)), AC)
    mps = []

    for y in 1:length(a)
        m = InfiniteMPS(
            PeriodicArray(AL[:, y]),
            PeriodicArray(AR[:, y]),
            PeriodicArray(C[:, y]),
            PeriodicArray(AC[:, y]),
        )
        push!(mps, m)
    end
    return MPSKit.Multiline{typeof(mps[1])}(PeriodicArray(mps))
end

function Base.convert(::Type{MPOMultiline}, a::InfMPO)
    b = permutedims(tomatrixt(a), (2, 1))
    return MPOMultiline(b)
end

function environments_mps(A::MultilineInfMPS, M::InfMPO)
    a = convert(MPSMultiline, A)
    m = convert(MPOMultiline, M)
    return MPSKit.environments(a, m)
end

function _getenvs(envs, A)
    lws = broadcast(z -> permute(z, (1,), (2, 3)), envs.lw)
    rws = broadcast(z -> permute(z, (2, 1), (3,)), envs.rw)
    FL = LeftEnv(LeftEnvTensor.(permutedims(lws, (2, 1))), A.cell)
    FR = RightEnv(RightEnvTensor.(permutedims(rws, (2, 1))), A.cell)
    # go use FL and FR
    return FL, FR
end

function environments_mps!(envs, A::MultilineInfMPS)
    nstate = convert(MPSMultiline, A)

    recalculate!(envs, nstate)

    lws = broadcast(z -> permute(z, (1,), (2, 3)), envs.lw)
    rws = broadcast(z -> permute(z, (2, 1), (3,)), envs.rw)
    FL = LeftEnv(LeftEnvTensor.(permutedims(lws, (2, 1))), A.cell)
    FR = RightEnv(RightEnvTensor.(permutedims(rws, (2, 1))), A.cell)
    # go use FL and FR
    return FL, FR
end
