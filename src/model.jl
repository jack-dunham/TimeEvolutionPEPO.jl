abstract type AbstractModel end

# struct Model{N}
#     xdata::Array{ComplexF64,N}
#     ydata::Array{ComplexF64,N}
# end

struct ModelMPO <: AbstractModel
    A::Matrix{ComplexF64}
    B::Matrix{ComplexF64}
    C::Matrix{ComplexF64}
    D::Matrix{ComplexF64}
end

struct Model <: AbstractModel
    xdata::Matrix{ComplexF64}
    ydata::Matrix{ComplexF64}
    param::Vector{String}
    hamiltonian::String
    linops::Vector{String}
    Model(x, y, p, h, l) = new(x, y, p, h, l)
    Model(x, y) = new(x, y)
end

abstract type AbstractTerm{N} end

# This only accepts concrete types
function newterm(::Type{T}, term, args...) where {N,T<:AbstractTerm{N}}
    term = map(convert_if_identity, term)
    return T(term, args...)
end

physicaldim(term::AbstractTerm) = map(x -> size(x, 1), term.term)

function Base.:(==)(t1::T, t2::T) where {T<:AbstractTerm}
    return all(name -> getfield(t1, name) == getfield(t2, name), fieldnames(T))
end

convert_if_identity(I::UniformScaling) = Matrix{ComplexF64}(I, (2, 2))
convert_if_identity(x) = x

## TERM STRUCTS

struct HamiltonianTerm{N} <: AbstractTerm{N}
    term::NTuple{N,Matrix{ComplexF64}}
    coeff::ComplexF64
end

HamiltonianTerm(term::NTuple{N,Any}, a=1) where {N} = newterm(HamiltonianTerm{N}, term, a)
scaleterm(a, A::HamiltonianTerm) = HamiltonianTerm(A.term, a * A.coeff)

struct LongRangeTerm <: AbstractTerm{2}
    term::NTuple{2,Matrix{ComplexF64}}
    decay::ComplexF64
    coeff::ComplexF64
    LongRangTerm(term::NTuple{2,Matrix{ComplexF64}}, d, c) = new(term, d, c)
end

LongRangeTerm(term, λ, a=1) = newterm(LongRangeTerm, term, λ, a)
scaleterm(a, A::LongRangeTerm) = LongRangeTerm(A.term, a * A.coeff, A.decay)

struct LindbladTerm <: AbstractTerm{1}
    term::NTuple{1,Matrix{ComplexF64}}
    coeff::ComplexF64
    LindbladTerm(term::NTuple{1,Matrix{ComplexF64}}, c) = new(term, c)
end

LindbladTerm(term, a=1) = newterm(LindbladTerm, term, a)
scaleterm(a, A::LindbladTerm) = LindbladTerm(A.term, a * A.coeff)

function (A::Type{<:AbstractArray})(term::Union{HamiltonianTerm,LindbladTerm})
    mat = term.coeff * kron(1, term.term...)
    return convert(A, mat)
end
## 

## TERM API + ALGEBRA

ham(A) = HamiltonianTerm((A,))
ham(A, B) = HamiltonianTerm((A, B))
ham(A, λ, B) = LongRangeTerm((A, B), λ)

lin(L) = LindbladTerm((L,))

Base.:*(a::Number, A::AbstractTerm) = scaleterm(a, A)
Base.:*(A::AbstractTerm, a::Number) = Base.:*(a, A)

##
abstract type AbstractLindbladEquation end

struct ZeroLindbladEquation <: AbstractLindbladEquation end

function push!!(eqn::ZeroLindbladEquation, term::AbstractTerm)
    outeqn = LindbladEquation(physicaldim(term)[1])
    push!!(outeqn, term)
    return outeqn
end

struct LindbladEquation <: AbstractLindbladEquation
    onelocal::Vector{HamiltonianTerm{1}}
    twolocal::Vector{HamiltonianTerm{2}}
    longrange::Vector{LongRangeTerm}
    linops::Vector{LindbladTerm}
    physicaldim::Int
end
LindbladEquation(p::Int) = LindbladEquation([], [], [], [], p::Int)

function Base.show(io::IO, ::MIME"text/plain", eqn::LindbladEquation)
    d = physicaldim(eqn)

    n1 = length(eqn.onelocal)
    n2 = length(eqn.twolocal)

    nl = length(eqn.longrange)

    nd = length(eqn.linops)

    println(io, "LindbladEquation with d = $d containing:")
    println(io, "  $n1 local terms")
    println(io, "  $n2 nearest-neighbour terms")
    println(io, "  $nl long-range terms")
    return print(io, "  $nd lindblad operators")
end

Base.zero(eqn::AbstractLindbladEquation) = zero(typeof(eqn))
Base.zero(::Type{<:AbstractLindbladEquation}) = ZeroLindbladEquation()

physicaldim(eqn::LindbladEquation) = eqn.physicaldim
function physicalspace(S::Type{<:IndexSpace}, eqn::LindbladEquation)
    d = eqn.physicaldim
    return ⊕(oneunit(S)^d...)
end

function physicalspace(S::Type{<:IndexSpace}, mpo::MatrixProductOperator)
    d = Int(sqrt(size(mpo.D[1,1], 1)))
    return ⊕(oneunit(S)^d...)
end

function Base.:+(A::AbstractTerm, B::AbstractTerm)
    eqn = push!!(zero(LindbladEquation), A)
    eqn = push!!(eqn, B)
    return eqn
end

function Base.:(==)(t1::T, t2::T) where {T<:LindbladEquation}
    return all(name -> getfield(t1, name) == getfield(t2, name), fieldnames(T))
end

function push!!(L::LindbladEquation, A::AbstractTerm)
    ld = physicaldim(L)
    ad = physicaldim(A)
    if ld == ad[1]
        push!(L, A)
        return L
    else
        throw(
            DimensionMismatch(
                "cannot add term of dimension $ad to lindblad equation of dimension $ld"
            ),
        )
    end
end

Base.push!(L::LindbladEquation, A::HamiltonianTerm{1}) = push!(L.onelocal, A)
Base.push!(L::LindbladEquation, A::HamiltonianTerm{2}) = push!(L.twolocal, A)
Base.push!(L::LindbladEquation, A::LongRangeTerm) = push!(L.longrange, A)
Base.push!(L::LindbladEquation, A::LindbladTerm) = push!(L.linops, A)

Base.:+(B::AbstractTerm, eqn::AbstractLindbladEquation) = +(eqn, B)
function Base.:+(eqn::AbstractLindbladEquation, B::AbstractTerm)
    eqnout = zero(eqn)
    eqnout = axpy!(1, eqn, eqnout)
    eqnout = push!!(eqnout, B)
    return eqnout
end

function LinearAlgebra.axpy!(
    a::Number, eqn1::ZeroLindbladEquation, eqn2::AbstractLindbladEquation
)
    return eqn2
end

function LinearAlgebra.axpy!(
    a::Number, eqn1::LindbladEquation, eqn2::AbstractLindbladEquation
)
    for name in (:onelocal, :twolocal, :longrange, :linops)
        for term in getfield(eqn1, name)
            eqn2 = push!!(eqn2, a * term)
        end
    end
    return eqn2
end

function Base.:+(eqn1::AbstractLindbladEquation, eqn2::AbstractLindbladEquation)
    eqnout = zero(eqn1)
    eqnout = axpy!(1, eqn1, eqnout)
    eqnout = axpy!(1, eqn2, eqnout)
    return eqnout
end

Base.:*(eqn::LindbladEquation, a::Number) = *(a, eqn)
function Base.:*(a::Number, eqn::LindbladEquation)
    eqnout = zero(eqn)
    axpy!(a, eqn, eqnout)
    return eqnout
end

function xyzlindblad(; Jx=0.5, Jy=1.2, Jz=1.0, Γ=1.0)
    X, Y, Z = PAULI
    M, _ = LADDER

    eqn = Jx * ham(X, X) + Jy * ham(Y, Y) + Jz * ham(Z, Z) + sqrt(Γ) * lin(M)

    return eqn
end

function exactlindblad(; V=1.0, Γ=1.0)
    _, Y, Z = PAULI

    eqn = V / 4 * ham(Z, Z) + sqrt(Γ) / 2 * lin(Y - im * Z)

    return eqn
end

const BlockMatrix = Matrix{Matrix{ComplexF64}}

struct MatrixProductOperator
    A::BlockMatrix
    B::BlockMatrix
    C::BlockMatrix
    D::BlockMatrix
    data::Matrix{BlockMatrix}
end

function Base.show(io::IO, ::MIME"text/plain", mpo::MatrixProductOperator)
    chi = size(mpo.A, 1)
    return print(io, "MatrixProductOperator with χ = $chi")
end

function MatrixProductOperator(A::T, B::T, C::T, D::T) where {T}
    data = Matrix{T}(undef, 3, 3)

    # row, col
    data[1, 1] = map(one, D)
    data[1, 2] = C
    data[1, 3] = D

    data[2, 1] = map(zero, B)
    data[2, 2] = A
    data[2, 3] = B

    data[3, 1] = map(zero, D)
    data[3, 2] = map(zero, C)
    data[3, 3] = map(one, D)

    return MatrixProductOperator(A, B, C, D, data)
end

Base.getindex(mpo::MatrixProductOperator, i1, i2) = mpo.data[i1, i2]

function MatrixProductOperator(eqn::LindbladEquation)
    Id = Matrix{ComplexF64}(I, 2, 2)

    chi = 2 * (length(eqn.twolocal) + length(eqn.longrange))
    T = BlockMatrix
    A = T(undef, chi, chi)
    B = T(undef, chi, 1)
    C = T(undef, 1, chi)
    D = T(undef, 1, 1)

    nil = kron(zero(Id), zero(Id))

    fill!(A, nil)
    D[1, 1] = nil

    i = 1

    for term in eqn.twolocal
        X, Y = term.term
        a = term.coeff

        B[i, 1] = -a * im * kron(X, Id)
        B[i + 1, 1] = a * im * kron(Id, transpose(X))

        C[1, i] = kron(Y, Id)
        C[1, i + 1] = kron(Id, transpose(Y))

        i += 2
    end

    for term in eqn.longrange
        X, Y = term.term
        a = term.coeff

        B[i, 1] = -a * im * kron(X, Id)
        B[i + 1, 1] = -a * im * kron(Id, transpose(X))

        C[1, i] = kron(Y, Id)
        C[1, i + 1] = kron(Id, transpose(Y))

        A[i, i] = term.decay * Id
        A[i + 1, i + 1] = term.decay * Id

        i += 2
    end

    # LOCAL TERMS

    D[1, 1] = 1 / 2 * sum(eqn.onelocal; init=D[1, 1]) do term
        H = Matrix{ComplexF64}(term)
        return -im * (kron(H, one(H)) - kron(one(H), transpose(H)))
    end

    D[1, 1] = 1 / 2 * sum(eqn.linops; init=D[1, 1]) do term
        L = Matrix{ComplexF64}(term)
        anticomm = kron(L' * L, one(L)) + kron(one(L), transpose(L' * L))
        return kron(L, transpose(L')) - 1 / 2 * anticomm
    end

    return MatrixProductOperator(A, B, C, D)
end

struct LazyPowerMPO{N}
    data::MatrixProductOperator
end

struct ExtensiveMPO{N}
    mpo::Matrix{BlockMatrix}
    levels::Vector{NTuple{N,Int}}
end

struct dynamicalmpo{N}
    data::Matrix{BlockMatrix}
    levels::Vector{NTuple{N,Int}}
end

function Base.getindex(
    mpo::LazyPowerMPO{N}, row::NTuple{N,Int}, col::NTuple{N,Int}
) where {N}
    H = mpo.data
    return kron((H[r, c] for (r, c) in zip(row, col))...)
end

function dynamicalmpo(eqn::LindbladEquation, dt::Number, order::Integer; kwargs...)
    return dynamicalmpo(MatrixProductOperator(eqn), dt, order; kwargs...)
end

function lazypower(H::MatrixProductOperator, row, col)
    terms = (H[r, c] for (r, c) in zip(row, col))
    return kron(1, terms...)
end

function insertat(level::Vector, pos::Integer, val)
    rv = copy(level)

    insert!(rv, pos, val)

    return rv
end

function dynamicalmpo(
    H::MatrixProductOperator,
    dt::Number,
    order::Integer;
    extend=true,
    compress=true,
    approx=true,
)
    N = order

    levels = Vector{Int}[]

    # Generate the levels
    for tup in Iterators.product(fill(1:3, N)...)
        v = collect(reverse(tup))
        push!(levels, v)
    end

    prodlevels = Iterators.product(levels, levels)

    # Raise H to the power of N
    data = map(i -> lazypower(H, i[1], i[2]), prodlevels)

    # Extension step
    if extend
        for (posa, a) in enumerate(levels), (posb, b) in enumerate(levels)
            # There is a typo in the paper; paper reads do if 1 ∈ b; should read
            # do if 1 ∉ b. As such, the following conditional skips if 1 ∈ b.
            if 1 ∈ b
                continue
            end

            if (2 ∉ a && 3 ∈ a)
                continue
            end

            aes = [insertat(a, pos, 1) for pos in 1:(N + 1)]
            bes = [insertat(b, pos, 3) for pos in 1:(N + 1)]

            for ae in aes, be in bes
                n1 = count(isequal(1), ae)
                n3 = count(isequal(3), be)

                coeff = factorial(N) / (factorial(N + 1) * n1 * n3)
                # @debug "Extending with ae=$ae, be=$be into a=$a, b=$b and n1*n3=$(n1 * n3)"

                data[posa, posb] += lazypower(H, ae, be) * dt * coeff
            end
        end
    end

    # Construct the time evolution MPO
    for a in 1:N
        generator = fill(1, N)
        for i in 1:a
            generator[end + 1 - i] = 3
        end

        @assert length(generator) == N

        P = permutations(generator)

        for lev in P
            if lev ∈ levels
                # get the index corresponding to that particular level
                b = findfirst(isequal(lev), levels)

                # add column b to the first column
                data[:, 1] += (dt^a * factorial(N - a) / factorial(N)) * data[:, b]

                # remove row and column b
                data = data[1:end .!= b, 1:end .!= b]
                deleteat!(levels, b)
            end
        end
    end

    if compress
        data = exactcompression!(levels, data)
    end

    if approx
        data = approxcompression!(levels, data, dt)
    end

    return data
end

function curly(mpo, args...)
    P = collect(permutations(args))
    S = Tuple(Set(P))

    func = term -> kron(1, term...)

    mats = map(term -> map(symb -> getfield(mpo, symb), term), S)

    return sum(func, mats)
end

topower(H::MatrixProductOperator, N) = topower(LazyPowerMPO{N}(H))

function topower(H::LazyPowerMPO{N}) where {N}
    # RaisedMPO{N}
    levels = NTuple{N,Int}[]

    mat = Matrix{BlockMatrix}(undef, 3^N, 3^N)

    for tup in Iterators.product(fill(1:3, N)...)
        push!(levels, reverse(tup))
    end

    mat = map(x -> H[x[1], x[2]], Iterators.product(levels, levels))

    return ExtensiveMPO(mat, levels)
end

function exactcompression!(levels, data)
    for (c, lev) in enumerate(levels)
        # sc, n1, sr, n3 = sort_1_3(lev, levels)

        k1 = map(x -> isequal(x, 1) ? 2 : 1, lev)
        k3 = map(x -> isequal(x, 3) ? 2 : 1, lev)

        sc = findfirst(isequal(lev[sortperm(k1)]), levels)
        sr = findfirst(isequal(lev[sortperm(k3)]), levels)

        n1 = count(isequal(1), lev)
        n3 = count(isequal(3), lev)

        if n3 ≤ n1 && sc != c
            data[sc, :] += data[c, :]

            @debug "Removing $lev"

            data = data[1:end .!= c, 1:end .!= c]
            deleteat!(levels, c)

        elseif n3 > n1 && sr != c
            data[:, sr] += data[:, c]

            @debug "Removing $lev"

            data = data[1:end .!= c, 1:end .!= c]
            deleteat!(levels, c)
        end
    end
    return data
end

function approxcompression!(levels::Vector{Vector{Int}}, data, dt)
    for (ac, alev) in enumerate(levels)
        N = length(alev)

        n3 = count(isequal(3), alev)
        (1 in alev || n3 == 0) && continue

        blev = [isequal(i, 3) ? 1 : i for i in alev]

        bc = findfirst(isequal(blev), levels)

        data[:, bc] += data[:, ac] * dt^n3 * factorial(N - n3) / factorial(N)

        data = data[1:end .!= ac, 1:end .!= ac]
        deleteat!(levels, ac)
    end
    return data
end

function remove_row_and_column(H::ExtensiveMPO, c)
    levels = H.levels
    i = findfirst(isequal(c), levels)
    new_data = H.mpo[1:end .!= i, 1:end .!= i]
    new_layers = deleteat!(copy(levels), i)
    return ExtensiveMPO(new_data, new_layers)
end

function sort_1_3(c, levels)
    sc = Int[]
    sc_rest = Int[]

    sr = Int[]
    sr_rest = Int[]
    for i in c
        if i == 1
            push!(sc, 1)
            push!(sr_rest, i)
        elseif i == 3
            push!(sr, 3)
            push!(sc_rest, i)
        else
            push!(sr_rest, i)
            push!(sc_rest, i)
        end
    end
    n1 = length(sc)
    n3 = length(sr)

    append!(sc, sc_rest)
    append!(sr, sr_rest)

    sc = findfirst(isequal(Tuple(sc)), levels)
    sr = findfirst(isequal(Tuple(sr)), levels)

    return sc, n1, sr, n3
end

function isingthermal()
    _, _, Z = PAULI

    Id = one(Z)
    hk = kron(Z, Id)
    hb = kron(Id, Z)

    T = BlockMatrix

    A = T(undef, 2, 2)
    B = T(undef, 2, 1)
    C = T(undef, 1, 2)
    D = T(undef, 1, 1)

    B[1, 1] = C[1, 1] = 1 / 2 * hk
    B[2, 1] = C[1, 2] = 1 / 2 * hb

    A .= Ref(zero(hk))
    D .= Ref(zero(hk))

    return MatrixProductOperator(A,B,C,D)
end

function totensor(mpo::dynamicalmpo)
    nr = sum(x -> size(x, 1), mpo.data)
    nc = sum(x -> size(x, 1), mpo.data)
    return (nr, nc)
end

####

locality(term::AbstractTerm) = length(term.data)

LinearAlgebra.ishermitian(term::AbstractTerm) = ishermitian(kron(term.data...))

function hamterm(args...)
    return HamiltonianTerm(Matrix{ComplexF64}([args...]))
end

struct SimpleModel <: AbstractModel
    oneloc::Vector{Matrix{ComplexF64}}
    twoloc::Vector{Matrix{ComplexF64}}
    lin::Vector{Matrix{ComplexF64}}
end
SimpleModel() = SimpleModel([], [], [])

struct ModelData <: AbstractModel
    xdata::Array{ComplexF64,6}
    ydata::Array{ComplexF64,6}
end

macro model(ex...)
    rv = quote
        xd, yd = @infliouvillian $ex
        Model(xd, yd)
    end
    return rv
end

macro defmodel(rules_ex, equation_ex...)
    return defmodel(rules_ex, equation_ex...)
end

function defmodel(rules_ex::Expr, equation_ex...)
    rules_block = MacroTools.postwalk(rules_ex) do x
        @capture(x, A_ => B_) || return x
        return :($A = $(esc(B)))
    end
    rules_block.head = :block

    rules_tuple = MacroTools.postwalk(rules_ex) do x
        @capture(x, A_ => B_) || return x
        return :($(string(A)) * " = " * (string($(esc(B)))))
        # return :($(string(A)) => $(esc(B)))
    end
    rules_tuple.head = :vcat

    str_param = string.(rules_tuple.args)
    str_H = string(unblock(equation_ex[1]))
    str_Ls = [string.(unblock(equation_ex[2:end]))...]

    rv = quote
        let
            $rules_block
            data_x, data_y = @infliouvillian($(equation_ex[1]), $(equation_ex[2:end]...))
            Model(data_x, data_y, $rules_tuple, $str_H, $str_Ls)
        end
    end
    return rv
end
function Base.show(io::IO, model::Model)
    if all(x -> isdefined(model, x), fieldnames(Model))
        params_to_print = []

        ham = model.hamiltonian

        for param_str in model.param
            param = Meta.parse(param_str)

            if param.args[2] == 0
                rem = "(?>" * string(param.args[1]) * ")"
                re = "($(rem).*?)([\\+\\-](?![^\\[]*\\])|\$)\\s*"
                ham = replace(ham, Regex(re) => "")
            else
                push!(params_to_print, param_str)
            end
        end

        # Remove trailing +/-
        ham = replace(ham, r"[\+\-]\s*$" => "")

        println("Liouvillian data of type Matrix{ComplexF64} generated from:")
        println(io, "  Hamiltonian:")
        println("    ", ham)
        println(io, "  Lindblad operators: ")
        foreach(iv -> println("    ", iv[1], ". ", iv[2]), enumerate(model.linops))
        println("  where:")
        foreach(x -> println("    ", x), params_to_print)
    else
        println("Unknown model data of type Matrix{ComplexF64}:")
        show(io, model.xdata)
        show(io, model.ydata)
    end
    return nothing
end

function lrtno()
    # 1 -> do nothing 2-> conventional path 3-> unconventional path 4-> fin
    # (L,D) -> (R,U)
    # -----------
    # 1,1,1,1 = I
    # 1,2,1,2 = I
    # 2 1 2 4 = I
    # 1 4 1 4 = I
    # 4 1 4 4 = I
    # -----------
    # 1 1 1 2 = A
    # 1 1 2 4 = A
    # 1 2 2 4 = I
    # 1 2 4 4 = B
    # 2 1 4 4 = B
    # -----------
    # 3 1 3 4 = I
    # 3 2 4 4 = I
    # 1 1 3 4 = B
    # -----------
    # TE (L,D) -> (R,U)
    # -----------
    # 1,1,1,1 = I
    # 1,2,1,2 = I
    # 2 1 2 4 = I
    # 1 4 1 4 = I
    # 4 1 4 4 = I
    # -----------
    # 1 1 1 2 = A
    # 1 1 2 4 = A
    # 1 2 2 4 = I
    # 1 2 4 4 = B
    # 2 1 4 4 = B
    # -----------
    # 3 1 3 4 = I
    # 3 2 4 4 = I
    # 1 1 3 4 = B
    # -----------
end

macro nterm(expr)
    @capture(expr, NTuple{Num_,TermType_})
    termtypes = [Expr(:curly, TermType, N) for N in 1:Num]
    vectortypes = map(termtypes) do term
        return Expr(:curly, :Vector, term)
    end
    return Expr(:curly, :Tuple, vectortypes...)
end
