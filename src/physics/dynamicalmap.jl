function trottergates(data, pspace, dt; trunc=truncerr(1e-8))
    data_exp = exp.(dt .* data)::typeof(data)

    ket = pspace

    ketp = circshift(ket, (-1, 0))

    # Bra1 Bra2, Ket1, Ket2
    space = @. adjoint(ket) * adjoint(ketp) * ket * ketp

    tensors = TensorMap.(data_exp, space, space)
    # im an idiot and got bra-ket swapped round. Everything assumes ket,bra so need to swap...
    ptensors = permute.(tensors, Ref((3, 4, 1, 2)), Ref((7, 8, 5, 6)))
    ptensors = permute.(ptensors, Ref((1, 3, 5, 7)), Ref((2, 4, 6, 8)))

    out = tsvd.(ptensors; trunc=trunc)

    u0, s0, v0 = out[1, 1]

    # TODO: find a better way to fix this source of type instabilty
    u = getindex.(out, 1)::Lattice{Square,typeof(u0),Matrix{typeof(u0)}}
    s = getindex.(out, 2)::Lattice{Square,typeof(s0),Matrix{typeof(s0)}}
    v = getindex.(out, 3)::Lattice{Square,typeof(v0),Matrix{typeof(v0)}}

    err = getindex.(out, 4)

    @info "Trotter compression error: $err with bond dim = $(dim.(codomain.(s)))"

    left = u .* sqrt.(s) # east 
    right = circshift(sqrt.(s) .* v, (1, 0)) # west

    return TrotterGates(left, _transpose.(right))
end

function sum_twoloc(model::SimpleModel, k1, k2, dt; kwargs...)
    return sum_twoloc(model.twoloc, k1, k2, dt; kwargs...)
end
function sum_twoloc(data, k1, k2, dt; trunc=truncerr(1e-8))
    d1 = dim(k1)^2
    d2 = dim(k2)^2

    nil = zeros(d1 * d2, d1 * d2)

    # b,k ⊗ b,k
    twogate = exp(dt * +(nil, data...))::Matrix

    # b1 k1, b2, k2
    space = adjoint(k1) * k1 * adjoint(k2) * k2

    twot = TensorMap(twogate, space, space)
    # im an idiot and got bra-ket swapped round. Everything assumes ket,bra so need to swap...
    # now k1,b1,k2,b2
    twot = permute(twot, (2, 1, 4, 3), (6, 5, 8, 7))
    twot = permute(twot, (1, 2, 5, 6), (3, 4, 7, 8))

    out = tsvd(twot; trunc=trunc)

    u, s, v, err = out

    @info "Trotter compression error: $err with bond dim = $(dim(codomain(s)))"

    site1 = u * sqrt(s)
    site2 = sqrt(s) * v

    return site1, _transpose(site2)
end

function trottergates(
    model::AbstractMatrix{SimpleModel}, local_space::AbstractLattice, dt; kwargs...
)
    k1 = local_space
    k2 = circshift(local_space, (-1, 0))

    locs = sum_oneloc.(model, local_space, dt)

    left_right = sum_twoloc.(model, k1, k2, dt; kwargs...) # left - right

    lefts = getindex.(left_right, 1)
    rights = circshift(getindex.(left_right, 1), (1, 0))

    return SimpleGates(rights, lefts, locs)
end

sum_oneloc(model, k, dt) = sum_oneloc(model.oneloc, model.lin, k, dt)
function sum_oneloc(ham, lin, k, dt)
    d = dim(k)^2
    nil = zeros(d, d)
    # b,k
    # divide by two for symmetry when applying across both x and y
    onegate = exp(dt / 2 * +(nil, ham..., lin...))::Matrix
    println(onegate)

    # b1 k1, b2, k2
    space = adjoint(k) * k

    onet = TensorMap(onegate, space, space)
    # im an idiot and got bra-ket swapped round. Everything assumes ket,bra so need to swap...
    # now k,b
    onet = permute(onet, (2, 1), (4, 3))

    return onet
end

function wiimpo(data::ModelMPO, dt)
    A = data.A
    B = data.B
    C = data.C
    D = data.D

    f! = factorial

    # M = Matrix{promote_type(eltype(A), eltype(B), eltype(C), eltype(D)}(zeros)

    M[:, :, :, :, 1, 1] = one(D) + dt * D + dt^2 / f!(2)
    M[:, :, :, :, 2, 1] = dt * B + dt^2 / f!(2) * sum_all_perms(B, D)
    M[:, :, :, :, 1, 2] = C + dt / 2 * sum_all_perms(C, D)
    M[:, :, :, :, 2, 2] = A + dt / 2 * (sum_all_perms(A, D) + sum_all_perms(B, C))

    return nothing
end

# function onesite!(mpo::WIIMPO, coeff, f)
#     f_k = kron(idn, f)
#     f_b = kron(permutedims(f), idn)
#     push!(mpo.D, im * coeff * (f_b - f_k))
#     return mpo
# end

function ham!(model::SimpleModel, coeff, g)
    if coeff == 0
        return model
    end
    push!(model.oneloc, onesite(coeff, g))
    return model
end
function ham!(model::SimpleModel, coeff, g, h)
    if coeff == 0
        return model
    end
    push!(model.twoloc, twosite(coeff, g, h))
    return model
end
function lin!(model::SimpleModel, coeff, l)
    if coeff == 0
        return model
    end
    push!(model.lin, lin(coeff, l))
    return model
end

function quickxyz(; Jx=0.0, Jy=0.0, Jz=1.0, hx=0.0, hy=0.0, hz=0.0, Γ=1.0)
    model = SimpleModel()

    ham!(model, Jx / 4, σ1, σ1)
    ham!(model, Jy / 4, σ2, σ2)
    ham!(model, Jz / 4, σ3, σ3)
    ham!(model, hx, σ1)
    ham!(model, hy, σ2)
    ham!(model, hz, σ3)
    lin!(model, Γ, σm)

    return model
end
function quickexact(V)
    model = SimpleModel()

    ham!(model, V / 4, σ3, σ3)
    # ham!(model, 0.5, σ2)
    lin!(model, 1 / 2, σ2 - im * σ3)

    return model
end

function onesite(coeff, g::T) where {T<:AbstractMatrix}
    idn = one(g)

    ke = x -> kron(idn, x)
    br = x -> kron(permutedims(x), idn)

    h = -im * coeff * (ke(g) - br(g))
    return h
end

# function twosite!(mpo::WIIMPO, coeff, f, g=f; lr=0)
#     idn = one(f)
#
#     f_k = kron(idn, f)
#     f_b = kron(permutedims(f), idn)
#
#     g_k = -im * coeff * kron(idn, g)
#     g_b = im * coeff * kron(permutedims(f), idn)
#
#     push!(mpo.C, f_k, f_b)
#     push!(mpo.B, g_k, g_b)
#
#     # push!(mpo.A, lr * )
#
#     return mpo
# end

function twosite(coeff, g::T, h::T) where {T<:AbstractMatrix}
    if size(g) != size(h)
        throw(DimensionMismatch())
    end

    idn = one(g)

    ke = x -> kron(idn, x)
    br = x -> kron(permutedims(x), idn)

    H = -im * coeff * (kron(ke(g), ke(h)) - kron(br(g), br(h)))
    return H
end

# adds lindblad operator coeff * l
function lin(coeff, l)
    idn = one(l)

    diss =
        1 / 2 *
        coeff^2 *
        (2 * kron(conj(l), l) - (kron(idn, l' * l) + kron(permutedims(l) * conj(l), idn)))

    return diss
end

# function lin!(mpo::WIIMPO, coeff, f)
#     idn = one(f)
#
#     diss =
#         kron(conj(f), permutedims(f)) -
#         0.5 * (kron(idn, f' * f) + kron(permutedims(f) * conj(f), idn))
#
#     return push!(mpo.D, coeff * diss)
# end

function sum_all_perms(args::NTuple{N}) where {N}
    perms = unique(permutations(args))
    return splat(+)(map(splat(*), perms))
end

function thermalising()
    Z = Matrix{ComplexF64}([1 0; 0 -1])
    ZZ = kron(Z, Z)
    return kron(ZZ, one(ZZ)) + kron(one(ZZ), ZZ)
end

function dynamicalmap(model::AbstractModel, space::AbstractLattice, evoalg::TEBD; kwargs...)
    tau = evoalg.timestep / evoalg.order

    xdata = fill(model.xdata, size(space))
    ydata = fill(model.ydata, size(space))

    xmap = trottergates(xdata, space, tau; kwargs...)
    ymap = trottergates(ydata, space, tau; kwargs...)

    return DynamicalMap(xmap, ymap)
end
function dynamicalmap(model::SimpleModel, space::AbstractLattice, evoalg::TEBD; kwargs...)
    tau = evoalg.timestep / evoalg.order

    xdata = fill(model, size(space))
    ydata = fill(model, size(space))

    xmap = trottergates(xdata, space, tau; kwargs...)
    ymap = trottergates(ydata, space, tau; kwargs...)

    return DynamicalMap(xmap, ymap)
end

function dynamicalmap(model::ModelData, space::AbstractLattice, evoalg::WII; kwargs...)
    (ex, wx) = size(model.xdata)[(end - 1):end]
    (ey, wy) = size(model.ydata)[(end - 1):end]

    xmap = broadcast(space) do sp
        return mpodata_to_tensor(model.xdata, sp, ex, wx)
    end

    ymap = broadcast(space) do sp
        return mpodata_to_tensor(model.ydata, sp, ey, wy)
    end

    return DynamicalMap(WIIMPO(xmap), WIIMPO(ymap))
end

function dynamicalmap(model::ModelData, space::AbstractLattice, evoalg::TNO; kwargs...)
    (ex, wx) = size(model.xdata)[(end - 1):end]
    (ey, wy) = size(model.ydata)[(end - 1):end]

    xmap = mpodata_to_tensor.(Ref(model.xdata), space, Ref(ex), Ref(wx))
    ymap = mpodata_to_tensor.(Ref(model.ydata), space, Ref(ey), Ref(wy))

    p = concat_map.(xmap, ymap)

    return DynamicalMap(WIIPEPO(p))
end

function concat_map(x, y)
    @tensoropt p[k1 b1 k2 b2; e s w n] := y[k1 b1 k b; s n] * x[k b k2 b2; e w]
end

function mpodata_to_tensor(model_data, space, chi_e, chi_w)
    # A space with dimension one
    unit = oneunit(space)

    east = ⊕((unit^chi_e)...)
    west = ⊕((unit^chi_w)...)

    cod = space * space' * space' * space
    dom = east * west'

    return TensorMap(model_data, cod, dom)
end

function exactwii(dt; V=1.0, Γ=1.0)
    Z = σ3
    l = sqrt(Γ) / 2 * (σ2 - im * σ3)
    #
    L = (l)
    Lc = (conj(l))
    Lt = (permutedims(l))
    Ld = (conj(permutedims(l)))
    #
    Id = one(Z)
    nil = zero(Z)

    T = Matrix{Matrix{ComplexF64}}

    A = T(undef, 2, 2)
    B = T(undef, 2, 1)
    C = T(undef, 1, 2)
    D = T(undef, 1, 1)

    A .= Ref(kron(nil, nil))

    B[1, 1] = -V / 4 * im * kron(Z, Id)
    B[2, 1] = V / 4 * im * kron(Id, Z)

    C[1, 1] = kron(Z, Id)
    C[1, 2] = kron(Id, Z)

    D[1, 1] = 1 / 2 * (kron(L, Lc) - 1 / 2 * (kron(Ld * L, Id) + kron(Id, Lt * Lc)))

    mpo = blockmul(A, B, C, D, dt)
    array = zeros(ComplexF64, 2, 2, 2, 2, size(mpo)...)

    for I in CartesianIndices(mpo)
        # in: k,b out: k b bond: west east
        array[:, :, :, :, I[2], I[1]] .= reshape(mpo[I], (2, 2, 2, 2))
    end

    return ModelData(array, array)
end

function _longrangewii(dt; V=1.0, Γ=1.0, λ=0.0)
    Z = σ3
    l = sqrt(Γ) / 2 * (σ2 - im * σ3)
    #
    L = (l)
    Lc = (conj(l))
    Lt = (permutedims(l))
    Ld = (conj(permutedims(l)))
    #
    Id = one(Z)
    nil = zero(Z)

    T = Matrix{Matrix{ComplexF64}}

    A = T(undef, 2, 2)
    B = T(undef, 2, 1)
    C = T(undef, 1, 2)
    D = T(undef, 1, 1)

    A[1, 1] = A[2, 2] = λ * kron(Id, Id)
    A[2, 1] = A[1, 2] = λ * kron(nil, nil)

    B[1, 1] = -V / 4 * im * kron(Z, Id)
    B[2, 1] = V / 4 * im * kron(Id, Z)

    C[1, 1] = kron(Z, Id)
    C[1, 2] = kron(Id, Z)

    D[1, 1] = 1 / 2 * (kron(L, Lc) - 1 / 2 * (kron(Ld * L, Id) + kron(Id, Lt * Lc)))

    mpo = blockmul(A, B, C, D, dt)
    array = zeros(ComplexF64, 2, 2, 2, 2, size(mpo)...)

    for I in CartesianIndices(mpo)
        # in: k,b out: k b bond: west east
        array[:, :, :, :, I[2], I[1]] .= reshape(mpo[I], (2, 2, 2, 2))
    end

    return array
end
function xyzwii(dt; Jx=0.5, Jy=1.2, Jz=1.0, Γ=1.0)
    X = σ1
    Y = σ2
    Z = σ3
    l = sqrt(Γ) * σm
    #
    L = (l)
    Lc = (conj(l))
    Lt = (permutedims(l))
    Ld = (conj(permutedims(l)))
    #
    Id = one(Z)
    nil = zero(Z)

    T = Matrix{Matrix{ComplexF64}}

    A = T(undef, 6, 6)
    B = T(undef, 6, 1)
    C = T(undef, 1, 6)
    D = T(undef, 1, 1)

    A .= Ref(kron(nil, nil))

    B[1, 1] = -Jx * im * kron(X, Id)
    B[2, 1] = Jx * im * kron(Id, X)
    B[3, 1] = -Jy * im * kron(Y, Id)
    B[4, 1] = Jy * im * kron(Id, Y)
    B[5, 1] = -Jz * im * kron(Z, Id)
    B[6, 1] = Jz * im * kron(Id, Z)

    C[1, 1] = kron(X, Id)
    C[1, 2] = kron(Id, X)
    C[1, 3] = kron(Y, Id)
    C[1, 4] = kron(Id, Y)
    C[1, 5] = kron(Z, Id)
    C[1, 6] = kron(Id, Z)

    # div by half cos 2d?
    D[1, 1] = 1 / 2 * (kron(L, Lc) - 1 / 2 * (kron(Ld * L, Id) + kron(Id, Lt * Lc)))

    mpo = blockmul(A, B, C, D, dt)
    array = zeros(ComplexF64, 2, 2, 2, 2, size(mpo)...)

    for I in CartesianIndices(mpo)
        # in: k,b out: k b bond: west east
        array[:, :, :, :, I[2], I[1]] .= reshape(mpo[I], (2, 2, 2, 2))
    end

    return A,B,C,D

    # return ModelData(array, array)
end

function longrangewii(dt; V=1.0, Γ=1.0, λx=0.0, λy=1.0)
    xarr = _longrangewii(dt; V=V, Γ=Γ, λ=λx)
    yarr = _longrangewii(dt; V=V, Γ=Γ, λ=λy)
    return ModelData(xarr, yarr)
end

function blockmul(A, B, C, D, dt)
    chi = length(B)

    mpo = similar(A, (1 + chi, 1 + chi))

    mpo[1:1, 1:1] = one.(D) + dt * D + dt^2 / 2 * kron(D, D) # 1 by 1
    mpo[1:1, 2:end] = C + dt / 2 * (kron(C, D) + kron(D, C)) # 1 by chi
    mpo[2:end, 1:1] = dt * B + dt^2 / 2 * (kron(B, D) + kron(D, B)) # chi by 1
    mpo[2:end, 2:end] = A + dt / 2 * (kron(A, D) + kron(D, A) + kron(B, C) + kron(C, B)) # chi by chi

    return mpo
end

function makempo(s)
    chi = length(B)

    d = dim(s)

    return data = zeros(ComplexF64, d, d, d, d, chi, chi)#Order: in;k b' out;k' b' bond; east west'
end

function thermalmpo(dx)
    βc = log(1 + sqrt(2)) / 2

    Z = σ3
    Id = one(Z)
    hk = kron(Z, Id)
    hb = kron(Id, Z)

    T = Matrix{Matrix{ComplexF64}}

    A = T(undef, 2, 2)
    B = T(undef, 2, 1)
    C = T(undef, 1, 2)
    D = T(undef, 1, 1)

    B[1, 1] = C[1, 1] = hk
    B[2, 1] = C[1, 2] = hb

    A .= Ref(zero(hk))
    D .= Ref(zero(hk))

    mpo = blockmul(A, B, C, D, dx * βc / 2)
    # array = Matrix{Array{ComplexF64,4}}(undef, size(mpo))
    array = zeros(ComplexF64, 2, 2, 2, 2, size(mpo)...)
    for I in CartesianIndices(mpo)
        # in: k,b out: k b bond: west east
        array[:, :, :, :, I[2], I[1]] .= reshape(mpo[I], (2, 2, 2, 2))
    end

    s = TensorKit.ℂ^2
    d = TensorKit.ℂ^3

    t = TensorMap(array, s * s' * s' * s, d * d')

    return ModelData(array, array)
end
