#=
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

              ┌─────────────────────┐
              │◄─────5       1◄─────┤
              │        u s v        │
              │◄─────6       2◄─────┤
              │                     │
              │                     │
              │                     │
              ├─────►3       7─────►│
              │       u' s' v'      │
              ├─────►4       8─────►│
              └─────────────────────┘
=#
function isometry_update!(
    vpn::AbstractTensorMap{S,4,2},
    vpe::AbstractTensorMap{S,5,3},
    u::AbstractTensorMap{S,2,1},
    s::AbstractTensorMap{S,1,1},
    v::AbstractTensorMap{S,1,2},
) where {S<:IndexSpace}
    u = u * sqrt(s)
    v = sqrt(s) * v
    @tensoropt vpn[b k -3 -4; -5 -6] = vpe[b k 3 4 -4; 6 7 -6] * v[-3; 3 4] * u[6 7; -5]
    return vpn
end

function isometry_update!(
    pn::AbstractTensorMap{S,3,3},
    pe::AbstractTensorMap{S,4,4},
    u::AbstractTensorMap{S,2,1},
    s::AbstractTensorMap{S,1,1},
    v::AbstractTensorMap{S,1,2},
) where {S<:IndexSpace}
    u = u * sqrt(s)
    v = sqrt(s) * v
    @tensoropt pn[k -3 -4; b -5 -6] = pe[k 3 4 -4; b 6 7 -6] * v[-3; 3 4] * u[6 7; -5]
    return pn
end

# NEW INDEX CONVENTION; NO COMPOSITE INDICES.
function isometryupdate!(
    p::AbstractTensorMap{S,3,3},
    pe::AbstractTensorMap{S,3,3},
    u::AbstractTensorMap{S,1,1},
    v::AbstractTensorMap{S,1,1},
) where {S<:IndexSpace}
    @tensoropt p[-1 -2 k; -4 -5 b] = pe[1 -2 k; 4 -5 b] * v[-1; 1] * u[4; -4]
    return p
end

function fidelity(
    γ::AbstractTensorMap{S,2,2},
    u::AbstractTensorMap{S,1,1},
    s::AbstractTensorMap{S,1,1},
    v::AbstractTensorMap{S,1,1},
) where {S<:IndexSpace}
    @tensor trϕϕ =
        γ[1 2; 3 4] *
        u[3; a1] *
        s[a1; a2] *
        v[a2; 1] *
        (v')[4; b2] *
        (s')[b2; b1] *
        (u')[b1; 2]
    @tensor trρρ = γ[1 2; 1 2]
    @tensor trρϕ = γ[1 3; 1 4] * (u')[a; 3] * (s')[b; a] * (v')[4; b]
    return real(trρϕ / sqrt(trϕϕ * trρρ))
end

function purity(bondenv)
    return @tensor trρρ = bondenv[1 2 3 4; 1 2 3 4]
end

function constructG(L, g, R)
    @tensoropt G[1 2; 3 4] := L[3; 3] * R[4; 4] * g[1 2; 3 4] * (L')[1; 1] * (R')[2; 2]
end

function isometryinit(metric::AbstractOnLattice)
    leftright = isometryinit.(metric)

    left = getindex.(leftright, 1)
    right = getindex.(leftright, 1)

    return left, right
end

# Isometry update requires one to contruct initial guesses for L & R.
function isometryinit(metric::AbsTen{2,2})
    η_left = domain(metric)[1]
    η_right = codomain(metric)[1]

    left = id(η_left)
    right = id(η_right)

    return left, right
end

function metricoptimise(left, right, metric::AbstractOnLattice, trunc; kwargs...)
    # TODO: better initialisation
    nx, ny = size(metric)

    u = Matrix(undef, nx, ny)
    v = Matrix(undef, nx, ny)
    s = Matrix(undef, nx, ny)

    # TODO: mulithreading
    for y in axes(metric, 2)
        for x in axes(metric, 1)
            u[x, y], s[x, y], v[mod(x + 1, 1:nx)y] = metricoptimise(
                left[x, y], right[x, y], metric[x, y], trunc; kwargs...
            )
        end
    end
    return u, s, v
end

# Optimise the isometries `left` and `right` with respect to `metric`.
function metricoptimise(left, right, metric::AbsTen{2,2}, trunc; kwargs...)
    # Convenience
    L = left
    R = right

    LR = permute(L * R, (1, 2), ())

    # But in the appropriate matrix form for the Ritz procedure. 
    Y = permute(metric, (2, 4), (3, 1))

    # Check if the metric is hermitian, If not, and use SU
    hermiticity = norm(Y - Y')

    if hermiticity > 1e-8
        @info "Environment not hermitian: using SU"
        u, s, v = tsvd(left * right; trunc=trunc)
        return u, s, v
    end

    # Obtain the normalisation constant.
    F0 = scalar(LR' * Y * LR)

    # Symmetrise and normalise the truncation metric.
    Y = (0.5 * (Y + Y')) / F0

    # Construct the cost function to measure convergence.
    cf = (Ma, Mb) -> costfunc(Ma, Mb, Y, left, right, F0)

    # Proceed with metric optimisation.
    u, s, v = optiso(cf, L, R, Y, trunc; kwargs...)

    return u, s, v
end

function _costfunc(Ma, Mb, g, Ra, Rb)
    left = (Ma * Mb - Ra * Rb)'
    right = left'
    @tensor err = left[b; a] * g[a b; c d] * right[c; d]
end

function costfunc(Ma, Mb, g, Ra, Rb, F0)
    return real(sqrt(_costfunc(Ma, Mb, g, Ra, Rb) / F0))
end

function optiso(cf, left, right, metric, trunc; EAT=false, maxiter=100)
    # Convenience
    Y = metric
    L = left
    R = right

    # The following tensor will be useful...
    LR = permute(L * R, (1, 2), ())
    Y_LR = Y * LR

    # Perform EAT if we want to.
    if EAT
        Ma0, Mb0 = eat(L, R, metric, trunc)
    else
        Ma0, Mb0 = L, R
    end

    # Calculate the initial tolerance
    ϵ0 = cf(Ma0, Mb0)

    # Initialise isometries for optimisation.
    u0, s0, Mb0 = tsvd(Ma0 * Mb0; trunc=trunc)

    # Alias for iteration.
    u, s, Mb = u0, s0, Mb0

    # Get an appropriate identity matrix for the spaces.
    id_a = id(codomain(u))
    id_b = transpose(id(domain(Mb)))

    counter = 0
    δϵ = 1.0
    ϵ = ϵ0
    print("Optimising isometries: [")
    while !(counter > maxiter || δϵ < 1e-10)
        ϵ_old = ϵ

        # Optimise for Ma with Mb fixed.
        Ma = u * s
        Ma_max = optsingleiso(id_a ⊗ transpose(Mb), Y, Y_LR)
        Ma, s, v = tsvd(Ma_max * Mb; trunc=trunc)

        # Optimise for Mb with Ma fixed.
        Mb = s * v
        Mb_max = optsingleiso(Ma ⊗ id_b, Y, Y_LR)
        u, s, Mb = tsvd(Ma * Mb_max; trunc=trunc)

        # Calculate truncation error.
        ϵ = cf(u * sqrt(s), sqrt(s) * Mb)

        # Calculate convergence measure.
        δϵ = abs(ϵ - ϵ_old)

        print("|")

        counter = counter + 1
    end
    print("]\n")

    # Return new isometries for transformation to the WTG
    if ϵ0 < ϵ
        @info "Isometries could not be improved from a tolerance of $ϵ0"
        return u0, s0, Mb0
    else
        @info "Isometries optimised to a tolerance of: $ϵ from $ϵ0 after $counter iterations."
        return u, s, Mb
    end
end

function optsingleiso(id_Mb, Y, Y_LR)
    temp = id_Mb' * Y

    g = id_Mb' * Y * id_Mb
    j = id_Mb' * Y_LR

    Ma_max = pinv(g; rtol=1e-10) * j

    # ϵ_j = scalar(Ma_max' * j)
    # ϵ = scalar(Ma_max' * g * Ma_max) - ϵ_j - ϵ_j' + ϵ0

    return permute(Ma_max, (1,), (2,)) # , abs(ϵ)
end

function eat(L, R, g, trunc)
    L = TensorMap(L.data, codomain(L), fuse(*(domain(L)...)))
    R = TensorMap(R.data, fuse(*(codomain(R)...)), domain(R))
    @tensoropt G[1 2; 3 4] := L[3; 3] * R[4; 4] * g[1 2; 3 4] * (L')[1; 1] * (R')[2; 2]

    Ga, _, Gb = tsvd(G, (3, 1), (4, 2); trunc=truncdim(1))

    Gap = permute(Ga, (2,), (1, 3))
    Ga = TensorMap(Gap.data, codomain(Gap), fuse(*(domain(Gap)...)))
    Gbp = permute(Gb, (2,), (1, 3))
    Gb = TensorMap(Gbp.data, codomain(Gbp), fuse(*(domain(Gbp)...)))

    λa, Ua = eigh(Ga)
    λb, Ub = eigh(Gb)

    Wa, λ, Wb = tsvd(sqrt(λa) * Ua * Ub * sqrt(λb); trunc=trunc)

    try
        Ma = L * Ua' * 1 / sqrt(λa) * Wa * sqrt(λ)
        Mb = sqrt(λ) * Wb * 1 / sqrt(λb) * Ub' * R
        @info "EAT success"
        return Ma, Mb
    catch
        @info "EAT failed"
        return L, R
    end
end

function optimise_isometries(u, s, v, t_metric, trunc; maxiter=50)
    Y = permute(t_metric, (2, 4), (3, 1))
    Y = 0.5 * (Y + Y') / tr(Y)
    γ = permute(Y, (4, 1), (3, 2))

    # println("Initial fidelity:  ", fidelity(γ, u, s, v))

    # u = TensorMap(rand, ComplexF64, codomain(u), domain(u))
    # v = TensorMap(rand, ComplexF64, codomain(v), domain(v))

    δu = 1.0
    δv = 1.0
    counter = 0
    while !(counter > maxiter || max(δu, δv) < 1e-6)
        up = u
        vp = v

        @tensor B1[1 -2; -3 4] := γ[1 2; 3 4] * u[3; -3] * (u')[-2; 2]

        @tensor P1[-2; 4] := γ[x 2; x 4] * (u')[-2; 2]

        B1inv = pinv(permute(B1, (3, 1), (2, 4)))
        R1 = permute(permute(P1, (), (1, 2)) * B1inv, (1,), (2,))
        u, s, v, err = tsvd(normalize(u * R1); trunc=trunc)

        @tensor B2[-1 2; 3 -4] := γ[1 2; 3 4] * v[-1; 1] * (v')[4; -4]

        @tensor P2[2; -4] := γ[x 2; x 4] * (v')[4; -4]

        B2inv = pinv(permute(B2, (3, 1), (2, 4)))
        R2 = permute(permute(P2, (), (1, 2)) * B2inv, (1,), (2,))
        u, s, v, err = tsvd(normalize(R2 * v); trunc=trunc)

        counter = counter + 1

        δu = sqrt(real(tr((u - up) * (u - up)')))
        δv = sqrt(real(tr((v - vp) * (v - vp)')))
        # println("du: ", δu, "\t dv: ", δv)
        # println("Fidelity:  ", fidelity(γ, u, s, v))
        print("|")
    end
    println("")
    # println("Final fidelity:  ", fidelity(γ, u, s, v))
    return u, s, v
end

#TODO: Test this and clean once verified.

# function optimise_isometries(γ::AbstractTensorMap{<:IndexSpace,2,2}, D; maxiter=100)
#     ntsp = codomain(γ)[1]
#
#     if dim(ntsp) < dim(D)
#         u, s, v = tsvd(id(D); trunc=truncspace(ntsp))
#         return v, id(D), u
#     end
#
#     σ = id(ntsp)
#
#     # @tensor N  = γ[x y; y x] 
#     # γ = γ * 1/N
#
#     u, s, v = tsvd(σ; trunc=truncspace(D))
#     u = TensorMap(rand, eltype(γ), ntsp, D)
#     v = TensorMap(rand, eltype(γ), D, ntsp)
#
#     # u = isometry(Matrix{eltype(γ)}, ntsp, D)
#     # v = permute(isometry(Matrix{eltype(γ)}, ntsp', D'),(2,),(1,))
#     # println("F0: ", _fidelity(γ, u, s, v))
#
#     # this is awfully confusing
#     δu = 1.0
#     δv = 1.0
#     counter = 0
#
#     print("FET: ")
#
#     while !(counter > maxiter || max(δu, δv) < 1e-10)
#         up = u
#         vp = v
#
#         # @tensor B1[1 2 -3; -4 7 8] := γ[1 2 3 4; 5 6 7 8] * u[5 6; -4] * (u')[-3; 3 4]
#         @tensor B1[1 -2; -3 4] := γ[1 2; 3 4] * u[3; -3] * (u')[-2; 2]
#
#         # @tensor P1[-3; 7 8] := γ[u d 3 4; u d 7 8] * (u')[-3; 3 4]
#         @tensor P1[-2; 4] := γ[x 2; x 4] * (u')[-2; 2]
#
#         # B1inv = pinv(permute(B1, (4, 1, 2), (3, 5, 6)))
#         # R1 = permute(permute(P1, (), (1, 2, 3)) * B1inv, (1,), (2, 3))
#         # u, s, v = tsvd(u * R1; trunc=truncspace(ntsp[1]))
#         B1inv = pinv(permute(B1, (3, 1), (2, 4)))
#         R1 = permute(permute(P1, (), (1, 2)) * B1inv, (1,), (2,))
#         u, s, v, err = tsvd(u * R1; trunc=truncspace(D))
#
#         # @tensor B2[3 4 -3; 5 6 -6] := γ[1 2 3 4; 5 6 7 8] * v[-3; 1 2] * (v')[7 8; -6]
#         @tensor B2[-1 2; 3 -4] := γ[1 2; 3 4] * v[-1; 1] * (v')[4; -4]
#
#         # @tensor P2[3 4; -3] := γ[u d 3 4; u d 7 8] * (v')[7 8; -3]
#         @tensor P2[2; -4] := γ[x 2; x 4] * (v')[4; -4]
#
#         # B2inv = pinv(permute(B2, (4, 5, 3), (1, 2, 6)))
#         # R2 = permute(permute(P2, (), (1, 2, 3)) * B2inv, (1, 2), (3,))
#         # u, s, v = tsvd(R2 * v; trunc=truncspace(ntsp[1]))
#         B2inv = pinv(permute(B2, (3, 1), (2, 4)))
#         R2 = permute(permute(P2, (), (1, 2)) * B2inv, (1,), (2,))
#         u, s, v, err = tsvd(R2 * v; trunc=truncspace(D))
#
#         counter = counter + 1
#
#         δu = sqrt(real(tr((u - up) * (u - up)')))
#         δv = sqrt(real(tr((v - vp) * (v - vp)')))
#         println("du: ", δu)
#         println("Fidelity:  ", fidelity(γ, u, s, v))
#         # print("|")
#     end
#     print("\n")
#     return u, s, v
# end

# p(x) u(x) s(x) v(x+1) p(x + 1) u(x+1) ...
function optimise_isometries(Y::TruncationMetric, truncspace)
    nx, ny = size(Y)
    rx = 1:nx
    ry = 1:ny

    u = Matrix(undef, nx, ny)
    v = Matrix(undef, nx, ny)
    s = Matrix(undef, nx, ny)

    for x in rx, y in ry
        u[x, y], s[x, y], v[mod(x + 1, rx), y] = optimise_isometries(Y[x, y], truncspace)
        # u(x) <- u(x) * √s(x) 
        # v(x + 1) <- √s(x) * v(x+1)
        # u[x,y] = u[x,y] * sqrt(s[x,y])
        # v[mod(x + 1,rx),y] = sqrt(s[x,y]) * v[mod(x + 1,rx),y] 
    end
    return u, s, v
end
