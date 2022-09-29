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
    @tensoropt pn[k -3 -4; b -5 -6] = pe[ k 3 4 -4;b 6 7 -6] * v[-3; 3 4] * u[6 7; -5]
    return pn
end

function fidelity(
    γ::AbstractTensorMap{S,4,4},
    u::AbstractTensorMap{S,2,1},
    s::AbstractTensorMap{S,1,1},
    v::AbstractTensorMap{S,1,2},
) where {S<:IndexSpace}
    @tensor trϕϕ =
        γ[1 2 3 4; 5 6 7 8] *
        u[5 6; a1] *
        s[a1; a2] *
        v[a2; 1 2] *
        (v')[7 8; b2] *
        (s')[b2; b1] *
        (u')[b1; 3 4]
    @tensor trρρ = γ[1 2 3 4; 1 2 3 4]
    @tensor trρϕ = γ[1 2 3 4; 1 2 7 8] * (u')[a; 3 4] * (s')[b; a] * (v')[7 8; b]
    return real(trρϕ / sqrt(trϕϕ * trρρ))
end

function purity(bondenv)
    return @tensor trρρ = bondenv[1 2 3 4; 1 2 3 4]
end

function calculate_isometries(γ::TensorMap{<:IndexSpace,4,4}, D; maxiter=100)
    ntsp = codomain(γ)
    σ = id(ntsp[1] * ntsp[2])
    u, s, v = tsvd(σ; trunc=truncspace(ntsp[1]))
    u = TensorMap(randn, codomain(u), domain(u))
    v = TensorMap(randn, codomain(v), domain(v))
    # println("F0: ", _fidelity(γ, u, s, v))

    # this is awfully confusing
    δu = 1.0
    δv = 1.0
    counter = 0
    while !(counter > maxiter || max(δu, δv) < 1e-8)
        up = u
        vp = v

        @tensor B1[1 2 -3; -4 7 8] := γ[1 2 3 4; 5 6 7 8] * u[5 6; -4] * (u')[-3; 3 4]
        @tensor P1[-3; 7 8] := γ[u d 3 4; u d 7 8] * (u')[-3; 3 4]
        B1inv = pinv(permute(B1, (4, 1, 2), (3, 5, 6)))
        R1 = permute(permute(P1, (), (1, 2, 3)) * B1inv, (1,), (2, 3))
        u, s, v = tsvd(u * R1; trunc=truncspace(ntsp[1]))

        @tensor B2[3 4 -3; 5 6 -6] := γ[1 2 3 4; 5 6 7 8] * v[-3; 1 2] * (v')[7 8; -6]
        @tensor P2[3 4; -3] := γ[u d 3 4; u d 7 8] * (v')[7 8; -3]
        B2inv = pinv(permute(B2, (4, 5, 3), (1, 2, 6)))
        R2 = permute(permute(P2, (), (1, 2, 3)) * B2inv, (1, 2), (3,))
        u, s, v = tsvd(R2 * v; trunc=truncspace(ntsp[1]))

        counter = counter + 1

        δu = sqrt(real(tr((u - up) * (u - up)')))
        δv = sqrt(real(tr((v - vp) * (v - vp)')))
        #println("δu: ", δu, "\tδv", δv)
    end
    println("fidelity: ", fidelity(γ, u, s, v))
    return u, s, v
end
