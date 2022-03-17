function calculate_isometries(bondenv::AbstractTensorMap, D::Integer)
    # convention:
    # 3. e1u |-<-  -<-| w2u 1.
    # 2. e1d |->-  ->-| w2d 4.
    # ==========
    # -<-|u>-<-
    #

    σ = id(domain(bondenv)[1])

    u, s, v = tsvd(σ, (1,), (2,); trunc=truncdim(D))
    u = TensorMap(randn, codomain(u), domain(u))
    v = TensorMap(randn, codomain(v), domain(v))
    # u = isometry(domain(bondenv)[1], D)
    # v = isometry(codomain(bondenv)[1], D)
    @tensor B1[w2u e1d; e1u w2d] := bondenv[w2u 1; 2 w2d] * (u')[e1d; 1] * u[2; e1u]
    @tensor P1[e1d; w2d] := bondenv[2 1; 2 w2d] * (u')[e1d; 1]
    B1inv = permute(pinv(B1), (2, 4), (3, 1))

    R1 = permute(permute(P1, (), (2, 1)) * B1inv, (2,), (1,))
    u, s, v = tsvd(u * R1, (1,), (2,); trunc=truncdim(D))

    @tensor B2[w2u e1d; e1u w2d] := bondenv[1 e1d; e1u 2] * (v)[w2u; 1] * (v')[2; w2d]
    @tensor P2[e1d; w2d] := bondenv[2 e1d; 2 1] * (v')[1; w2d]
    B2inv = permute(pinv(B2), (2, 4), (3, 1))

    R2 = permute(permute(P2, (), (2, 1)) * B2inv, (2,), (1,))
    u, s, v = tsvd(R2 * v, (1,), (2,); trunc=truncdim(D))

    for i in 1:20
        print(PEPOKit.fidelity(bondenv, u, s, v), "\n")
        @tensor B1[w2u e1d; e1u w2d] = bondenv[w2u 1; 2 w2d] * (u')[e1d; 1] * u[2; e1u]
        @tensor P1[e1d; w2d] = bondenv[2 1; 2 w2d] * (u')[e1d; 1]
        B1inv = permute(pinv(B1), (2, 4), (3, 1))

        R1 = permute(permute(P1, (), (2, 1)) * B1inv, (2,), (1,))
        u, s, v = tsvd(u * R1, (1,), (2,); trunc=truncdim(D))

        @tensor B2[w2u e1d; e1u w2d] = bondenv[1 e1d; e1u 2] * (v)[w2u; 1] * (v')[2; w2d]
        @tensor P2[e1d; w2d] = bondenv[2 e1d; 2 1] * (v')[1; w2d]
        B2inv = permute(pinv(B2), (2, 4), (3, 1))

        R2 = permute(permute(P2, (), (2, 1)) * B2inv, (2,), (1,))
        u, s, v = tsvd(R2 * v, (1,), (2,); trunc=truncdim(D))
    end
    return u, s, v
end

function fidelity(bondenv, u, s, v)
    @tensor trϕϕ =
        bondenv[w2u e1d; e1u w2d] *
        u[e1u; s1u] *
        s[s1u; s2u] *
        v[s2u; w2u] *
        (v')[w2d; s1d] *
        s[s1d; s2d] *
        (u')[s2d; e1d]
    @tensor trρρ = bondenv[1 2; 1 2]
    @tensor trρϕ = bondenv[1 e1d; 1 w2d] * (v')[w2d; s1d] * s[s1d; s2d] * (u')[s2d; e1d]

    return real(trρϕ / sqrt(trϕϕ * trρρ))
end
