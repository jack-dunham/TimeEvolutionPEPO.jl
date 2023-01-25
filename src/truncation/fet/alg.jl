@doc raw"""
    truncate!(P, PE, Y::TruncationMetric; gauge=2)

Truncate the tensor `PE` along the east-west bond according the truncation metric `Y`, and 
store the result in `P`.
"""
function truncate!(P, PE, Y::TruncationMetric; ξ=Inf, gauge=2)
    truncspace = codomain(P[1, 1])[1] # The space we wish to truncate down to.

    # Optimise the isometries to maximise fidelity.
    U, S, V, = optimise_isometries(Y, truncspace)

    # Convert the isometries to the weighted trace gauge 
    if !(gauge == 0)
        U, S, V = wtg!(U, S, V, Y; ξ=ξ, test=false)

        if !(gauge == 1)
            tested = wtg_test.(U, S, V, Y)
            for i in tested
                left = i[1]
                right = i[2]
                if !left
                    @warn "Left weighted trace gauge condition was not met"
                end
                if !right
                    @warn "Right weighted trace gauge condition was not met"
                end
            end
        end
    end

    # Truncate the bonds using the isometries
    U = @. U * sqrt(S)
    V = sqrt.(circshift(S, (0, 1))) .* V

    isometryupdate!.(P, PE, U, V)

    return P
end

function fulltrotterstep!(pepo, op_tuple_x, op_tuple_y, bx_i, by_i)
    pepo, bx = axistrotterstep!(pepo, op_tuple_x)

    perm = (2, 1, 3, 5, 4, 6)
    pepo = _permute.(pepo, Ref(perm))

    pepo, by = axistrotterstep!(pepo, op_tuple_y)

    pepo = _permute.(pepo, Ref(invperm(perm)))

    return pepo, bx, by
end

function axistrotterstep!(pepo, op_tuple)
    pepo_hs = hilbertschmidt(pepo)

    boundary = get_initialboundary(pepo_hs)
    boundary = calculate!(pepo_hs)

    updatestep!(pepo, op_tuple, boundary)

    return pepo, boundary
end

function updatestep!(pepo, op_tuple, boundary)
    # Convenience
    P = pepo
    O = op_tuple
    E = boundary

    Nx, Ny = size(pepo)
    for y in 1:Ny
        for x in 1:Nx
            # Odd bond
            fetstep!(P[x, y], P[x + 1, y], O[x, y], E[x, y], trunc)
            # Even bond
            fetstep!(P[x + 1, y], P[x + 2, y], O[x + 1, y], E[x + 1, y], trunc)
        end
    end
    return P
end

function fetstep!(pepo_left, pepo_right, op_tuple, environment, trunc::TruncationScheme)
    op_left = op_tuple[1]
    op_right = op_tuple[2]

    q_left, l = evolveleft(pepo_left, op_left)
    r, q_right = evolveright(pepo_right, op_right)

    u, s, v = tsvd(r * l; trunc=trunc)

    t_metric = truncmetric(q_left, q_right, environment)

    u, s, v = optimise_isometries(u, s, v, t_metric, trunc)
    wtg!(u, s, v, t_metric)

    truncrightbond!(pepo_left, q_left, u * sqrt(s))
    truncleftbond!(pepo_right, q_right, sqrt(s) * v)

    return pepo_left, pepo_right
end

function truncrightbond!(p, q, u)
    @tensoropt p[w s k; e n b] = q[w s k; ee n b] * u[ee; e]
    return p
end
function truncleftbond!(p, q, u)
    @tensoropt p[w s k; e n b] = q[ww s k; e n b] * v[w; ww]
    return p
end
