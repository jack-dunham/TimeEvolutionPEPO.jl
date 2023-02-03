function updatestep!(pepo, op_tuple, boundary, trunc)
    # Convenience
    P = pepo
    O = op_tuple

    Nx, Ny = size(pepo)
    for y in 1:Ny
        if Nx == 1
            PL, PR = fetstep(P[1, y], P[1, y], O[1, y], boundary, trunc, 1, y)
            P[1, y], _ = fetstep(PR, PL, O[1, y], boundary, trunc, 1, y)
        else
            for x in 1:1
                # Odd bond
                P[x, y], P[x + 1, y] = fetstep(
                    P[x, y], P[x + 1, y], O[x, y], boundary, trunc, x, y
                )
                P[x + 1, y], P[x, y] = fetstep(
                    P[x + 1, y], P[x, y], O[x + 1, y], boundary, trunc, x + 1, y
                )
            end
        end
    end
    return P
end

function fetstep(
    pepo_left,
    pepo_right,
    op_tuple,
    environment,
    trunc::TruncationScheme,
    x,
    y;
    ξ=Inf,
    gauge=2,
)
    # Get the operators that will apply to the left and right sites
    op_left = op_tuple[1]
    op_right = op_tuple[2]

    # Apply evolution operator, and then extract isometry
    A, q_left, l = evolveleft_alt(pepo_left, op_left)
    B, r, q_right = evolveright_alt(pepo_right, op_right)

    # q_left = evolveleft_alt(pepo_left, op_left)
    # q_right = evolveright_alt(pepo_right, op_right)

    # Apply an initial truncation to isometries
    u, s, v, err = tsvd(l * r; trunc=truncdim(3))
    println(err)

    # Obtain the truncation metric with respect to the updated sites
    t_metric = truncmetric(q_left, q_left, q_right, q_right, environment.data, x, y)

    # Optimise the isometries with respect to the truncation metric
    # u, s, v = tsvd(id(domain(A)[1]); trunc=trunc)

    # u, s, v = optimise_isometries(u, s, v, t_metric, trunc)
    u, s, v = optiso(l, r, t_metric, trunc)
    println(s)

    # Convert to the weighted trace gauge.
    if !(gauge == 0)
        u, s, v = wtg(u, s, v, t_metric; ξ=ξ, test=false)

        if !(gauge == 1)
            tested = wtg_test(u, s, v, t_metric)
            left = tested[1]
            right = tested[2]
            if !left
                @warn "Left weighted trace gauge condition was not met"
            end
            if !right
                @warn "Right weighted trace gauge condition was not met"
            end
        end
    end

    # Update left and right sites (truncation step)
    new_pepo_left = similar(pepo_left)
    new_pepo_right = similar(pepo_right)

    truncrightbond!(new_pepo_left, q_left, u * sqrt(s))
    truncleftbond!(new_pepo_right, q_right, sqrt(s) * v)

    # Return updated sites
    return normalize(new_pepo_left), normalize(new_pepo_right)
end

function truncrightbond!(p, q, u)
    @tensoropt p[w s k; e n b] = q[w s k; ee n b] * u[ee; e]
    return p
end
function truncleftbond!(p, q, v)
    @tensoropt p[w s k; e n b] = q[ww s k; e n b] * v[w; ww]
    return p
end
