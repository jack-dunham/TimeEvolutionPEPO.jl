function updatestep!(pepo, coeffs, op_tuple, boundary, trunc; kwargs...)
    # Convenience
    P = pepo
    O = op_tuple

    Nx, Ny = size(pepo)
    for y in 1:Ny
        # if Nx == 1
        #     PL, PR, coeffs[1, y] = fetstep(P[1, y], P[1, y], O[1, y], boundary, trunc, 1, y)
        #     P[1, y], _, coeffs[2, y] = fetstep(PR, PL, O[1, y], boundary, trunc, 1, y)
        #     # P[1, y] = PL
        #     # coeffs[2, y] = coeffs[1, y]
        # else
        for x in 1:Nx # Update all bonds
            P[x, y], P[mod(x + 1, 1:Nx), y], coeffs[x, y] = fetstep(
                P[x, y], P[x + 1, y], O[x, y], boundary, trunc, x, y; kwargs...
            )
        end
        if latticestyle(P).issymetric == true
            for x in 1:(Nx - 1)
                P[x, y + 1], P[x + 1, y + 1] = P[x + 1, y], P[x + 2, y]
            end
            break
        end
    end
    return P
end

function fetstep(
    pepo_left, pepo_right, op_tuple, boundary, trunc::TruncationScheme, x, y; ξ=Inf, gauge=2
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
    u, s, v, err = tsvd(l * r; trunc=trunc)

    # Obtain the truncation metric with respect to the updated sites
    t_metric = truncmetric(q_left, q_left, q_right, q_right, boundary.data, x, y)
    # t_metric = truncmetric(A, A, B, B, environment.data, x, y)

    # Optimise the isometries with respect to the truncation metric
    # u, s, v = tsvd(id(domain(A)[1]); trunc=trunc)

    # u, s, v = optimise_isometries(u, s, v, t_metric, trunc)
    u, s, v = optiso(l, r, t_metric, trunc)

    # Convert to the weighted trace gauge.
    if !(gauge == 0)
        @info "Gauging..."
        u, s, v = wtg(u, s, v, t_metric; test=false)
        @info "...done."

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
    # Rescale the singular values.
    if ξ != Inf
        sclm = diagm([exp(-(i - 1) / ξ) for i in 1:dim(codomain(s))])
        sclt = TensorMap(sclm, codomain(s), domain(s))
        s = s * sclt

        if !(gauge == 0)
            @info "Gauging..."
            u, s, v = wtg(u, s, v, t_metric; test=false)
            @info "...done."

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
    end

    # Update left and right sites (truncation step)
    new_pepo_left = similar(pepo_left)
    new_pepo_right = similar(pepo_right)

    truncrightbond!(new_pepo_left, q_left, u * sqrt(s))

    # new_new_pepo_left, _, _ = evolveright_alt(new_pepo_left, op_right)
    # truncleftbond!(new_pepo_left, new_new_pepo_left, sqrt(s) * v)
    # new_pepo_right = new_pepo_left

    truncleftbond!(new_pepo_right, q_right, sqrt(s) * v)

    # Return updated sites and bond matrix. 
    return normalize(new_pepo_left), normalize(new_pepo_right), convert(Array, s)
end

function truncrightbond!(p, q, u)
    @tensoropt p[w s k; e n b] = q[w s k; ee n b] * u[ee; e]
    return p
end
function truncleftbond!(p, q, v)
    @tensoropt p[w s k; e n b] = q[ww s k; e n b] * v[w; ww]
    return p
end
function truncrightbond(q, u)
    @tensoropt p[w s k; e n b] := q[w s k; ee n b] * u[ee; e]
    return p
end
function truncleftbond(q, v)
    @tensoropt p[w s k; e n b] := q[ww s k; e n b] * v[w; ww]
    return p
end

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
