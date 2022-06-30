function fetstep!(
    vp::TensorMap{S,4,2}, m::TensorMap{S,3,3}, axis, ξ=Inf
) where {S<:IndexSpace}
    D = dim(codomain(vp)[3])

    if axis == :x
        perm = (1, 2, 3, 4, 5, 6)
    elseif axis == :y
        perm = (1, 2, 4, 3, 6, 5)
    else
        throw(ArgumentError("Axis must be either :x or :y"))
    end
    vpp = _permute(vp, perm)

    vpe = evolve(vpp, m)
    hs = _hilbertschmidt(vpe)
    γ = hsbondenv(hs, D)
    u, s, v = calculate_isometries(γ, D)

    u, s, v = wtg(γ, u, s, v, ξ; test=false)

    if axis == :y
        left, right = wtg_test(γ, u, s, v)
        if !left
            @warn "Left gauge condition was not met"
        end
        if !right
            @warn "Right gauge condition was not met"
        end
    end

    # u = u * sqrt(s)
    # v = sqrt(s) * v
    # @tensoropt vpn[b k -3 -4; -5 -6] :=
    #     vpe[b k 3 4 -4; 6 7 -6] * v[-3; 3 4] * u[6 7; -5]

    isometry_update!(vp, vpe, u, s, v)
    #= 
    Simple normalisation to avoid eigenvalues blowing up. To physically normalise the state
    one must use the trace environment to obtain the tr(ρ) and then divide through.
    =#
    vp = vp / norm(vp)

    return _permute(vp, invperm(perm))
end

function fet(vp::TensorMap{S,4,2}, m::TensorMap{S,3,3}, ξ=Inf) where {S<:IndexSpace}
    vpn = vp
    for i in 1:20
        if i == 10
            @warn "phase..."
        end
        vpn = fetstep!(vpn, m, :x)
        vpn = fetstep!(vpn, m, :y)
    end
    return vpn
end
