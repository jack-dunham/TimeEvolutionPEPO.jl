function fetstep(
    vp::TensorMap{S,4,2},
    m::TensorMap{S,3,3},
    A::TensorMap{S,2,2},
    axis,
    ξ=Inf;
    alg::VUMPS=VUMPS(),
    gauge::Int=2,
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
    hs = hilbertschmidt(vpe)
    γ, A = hsbondenv(hs, A; alg=alg)
    u, s, v = calculate_isometries(γ, D)

    if !(gauge == 0)
        u, s, v = wtg(γ, u, s, v, ξ; test=false)

        if !(gauge == 1) && axis == :y
            left, right = wtg_test(γ, u, s, v)
            if !left
                @warn "Left gauge condition was not met"
            end
            if !right
                @warn "Right gauge condition was not met"
            end
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

    return _permute(vp, invperm(perm)), A
end

function fetstep(
    p::TensorMap{S,3,3},
    m::TensorMap{S,3,3},
    A::TensorMap{S,2,2},
    axis,
    ξ=Inf;
    alg::VUMPS=VUMPS(),
    gauge::Int=2,
) where {S<:IndexSpace}
    D = dim(codomain(p)[2])

    if axis == :x
        perm = (1, 2, 3, 4, 5, 6)
    elseif axis == :y
        perm = (1, 3, 2, 4, 6, 5)
    else
        throw(ArgumentError("Axis must be either :x or :y"))
    end
    pp = _permute(p, perm)

    pe = evolve(pp, m)
    hs = hilbertschmidt(pe)
    γ, A = hsbondenv(hs, A; alg=alg)
    u, s, v = calculate_isometries(γ, D)

    if !(gauge == 0)
        u, s, v = wtg(γ, u, s, v, ξ; test=false)

        if !(gauge == 1) && axis == :y
            left, right = wtg_test(γ, u, s, v)
            if !left
                @warn "Left gauge condition was not met"
            end
            if !right
                @warn "Right gauge condition was not met"
            end
        end
    end

    # u = u * sqrt(s)
    # v = sqrt(s) * v
    # @tensoropt vpn[b k -3 -4; -5 -6] :=
    #     vpe[b k 3 4 -4; 6 7 -6] * v[-3; 3 4] * u[6 7; -5]

    isometry_update!(p, pe, u, s, v)
    #= 
    Simple normalisation to avoid eigenvalues blowing up. To physically normalise the state
    one must use the trace environment to obtain the tr(ρ) and then divide through.
    =#
    p = p / norm(p)

    return _permute(p, invperm(perm)), A
end
