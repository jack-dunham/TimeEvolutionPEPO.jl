#=
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

All of this still obeys newly defined convention convention 
                ┌───────────────┐             ┌─────┐
                │               │     v  j◄───┤1   3│◄───i u
                │◄───3 i j 1◄───┤             │     │
                │               │  =          │     │
                │───►2 i'j'4───►│             │     │
                │               │    v' j'───►│4   2├───►i' u'
                └───────────────┘             └─────┘

s is the bond matrix (i think)

=#

# Vectorised form

rescale!(S, ξ) = S .= rescale.(S, ξ)
function rescale(s, ξ)
    sclm = diagm([exp(-(i - 1) / ξ) for i in 1:dim(codomain(s))])
    sclt = TensorMap(sclm, codomain(s), domain(s))
    s = s * sclt
    return s
end

wtg!(iso, metric, ::Val{false}) = iso, metric
function wtg!(iso, metric, ::Val{true})
    @info "Fixing to weighted trace gauge..."
    return wtg!(iso.u, iso.s, iso.v, metric)
end

function wtg!(
    u0::TensorMap{S,1,1}, # u[x] 
    s0::TensorMap{S,1,1}, # s[x]
    v0::TensorMap{S,1,1}, # v[x+1]
    Y::TensorMap{S,2,2}; # γ[x]
    test=true,
) where {S<:IndexSpace}
    # who knows if this works
    # if test == true
    #     skip = wtg_test(u0, s0, v0, Y)
    #     if skip[1] && skip[2]
    #         @info "Already in weighted trace gauge"
    #         return u0, s0, v0
    #     end
    # end

    u = u0 * sqrt(s0)
    v = sqrt(s0) * v0

    s_vals = [convert(Array, s0)[i, i] for i in 1:dim(codomain(s0))]
    s_min = minimum(abs.(s_vals))

    s_min < 1e-10 ? (@info "Skipped gauge fixing"; return u0, s0, v0) : nothing

    @debug "Singular values: " s_vals

    # @tensoropt γi[-1 -2; -3 -4] :=
    #     γ[1 2 3 4; 5 6 7 8] * u[5 6; -3] * v[-1; 1 2] * (u')[-2; 3 4] * (v')[7 8; -4]

    @tensoropt yiso[-1 -2; -3 -4] :=
        Y[1 2; 3 4] * u[3; -3] * v[-1; 1] * (u')[-2; 2] * (v')[4; -4]

    yf = permute(yiso, (2, 4), (3, 1))

    yf = (0.5 * (yf + yf'))

    posapprox!(yf)

    permute!(yiso, yf, ((4, 1), (3, 2)))

    γm = permute(yiso, ((1, 4), (3, 2))) # Transform to matrix
    @tensor γN = γm[a b; a b] # Y_ji
    γm = γm / γN

    # val,vec=eigen(γm)
    # println(val)

    # γmr = γm * (s ⊗ permute(s', (2,), (1,)))
    ri = Tensor(randn, ComplexF64, domain(γm))
    re, rs, _ = eigsolve(x -> γm * x, ri, 1, :LM; eager=true, ishermitian=false)

    li = TensorMap(randn, ComplexF64, one(codomain(γm)), codomain(γm))
    # γml = (s ⊗ permute(s', (2,), (1,))) * γm
    le, ls, _ = eigsolve(x -> x * γm, li, 1, :LM; eager=true, ishermitian=false)


    # println("re: ", re)
    # println("le: ", le)

    l0 = permute(ls[1], (2,), (1,))
    r0 = permute(rs[1], (1,), (2,))

    println(norm(l0 - cj(l0')))
    println(norm(r0 - cj(r0')))

    # dl, ul = eigen!(0.5 * (l0 + cj(l0')))
    # dr, ur = eigen!(0.5 * (r0 + cj(r0')))

    dl, ul = eigen!(l0)
    dr, ur = eigen!(r0)

    # sp = sqrt(dl) * ul' * s * ur * sqrt(dr)
    sp = sqrt(dl) * ul' * ur * sqrt(dr)
    vl, st, vrt = tsvd!(sp)

    x = vl' * sqrt(dl) * ul'
    y = ur * sqrt(dr) * vrt'

    # x = ul * inv(sqrt(dl)) * vl
    # y = vrt * inv(sqrt(dr)) * ur'

    mul!(u0, u, inv(x))
    mul!(v0, inv(y), v)

    # normalize!(st)

    for (_, v) in blocks(st)
        max = zero(scalartype(st))
        vmax = maximum(abs.(v))
        if vmax > max
            max = vmax
        end
        rmul!(st, 1/max)
    end

    println(convert(Array, st))

    # Test if the metric is gauged correctly.
    if test == true
        left, right = wtg_test(u0, st, v0, Y)
        if !left
            @warn "Left gauge condition was not met"
        end
        if !right
            @warn "Right gauge condition was not met"
        end
    end


    return u0, copy!(s0, st), v0

    #### this is an approach for complex space
    # eigr = eigh(γm; sortby = x -> -abs(x))

    # rδ = Tensor(zeros, ComplexF64, domain(eigr[1]))
    # rδ[1] = 1

    # r = eigr[2]*rδ 
    ####

    # @tensoropt Lm[-1 2; 3 -4] := γi[1 2; 3 4] * s[-1; 1] * (s')[4; -4]
end

#TODO: Tidy this once I am sure it is working.
function wtg_test(
    u::TensorMap{S,1,1}, s::TensorMap{S,1,1}, v::TensorMap{S,1,1}, γ::TensorMap{S,2,2}
) where {S<:IndexSpace}
    @tensoropt γin[-1 -2; -3 -4] :=
        γ[1 2; 3 4] * u[3; -3] * v[-1; 1] * (u')[-2; 2] * (v')[4; -4]

    # @tensoropt γin[-1 -2; -3 -4] :=
    #     γ[1 2 3 4; 5 6 7 8] * u[5 6; -3] * v[-1; 1 2] * (u')[-2; 3 4] * (v')[7 8; -4]

    # γmp = permute(γm, (1,4),(3,2))
    # @tensoropt γin[-1 -2; -3 -4] :=
    # γmp[1 2; 3 4] * x[3; -3] * y[-1; 1 ] * (x')[-2; 2] * (y')[4; -4]

    # γin = permute(γin, (1, 4), (3, 2)) #matricised 
    # @tensor γinN = γin[a b; a b]
    # γin = γin/γinN
    # γin = permute(γin, (1, 4), (3, 2)) #unmatricised 

    @tensoropt γl[-2; -3] := γin[-1 -2; -3 -4] * s[a; -1] * s[-4; a]
    @tensoropt γr[-1; -4] := γin[-1 -2; -3 -4] * s[a; -2] * s[-3; a]

    γla = convert(Array, γl)
    γlN = γla[1, 1]

    γra = convert(Array, γr)
    γrN = γra[1, 1]

    # @tensoropt γl[-2; -3] := γin[a -2; -3 a] 
    # @tensoropt γr[-1; -4] := γin[-1 b; b -4] 

    # println("WTG TEST", γl/tr(γl) *dim(domain(γl)))

    # if γl / tr(γl) * dim(domain(γl)) ≈ one(γl)
    #     left = true
    # else
    #     left = false
    # end
    # if γr / tr(γr) * dim(domain(γr)) ≈ one(γr)
    #     right = true
    # else
    #     right = false
    # end
    if isapprox(γl / γlN, one(γl))
        left = true
    else
        left = false
        @warn "" diff =γl / γlN -  one(γl)
    end
    if isapprox(γr / γrN, one(γr))
        right = true
    else
        right = false
        @warn "" diff = norm(γr / γrN -  one(γr))
    end
    return left, right
end
