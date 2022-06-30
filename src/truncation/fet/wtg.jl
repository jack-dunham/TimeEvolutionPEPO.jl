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

                ┌───────────────┐          ┌─────┐
                │               │     j◄───┤1   3│◄───i
                │◄───3 i j 1◄───┤          │     │
                │               │  =       │     │
                │───►2 i'j'4───►│          │     │
                │               │    j'───►│4   2├───►i' 
                └───────────────┘          └─────┘

s is the bond matrix (i think)

=#
function wtg(
    γ::TensorMap{S,4,4},
    u::TensorMap{S,2,1},
    s::TensorMap{S,1,1},
    v::TensorMap{S,1,2},
    ξ=Inf;
    test=true,
    ) where {S<:IndexSpace}
    # who knows if this works
    if test == true
        skip = wtg_test(γ, u, s, v)
        if skip[1] && skip[2]
            @info "Already in WTG"
            return u, s, v
        end
    end

    u = u * sqrt(s)
    v = sqrt(s) * v

    @tensoropt γi[-1 -2; -3 -4] :=
        γ[1 2 3 4; 5 6 7 8] * u[5 6; -3] * v[-1; 1 2] * (u')[-2; 3 4] * (v')[7 8; -4]

    γm = permute(γi, (1, 4), (3, 2)) #matricised 
    @tensor γN = γm[a b; a b]
    γm = γm / γN
    γm = 1 / 2 * (γm' + γm) # γm is now hermitian, so SVD gived eigen
    


    # γmr = γm * (s ⊗ permute(s', (2,), (1,)))
    rλs, rs, rinfo = eigsolve(
        x -> γm * x,
        Tensor(randn, ComplexF64, domain(γm)),
        1,
        :LM;
        eager=true,
        ishermitian=true,
    )

    li = TensorMap(randn, ComplexF64, one(codomain(γm)), codomain(γm))
    # γml = (s ⊗ permute(s', (2,), (1,))) * γm
    lλs, ls, linfo = eigsolve(x -> x * γm, li, 1, :LM; eager=true, ishermitian=true)

    l0 = permute(ls[1], (2,), (1,))
    r0 = permute(rs[1], (1,), (2,))

    dl, ul = eigen!(l0)
    dr, ur = eigen!(r0)

    # sp = sqrt(dl) * ul' * s * ur * sqrt(dr)
    sp = sqrt(dl) * ul' * ur * sqrt(dr)
    vl, st, vrt = tsvd!(sp)

    if ξ != Inf
        sclm = diagm([exp(-(i - 1) / ξ) for i in 1:dim(codomain(st))])
        sclt = TensorMap(sclm, codomain(st), domain(st))
        st = st * sclt
    end

    # x = vl' * sqrt(dl) * ul' 
    # y = ur * sqrt(dr) * vrt'

    x = ul * 1 / sqrt(dl) * vl
    y = vrt * 1 / sqrt(dr) * ur'

    un = u * x
    vn = y * v

    #### testing 
    if test == true
        left, right = wtg_test(γ, un, st, vn)
        if !left
            @warn "Left gauge condition was not met"
        end
        if !right
            @warn "Right gauge condition was not met"
        end
    end

    return un, st, vn

    #### this is an approach for complex space
    # eigr = eigh(γm; sortby = x -> -abs(x))

    # rδ = Tensor(zeros, ComplexF64, domain(eigr[1]))
    # rδ[1] = 1

    # r = eigr[2]*rδ 
    ####

    # @tensoropt Lm[-1 2; 3 -4] := γi[1 2; 3 4] * s[-1; 1] * (s')[4; -4]
end

function wtg_test(
    γ::TensorMap{S,4,4}, u::TensorMap{S,2,1}, s::TensorMap{S,1,1}, v::TensorMap{S,1,2}
    ) where {S<:IndexSpace}
    @tensoropt γin[-1 -2; -3 -4] :=
        γ[1 2 3 4; 5 6 7 8] * u[5 6; -3] * v[-1; 1 2] * (u')[-2; 3 4] * (v')[7 8; -4]
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
    if isapprox(γl / γlN, one(γl); rtol=1e-5)
        left = true
    else
        left = false
    end
    if isapprox(γr / γrN, one(γr); rtol=1e-5)
        right = true
    else
        right = false
    end
    return left, right
end
