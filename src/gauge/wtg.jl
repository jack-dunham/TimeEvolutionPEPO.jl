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

fixgauge!(iso, _, gauge::NoGauge) = iso
function fixgauge!(iso, metric, gauge::WTG)
    iso = wtg!(iso, metric; test=gauge.test, skiptol=gauge.skiptol, postol=gauge.postol)
    return iso
end

function rescale!(iso, xi; verbose=true)
    verbose && @info "Rescaling singular values with xi = $(xi)..."
    s = iso.s
    D = dim(codomain(s))

    prevu = iso.u * inv(sqrt(s))
    prevv = inv(sqrt(s)) * iso.v

    sclm = zeros(scalartype(s), D)

    if iszero(xi)
        sclm[begin] = 1
    else
        for i in eachindex(sclm)
            p = -(i - 1) / xi
            sclm[i] = exp(p)
        end
    end
    sclt = TensorMap(diagm(sclm), codomain(s), domain(s))

    si = s * sclt

    maxnormalize!(si)

    mul!(iso.u, prevu, sqrt(si))
    mul!(iso.v, sqrt(si), prevv)

    copy!(iso.s, si)

    return iso
end

function applyweight(uv, s, xi; verbose=true)
    u0, v0 = uv

    if xi < Inf
        verbose && @info "Rescaling singular values with xi = $(xi)..."

        D = dim(codomain(s))

        sclm = zeros(scalartype(s), D)

        if iszero(xi)
            sclm[begin] = 1
        else
            for i in eachindex(sclm)
                p = -(i - 1) / xi
                sclm[i] = exp(p)
            end
        end
        sclt = TensorMap(diagm(sclm), codomain(s), domain(s))

        s = s * sclt

        maxnormalize!(s)
        # rmul!(s, 1 / tr(s))
    end

    u = u0 * sqrt(s)
    v = sqrt(s) * v0

    return u => v, s
end

function rescale!(dst, s, zeta)
    sclm = diagm([exp(-(i - 1) / zeta) for i in 1:dim(codomain(s))])
    sclt = TensorMap(sclm, codomain(s), domain(s))

    mul!(dst, s, sclt)

    return dst
end

function wtg!(iso, metric; kwargs...)
    @info "Fixing to weighted trace gauge..."
    return wtg!(iso.u, iso.s, iso.v, metric; kwargs...)
end
function barewtg!(u, s, v, yiso; test=true, skiptol=1e-10, postol=eps())
    if test == true && bareverifywtg(s, yiso; verbose=false)
        @info "Already in weighted trace gauge"
        return u, s, v
    end

    svals = diag(convert(Array, s))
    smin = minimum(abs.(svals))

    if smin < skiptol
        @info "Skipped gauge fixing"
        return u, s, v
    end

    @debug "Singular values: " svals

    yf = permute(yiso, (2, 4), (3, 1))

    yf = (0.5 * (yf + yf'))
    posapprox!(yf; tol=postol)

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

    @debug norm(l0 - cj(l0'))
    @debug norm(r0 - cj(r0'))

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

    maxnormalize!(st)

    utemp = u * inv(x)
    vtemp = inv(y) * v

    # Test if the metric is gauged correctly.
    test && verifywtg(inv(x), st, inv(y), yiso)

    copy!(s, st)

    mul!(u, utemp, sqrt(s))
    mul!(v, sqrt(s), vtemp)

    return u, s, v
end

function wtg!(u, s, v, Y; test=true, skiptol=1e-14, postol=eps())
    if test == true && verifywtg(u, s, v, Y; verbose=false)
        @info "Already in weighted trace gauge"
        return u, s, v
    end

    svals = diag(convert(Array, s))
    smin = minimum(abs.(svals))

    # if smin < skiptol
    #     @info "Skipped gauge fixing"
    #     return u, s, v
    # end

    @debug "Singular values: " svals

    # @tensoropt γi[-1 -2; -3 -4] :=
    #     γ[1 2 3 4; 5 6 7 8] * u[5 6; -3] * v[-1; 1 2] * (u')[-2; 3 4] * (v')[7 8; -4]

    @tensoropt yiso[-1 -2; -3 -4] :=
        Y[1 2; 3 4] * u[3; -3] * v[-1; 1] * (u')[-2; 2] * (v')[4; -4]

    yf = permute(yiso, (2, 4), (3, 1))

    # yf = (0.5 * (yf + yf'))
    stablizemetric!(yf)
    posapprox!(yf; tol=postol)

    permute!(yiso, yf, ((4, 1), (3, 2)))

    γm = permute(yiso, ((1, 4), (3, 2))) # Transform to matrix
    @tensor γN = γm[a b; a b] # Y_ji
    γm = γm / γN

    # val,vec=eigen(γm)
    # println(val)

    # γmr = γm * (s ⊗ permute(s', (2,), (1,)))
    ri = Tensor(randn, ComplexF64, domain(γm))
    # re, rs, _ = eigsolve(x -> γm * x, ri, 1, :LM; eager=true, ishermitian=false)
    DR,V = eigen(γm)
    rs1 = V * isometry(domain(V), one(spacetype(V)))


    li = TensorMap(randn, ComplexF64, one(codomain(γm)), codomain(γm))
    # γml = (s ⊗ permute(s', (2,), (1,))) * γm
    # le, ls, _ = eigsolve(x -> x * γm, li, 1, :LM; eager=true, ishermitian=false)
    DL,V = eigen(_transpose(γm))
    ls1 = _transpose(V * isometry(domain(V), one(spacetype(V))))

    lmin = minimum(abs.(diag(convert(Array, DL))))
    rmin = minimum(abs.(diag(convert(Array, DR))))

    if (lmin < skiptol || rmin < skiptol)
        @info "skipped gauge fixing $lmin $rmin"
        return u,s,v
    end


    l0 = permute(ls1, (2,), (1,))
    r0 = permute(rs1, (1,), (2,))

    @debug norm(l0 - cj(l0'))
    @debug norm(r0 - cj(r0'))

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

    maxnormalize!(st)

    utemp = u * inv(x)
    vtemp = inv(y) * v

    # Test if the metric is gauged correctly.
    test && verifywtg(utemp, st, vtemp, Y)

    copy!(s, st)

    mul!(u, utemp, sqrt(s))
    mul!(v, sqrt(s), vtemp)

    return u, s, v
end

function _wtg!(u, s, v, Y; test=true, skiptol=1e-14, postol=eps())
    if test == true && verifywtg(u, s, v, Y; verbose=false)
        @info "Already in weighted trace gauge"
        return u, s, v
    end

    svals = diag(convert(Array, s))
    smin = minimum(abs.(svals))

    if smin < skiptol
        @info "Skipped gauge fixing"
        return u, s, v
    end

    @debug "Singular values: " svals

    # @tensoropt γi[-1 -2; -3 -4] :=
    #     γ[1 2 3 4; 5 6 7 8] * u[5 6; -3] * v[-1; 1 2] * (u')[-2; 3 4] * (v')[7 8; -4]

    @tensoropt yiso[-1 -2; -3 -4] :=
        Y[1 2; 3 4] * u[3; -3] * v[-1; 1] * (u')[-2; 2] * (v')[4; -4]

    yf = permute(yiso, (2, 4), (3, 1))

    # yf = (0.5 * (yf + yf'))
    stablizemetric!(yf)
    posapprox!(yf; tol=postol)

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

    println("re: ", re)
    println("le: ", le)

    l0 = permute(ls[1], (2,), (1,))
    r0 = permute(rs[1], (1,), (2,))

    @debug norm(l0 - cj(l0'))
    @debug norm(r0 - cj(r0'))

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

    maxnormalize!(st)

    utemp = u * inv(x)
    vtemp = inv(y) * v

    # Test if the metric is gauged correctly.
    test && verifywtg(utemp, st, vtemp, Y)

    copy!(s, st)

    mul!(u, utemp, sqrt(s))
    mul!(v, sqrt(s), vtemp)

    return u, s, v
end

function wtg2!(u, s, v, Y; skiptol=1e-10, postol=eps())
    svals = diag(convert(Array, s))
    smin = minimum(abs.(svals))

    if smin < skiptol
        @info "Skipped gauge fixing"
        return u, s, v
    end

    @debug "Singular values: " svals

    yf = permute(Y, (2, 4), (3, 1))

    yf = (0.5 * (yf + yf'))

    posapprox!(yf; tol=postol)

    permute!(Y, yf, ((4, 1), (3, 2)))

    γm = permute(Y, ((1, 4), (3, 2))) # Transform to matrix
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

    @debug norm(l0 - cj(l0'))
    @debug norm(r0 - cj(r0'))

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

    maxnormalize!(st)

    utemp = u * inv(x)
    vtemp = inv(y) * v

    copy!(s, st)

    mul!(u, utemp, sqrt(s))
    mul!(v, sqrt(s), vtemp)

    return u, s, v
end

function bareverifywtg(s, γin; verbose=true)

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

    isgauged = true

    if γl / γlN ≉ one(γl)
        isgauged = false
        if verbose
            diff = norm(γl / γlN - one(γl))
            @warn "Left gauge condition was not met: δ = $diff"
        end
    end

    if γr / γrN ≉ one(γr)
        isgauged = false
        if verbose
            diff = norm(γr / γrN - one(γr))
            @warn "Right gauge condition was not met: δ = $diff"
        end
    end

    return isgauged
end
#TODO: Tidy this once I am sure it is working.
function verifywtg(u, s, v, γ; verbose=true)
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

    isgauged = true

    if γl / γlN ≉ one(γl)
        isgauged = false
        if verbose
            diff = norm(γl / γlN - one(γl))
            @warn "Left gauge condition was not met: δ = $diff"
        end
    end

    if γr / γrN ≉ one(γr)
        isgauged = false
        if verbose
            diff = norm(γr / γrN - one(γr))
            @warn "Right gauge condition was not met: δ = $diff"
        end
    end

    return isgauged
end

function wtgisometries!(isoms, pepo, stepper; maxiter=100, tol=1e-8, verbose=false)
    conv = Inf

    for numiter in 1:maxiter
        for bond in eachbond(axisbuffer(pepo))
            temppepo = deepcopy(pepo)

            for I in eachindex(axisbuffer(temppepo))
                dst = axisbuffer(temppepo)[I]
                op = axisbuffer(stepper)[I]
                let (e, s, w, n) = isoms[I]
                    truncateto!(dst, deepcopy(dst), op, e, s, w, n)
                end
            end

            env = weightenvironment(temppepo[bond], weights(pepo), bond)

            if bondaxis(bond) == 1
                (u, _, _, _), (_, _, v, _) = isoms[bond]
            elseif bondaxis(bond) == 2
                (_, u, _, _), (_, _, _, v) = isoms[bond]
            end

            s = weights(pepo)[bond]

            oldS = bondentropy(weights(pepo))[1, 1][bondaxis(bond)]
            barewtg!(u, s, v, env)
            conv = abs(bondentropy(weights(pepo))[1, 1][bondaxis(bond)] - oldS)
        end

        verbose && println(conv)

        if conv < tol
            break
        end
    end

    return isoms
end
