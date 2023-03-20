#=
             12  6
              │  ▲
              ▼  │
            ┌────┴──┐
      1◄────┤       │◄────11
      2◄────┤       │◄────10
            │   M   │
      7────►│       ├────►5
      8────►│       ├────►4
            └─┬─────┘
              │  ▲
              ▼  │
              3  9
better convention is below (not implemented)
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

             ┌───────6
             ▼
           ┌───┐
           │   │◄───4
           │   │◄───5
           │ L │
           │   ├───►1
           │   ├───►2
           └─┬─┘
             └──────►3

      3◄─────┐
           ┌─┴─┐
      1◄───┤   │
      2◄───┤   │
           │ R │
      4───►│   │
      5───►│   │
           └───┘
             ▲
      6──────┘

             ┌───┐
         2◄──┤ a │◄──4
             └┬──┘ 
              │ ▲  
              ▼ │
              1 3

              3 1
              │ ▲
              ▼ │
             ┌──┴┐      
         4──►│ a'│──►2 
             └───┘     
TODO: rewrite in above convention
Index convention for position in index array, is codomain counterclockwise from most 
west-northerly (west first) positon, then domain in same order.

                           ┌───┐
     v ^               1◄──┤ a │◄──4
     2 3                   └┬──┘ 
--------------              │ ▲  
     4 1                    ▼ │
     v ^                    2 3

3 ->[ a']-> 2               4 1
                            │ ▲
      3                     ▼ │
      v                    ┌──┴┐      
2 ->[ a']->1           3──►│ a'│──►2 
                           └───┘     
=#

# function applyH1(
#     AC::TensorMap{S,2,2}, FL::TensorMap{S,3,3}, FR::TensorMap{S,3,3}, M::TensorMap{S,6,6}
# ) where {S<:IndexSpace}
#     @tensor HAC[l ld; rd r] :=
#         FL[l 8 7; 2 1 a] *
#         AC[a 12; 6 b] *
#         M[1 2 ld 4 5 6; 7 8 rd 10 11 12] *
#         FR[11 10 b; 5 4 r]
#     return HAC
# end

## For TR
# PERF: Convention corrected
function applyhac(
    ac::AbstractTensorMap{S,2,1},
    fl::AbstractTensorMap{S,1,2},
    fr::AbstractTensorMap{S,2,1},
    m::AbstractTensorMap{S,2,2},
) where {S<:IndexSpace}
    @tensoropt begin
        hac[-2 -1; -3] := ac[a 4; b] * fl[-2; a 1] * m[1 -1; 3 4] * fr[b 3; -3]
    end
    return hac
end

#for TR
# PERF: Convention corrected
function applyhc(
    c::AbstractTensorMap{S,1,1}, fl::AbstractTensorMap{S,1,2}, fr::AbstractTensorMap{S,2,1}
) where {S<:IndexSpace}
    @tensoropt hc[a; b] := c[x; y] * fl[a; x m] * fr[y m; b]
    return hc
end

@doc raw"""
    leftgauge(A::TensorMap, [C]; kwargs...)

Given a MPS tensor `A`, return a left-canonical MPS tensor `AL`, a gauge transform `C` and
a scalar factor `λ` such that ``λ AL^s C = C A^s``, where an initial guess for `C` can be
provided.

                   ┌────┐   ┌───┐          ┌───┐   ┌───┐
            λ * ◄──┤ AL │◄──┤ C │◄─── = ◄──┤ C │◄──┤ A │◄──
                   └─┬──┘   └─┬─┘          └─┬─┘   └─┬─┘
                     ▼        ▼              ▼       ▼
                     s        s              s       s

"""
function leftgauge(
    A,
    C=TensorMap(randn, ComplexF64, codomain(A)[2], codomain(A)[2]);
    tol=1e-12,
    maxiter=100,
)
    AL, C, R, λ = leftgaugeinitial(A, C; tol=tol)
    numiter = 1
    while norm(C - R) > tol && numiter < maxiter
        λs, Cs, info = eigsolve(
            x -> lgsolve(A, AL, x),
            R,
            1,
            :LM;
            ishermitian=false,
            tol=tol,
            eager=true,
            maxiter=1,
        )
        AL, C, R, λ = leftgaugefactor(Cs[1], A)
        numiter += 1
    end
    C = R
    return AL, C, λ
end

function leftgaugefactor(C, A)
    # _, C = leftorth(C)
    CA = _CA(C, A)

    n1 = numout(CA)
    inds = Tuple(1:numind(CA))

    Q, R = leftorth(CA, inds[1:(end - 1)], inds[end:end])
    # Q, R = leftorth(CA, (1, 2, 3), (4,))

    AL = permute(Q, inds[1:n1], inds[(n1 + 1):end])
    # AL = permute(Q, (1, 2), (3, 4))

    λ = norm(R)
    R = R * 1 / λ
    return AL, C, R, λ
end

@doc raw"""

    _CA(C::TensorMap{S,1,1}, A::TensorMap{S,2,2}) where {S<:IndexSpace}

Apply a matrix `C` to the left bond of a MPS tensor `A`
                                          
    ┌─┐ a ┌───┐     |      ┌─┐ a ┌───┐    
2◄──│C│◄──┤ A │◄──3 |  2◄──│C│◄──┤ A │◄──4
    └─┘   └─┬─┘     |      └─┘   └┬──┘    
            ▼                     │ ▲    
            1                     ▼ │
                                  1 3
"""
function _CA(C::AbstractTensorMap{S,1,1}, A::AbstractTensorMap{S,2,1}) where {S<:IndexSpace}
    return @tensor CA[1 2; 3] := C[1; a] * A[a 2; 3]
end

# Apply a matrix C to the right bond of a MPS Tensor
# assumes left bond is last cod index, and right bond is last dom index
function _AC(A::AbstractTensorMap{S}, C::AbstractTensorMap{S,1,1}) where {S<:IndexSpace}
    p = (3, 2, 1)
    CA = _CA(_permute(C, (2, 1)), _permute(A, p))
    return _permute(CA, invperm(p))
end

function leftgauge_check(al::TensorMap{<:IndexSpace,2,1})
    return al' * al
end
function rightgauge_check(ar::TensorMap{<:IndexSpace,2,1})
    arr = permute(ar, (2,), (1, 3))
    return arr * arr'
end

function updateleft!(al::AbstractTensorMap, ac::AbstractTensorMap, c::AbstractTensorMap)
    qac, rac = leftorth(ac)
    qc, rc = leftorth(c)

    mul!(al, qac, qc') # QAC * QC' -> AL
    errL = norm(rac - rc)

    return errL
end

function updateleft!(A::InfiniteMPS)
    AL, C, _, AC = unpack(A)
    errL = updateleft!.(AL, AC, C)
    return errL
end
function updateleft!(A::MultilineInfiniteMPS)
    errL = updateleft!.(A)
    return A, hcat(errL...)
end

function updateright!(ar::AbstractTensorMap, ac::AbstractTensorMap, c::AbstractTensorMap)
    lac, qac = rightorth(ac, (1,), (2, 3))
    lc, qc = rightorth(c)

    permute!(ar, qc' * qac, (1, 2), (3,))
    errR = norm(lac - lc)

    return errR
end

function updateright!(A::InfiniteMPS)
    _, C, AR, AC = unpack(A)
    errR = updateright!.(AR, AC, circshift(C, 1))
    return errR
end

function updateright!(A::MultilineInfiniteMPS)
    errR = updateright!.(A)
    return A, hcat(errR...)
end
# function vumps_rightupdate(AC,C)
#     U,S,V = tsvd(_CA(C', AC), (2,),(1,3))
#     AR = _CA(U, permute(V, (2,1),(3,)))
#     ϵR = norm(AC - _CA(C,AR))
#     return AR, ϵR
# end
function updateboth!(A::MultilineInfiniteMPS)
    _, errL = updateleft!(A)
    _, errR = updateright!(A)
    return A, errL, errR
end

function applyhac(
    z::RecursiveVec, FL::AbstractVector, FR::AbstractVector, M::AbstractVector
)
    Ny = length(FL)
    rv = Vector{eltype(z.vecs)}(undef, Ny)
    for y in 1:Ny
        rv[y] = applyhac(z[y], FL[y], FR[y], M[y])
    end
    rv = circshift(rv, 1)
    return RecursiveVec(rv...)
end

function applyhc(z::RecursiveVec, FL::AbstractVector, FR::AbstractVector)
    Ny = length(FL)
    rv = Vector{eltype(z.vecs)}(undef, Ny)
    for y in 1:Ny
        rv[y] = applyhc(z[y], FL[y], FR[y])
    end
    rv = circshift(rv, 1)
    return RecursiveVec(rv...)
end

function vumpsupdate!(
    A::MultilineInfiniteMPS, FL::LeftEnvironment, FR::RightEnvironment, M::InfiniteMPO
)
    nx, ny = size(M)
    rx = 1:nx
    ry = 1:ny

    AC = centraltensor(A)
    C = bondtensor(A)

    for x in rx

        # take mps[y], get mps[y].AC, send in mps[y].AC[x]
        Threads.@sync begin
            Threads.@spawn begin
                _, ACs, _ = eigsolve(
                    z -> applyhac(z, ($FL[x, :]), ($FR[x, :]), ($M[x, :])),
                    RecursiveVec(($AC[x, :])...),
                    1,
                    :LM,
                )

                # @info "μ1 $(μ1s)"
                for y in ry
                    A[y].AC[x] = ACs[1][y]
                end
            end
            # now set mps[y-1, mod].AC[x] to output of above

            # A[mod(y - 1, ry)].AC[x] = ACs[1]
            Threads.@spawn begin
                _, Cs, _ = eigsolve(
                    z -> applyhc(z, ($FL[x + 1, :]), ($FR[x, :])),
                    RecursiveVec(($C[x, :])...),
                    1,
                    :LM,
                )

                # @info "μ0 $(μ0s[1]))"

                for y in ry
                    A[y].C[x] = Cs[1][y]
                end
            end
        end
        # A[mod(y - 1, ry)].C[x] = Cs[1]
        # λ = real(μ1s[1] / μ0s[1])
    end

    #A now as updated AC, C, need to update AL, AR
    _, errL, errR = updateboth!(A)

    errL, _ = findmax(errL)
    errR, _ = findmax(errR)
    # @info "errL = $errL \t errR = $errR"
    return A, errL, errR
end

function vumpsstep!(
    A::MultilineInfiniteMPS, FL::LeftEnvironment, FR::RightEnvironment, M::InfiniteMPO
)
    A, errL, errR = vumpsupdate!(A, FL, FR, M) # array form

    _, _ = environments!(FL, FR, A, M)

    AC = centraltensor(A)

    # Error calculation
    vl = leftnull.(circshift(AC, (0, -1)))
    ac = applyhac.(AC, FL, FR, M)
    err = @. norm(adjoint(vl) * ac)

    # nx,ny = size(M)
    # err = zeros(Float64, nx, ny)

    # rx = 1:nx; ry=1:ny
    # for x in rx, y in ry
    #     yp = mod(y+1, ry) 

    #     vl = leftnull(AC[x,yp])

    #     ac = applyhac(AC[x,y],FL[x,y],FR[x,y],M[x,y])

    #     err[x,y] = norm(vl[x,y]'* ac[x,y])
    # end

    erri = maximum(err)

    return A, FL, FR, erri
end

function vumpsstep!(T::VUMPS, M::InfiniteMPO)
    # Function barrier.
    return vumpsstep!(T.A, T.FL, T.FR, M)
end

function vumps!(v::VUMPSState)
    return vumps!(v, Val(v.info.finished))
end

#mutates the vumps tensors
function vumps!(v::VUMPSState, ::Val{false})
    st = v.data
    mpo = v.mpo
    ## alg
    param = v.param
    #
    verbosity = param.verbosity
    maxiter = param.maxiter
    tol = param.tol
    ## info
    info = v.info
    #
    converged = info.converged
    error = info.error
    iterations = info.iterations

    while error > tol && iterations < maxiter
        _, _, _, error = vumpsstep!(st, mpo)
        verbosity > 0 && @info "\t Step $(iterations): error ≈ $(error)"

        iterations += 1
    end
    error > tol ? converged = false : converged = true
    info.finished = true
    return v
end

function vumps!(v::VUMPSState, ::Val{true})
    @info "This algorithm has already finished according to the parameters set"
    return v
end

function vumps_test()
    βc = log(1 + sqrt(2)) / 2
    β = 1.0 * βc

    println("β = \t", β)

    aM, aM2 = classicalisingmpo_alt(β)
    tM = TensorMap(ComplexF64.(aM), ℂ^2 * ℂ^2, ℂ^2 * ℂ^2) #mpo
    tM2 = TensorMap(aM2, ℂ^2 * ℂ^2, ℂ^2 * ℂ^2) #mpo

    M = OnLattice(MPO(), Infinite{2,2}(), tM)

    alg = initialise_state(VUMPS, M, BoundaryParameters(2, ℂ^4, 100, 1e-10))

    st = alg.data

    @assert isgauged(st.A[1])

    return calculate!(alg)

    # _, FL, FR = vumps!(A, M)

    # al = lefttensor(A)
    # ar = righttensor(A)
    # c = bondtensor(A)
    # ac = centraltensor(A)

    # z2 = expval(tM, tomatrixt(ac), tomatrixt(FL), tomatrixt(FR))
    # magn = expval(tM2, tomatrixt(ac), tomatrixt(FL), tomatrixt(FR))

    # return magn ./ z2
end

function vumps_test_dimer(x, A=MultilineInfMPS{2,2}(rand, ComplexF64, ℂ^4, ℂ^10))
    Tc = 0.65

    T = x * Tc

    println("T = \t", T)

    t = dimer(T)

    M = InfMPO{2,2}(MPOTensor.(t))

    _, FL, FR = vumps!(A, M; verbose=true, maxiter=100)

    al = lefttensor(A)
    ar = righttensor(A)
    c = bondtensor(A)
    ac = centraltensor(A)

    z2 = expval_2(t, normalize.(tomatrixt(ac)), tomatrixt(FL), tomatrixt(FR))
    ds = expval_2(dimer_d(t), normalize.(tomatrixt(ac)), tomatrixt(FL), tomatrixt(FR))

    ps = []
    for l in 1:4
        tp = dimer_p(l, t)
        p = expval_2(tp, tomatrixt(ac), tomatrixt(FL), tomatrixt(FR))
        push!(ps, abs.(p ./ z2))
    end

    return ps, abs(sum(ds ./ z2) / 4)
end

function expval(m, ac, fl, fr)
    nx, ny = size(ac)
    rx = 1:nx
    ry = 1:ny

    rv = zeros(ComplexF64, nx, ny)

    for x in rx, y in ry
        yp = mod(y + 1, ry)
        @tensoropt rvi = scalar(
            (fl[x, y])[b1; 1 a1] *
            (ac[x, y])[4 a1; a2] *
            m[1 2; 3 4] *
            (ac[x, yp]')[b2; 2 b1] *
            (fr[x, y])[3 a2; b2],
        )

        rv[x, y] = rvi
    end
    return rv
end
function expval_2(m, ac, fl, fr)
    nx, ny = size(ac)
    rx = 1:nx
    ry = 1:ny

    rv = zeros(ComplexF64, nx, ny)

    for x in rx, y in ry
        xp = mod(x + 1, rx)
        xm = mod(x - 1, rx)
        yp = mod(y + 1, ry)
        @tensoropt rvi = scalar(
            (fl[x, y])[b1; 1 a1] *
            (ac[x, y])[4 a1; a2] *
            (m[x, y])[1 2; 3 4] *
            (ac[x, yp]')[b2; 2 b1] *
            (fr[x, y])[3 a2; b2],
        )

        rv[x, y] = rvi
    end
    return rv
end

# AL[x], C[x], AR[x+1]
function get_aac(al, c, ar)
    nx, ny = size(c)
    rx = 1:nx
    ry = 1:ny

    aac = Matrix{TensorMap}(undef, 2, 2)

    for x in rx, y in ry
        xp = mod(x + 1, rx)
        @tensoropt (aac[x, y])[s1 s2 a; b] :=
            (al[x, y])[s1 a; x] * (c[x, y])[x; y] * (ar[xp, y])[s2 y; b]
    end
    return aac
end

function partition_function(M, AAC, FL, FR)
    nx, ny = size(M)
    rx = 1:nx
    ry = 1:ny

    Z2 = zeros(ComplexF64, nx, ny)

    for x in rx, y in ry
        xp = mod(x + 1, rx)
        yp = mod(y + 1, ry)

        @tensoropt begin
            Z2i = scalar(
                (FL[x, y])[b1; 1 a1] *
                (AAC[x, y])[s1 s2 a1; a2] *
                (M[x, y])[1 t1; 4 s1] *
                (M[xp, y])[4 t2; 3 s2] *
                (AAC[x, yp]')[b2; t1 t2 b1] *
                (FR[xp, y])[3 a2; b2],
            )
        end
        Z2[x, y] = Z2i
    end
    return Z2
end
function observable(M2, AAC, FL, FR, Z2)
    nx, ny = size(FL)
    rx = 1:nx
    ry = 1:ny

    obs = zeros(ComplexF64, nx, ny)

    for x in rx, y in ry
        xp = mod(x + 1, rx)
        yp = mod(y + 1, ry)

        @tensoropt obsi = scalar(
            (FL[x, y])[b1; 1 a1] *
            (AAC[x, y])[s1 s2 a1; a2] *
            M2[1 t1 t2; 3 s2 s1] *
            (AAC[x, yp]')[b2; t1 t2 b1] *
            (FR[xp, y])[3 a2; b2],
        )
        obs[x, y] = obsi / Z2[x, y]
    end
    return obs
end

function statmechmpo(β, h, D)
    M = zeros(D, D, D, D)
    for i in 1:D
        M[i, i, i, i] = 1
    end
    X = zeros(D, D)
    for j in 1:D, i in 1:D
        X[i, j] = exp(-β * h(i, j))
    end
    Xsq = sqrt(X)
    @tensor M1[a, b, c, d] :=
        M[a', b', c', d'] * Xsq[c', c] * Xsq[d', d] * Xsq[a, a'] * Xsq[b, b']

    # For computing energy: M2 is a tensor across 2 nearest neighbor sites in the lattice, whose
    # expectation value in the converged fixed point of the transfer matrix represents the energy
    Y = zeros(D, D)
    for j in 1:D, i in 1:D
        Y[i, j] = h(i, j) * exp(-β * h(i, j))
    end
    @tensor M2[a, b1, b2, c, d2, d1] :=
        M[a', b1', c1, d1'] *
        Xsq[a, a'] *
        Xsq[b1, b1'] *
        Xsq[d1', d1] *
        Y[c1, c2] *
        M[c2, b2', c', d2'] *
        Xsq[b2, b2'] *
        Xsq[d2', d2] *
        Xsq[c', c]

    Z0 = [1.0 0.0; 0.0 -1.0]
    Z = zeros(D, D)
    for j in 1:D, i in 1:D
        Z[i, j] = Z0[i, j] * exp(-β * h(i, j))
    end
    @tensor M3[a, b1, b2, c, d2, d1] :=
        M[a', b1', c1, d1'] *
        Xsq[a, a'] *
        Xsq[b1, b1'] *
        Xsq[d1', d1] *
        Z[c1, c2] *
        M[c2, b2', c', d2'] *
        Xsq[b2, b2'] *
        Xsq[d2', d2] *
        Xsq[c', c]

    return M1, M2, M3
end

function statmechmpo_alt(β, h, D)
    δ1 = zeros(D, D, D, D)
    for i in 1:D
        δ1[i, i, i, i] = 1 # kronecker delta
    end

    δ2 = zeros(D, D, D, D, D)
    for i in 1:D
        δ2[i, i, i, i, i] = 1 # kronecker delta
    end

    X = zeros(D, D)
    for j in 1:D, i in 1:D
        X[i, j] = exp(-β * h(i, j))
    end
    Xsq = sqrt(X)

    Z = [1.0, -1.0]

    @tensor M1[a, b, c, d] :=
        δ1[a', b', c', d'] * Xsq[c', c] * Xsq[d', d] * Xsq[a, a'] * Xsq[b, b']

    @tensor M2[a, b, c, d] :=
        δ2[a', b', c', d', e'] * Xsq[c', c] * Xsq[d', d] * Xsq[a, a'] * Xsq[b, b'] * Z[e']

    return M1, M2
end

function classicalisingmpo_alt(β; J=1.0, h=0.0)
    return statmechmpo_alt(
        β, (s1, s2) -> -J * (-1)^(s1 != s2) - h / 2 * (s1 == 1 + s2 == 1), 2
    )
end

function dimer(T)
    r = 1:4

    A = zeros(4, 4, 4, 4)
    for i in r
        A[i, mod(i - 1, r), mod(i - 2, r), mod(i - 3, r)] = 1
    end
    B = zeros(4, 4, 4, 4)
    for i in r
        B[i, mod(i + 1, r), mod(i + 2, r), mod(i + 3, r)] = 1
    end

    h = exp(1 / (T * 2))

    t = [1 0 0 0; 0 h 1 1; 0 1 1 1; 0 1 1 h]

    q = sqrt(t)

    @tensor Ab[a, b, c, d] := A[a', b', c', d'] * q[a', a] * q[b', b] * q[c', c] * q[d', d]
    @tensor Bb[a, b, c, d] := B[a', b', c', d'] * q[a', a] * q[b', b] * q[c', c] * q[d', d]

    sp = ℂ^4 * ℂ^4

    At = TensorMap(Ab, sp, sp)
    Bt = TensorMap(Bb, sp, sp)

    return [At Bt; Bt At]
end

function dimer_p(l, m)
    δa = zeros(4, 4)
    δa[l, l] = 1
    δ = TensorMap(δa, ℂ^4, ℂ^4)
    mo = similar(m)
    for x in 1:2, y in 1:2
        @tensor (mo[x, y])[a b; c d] := δ[a; a'] * (m[x, y])[a' b; c d]
    end
    return mo
end

function dimer_d(m)
    δa = zeros(4, 4)
    δa[1, 1] = δa[3, 3] = 1
    δa[2, 2] = δa[4, 4] = -1
    δ = TensorMap(δa, ℂ^4, ℂ^4)
    mo = similar(m)
    for x in 1:2, y in 1:2
        @tensor (mo[x, y])[a b; c d] := δ[a; a'] * (m[x, y])[a' b; c d]
    end
    return mo
end
