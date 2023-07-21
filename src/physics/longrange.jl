function approx_sum_as_exp(α, n; N=50)
    f = N -> 1 / (N)^(α)
    F = zeros(Float64, N - n + 1, n)

    for r in axes(F, 1)
        for c in axes(F, 2)
            F[r, c] = f(r + c - 1)
        end
    end

    QR = qr(F)
    U = QR.Q
    # V = QR.R

    U1 = U[1:(N - n), 1:n]
    U2 = U[2:end, 1:n]

    Λ = eigen(pinv(U1) * U2).values

    A = zeros(Float64, N, n)
    for c in axes(A, 1)
        A[c, :] .= Λ .^ (c)
    end

    invsol = [f(k) for k in 1:N]

    sol = pinv(A) * invsol

    # return k -> (A * sol)[k]
    return sol, Λ
end

powerlaw(x; α=3) = 1 / (x)^(α)


function powerlawwii
    
end
