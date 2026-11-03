
function xyzlindblad(; Jx=0.5, Jy=1.2, Jz=1.0, Γ=1.0)
    X, Y, Z = PAULI
    M, _ = LADDER

    eqn = Jx * ham(X, X) + Jy * ham(Y, Y) + Jz * ham(Z, Z) + sqrt(Γ) * lin(M)

    return eqn
end

function exactlindblad(; V=1.0, Γ=1.0)
    _, Y, Z = PAULI

    eqn = V / 4 * ham(Z, Z) + sqrt(Γ) / 2 * lin(Y - im * Z)

    return eqn
end

function axylindblad(; J = 1.0, Γ =1.0)
    X, Y, _ = PAULI
    M, _ = LADDER

    eqn = J / 4 * ham(X, X) + -J/4 * ham(Y, Y)  + sqrt(Γ) * lin(M)
     return eqn
end


function isingthermal(λ)
    _, _, Z = PAULI

    βc = log(1 + sqrt(2)) / 2

    Id = one(Z)
    hk = kron(Z, Id)
    hb = kron(Id, Z)

    T = Matrix{Matrix{ComplexF64}}

    A = T(undef, 2, 2)
    B = T(undef, 2, 1)
    C = T(undef, 1, 2)
    D = T(undef, 1, 1)

    B[1, 1] = βc * hk
    C[1, 1] = 1 / 2 * hk

    B[2, 1] = βc * hb
    C[1, 2] = 1 / 2 * hb

    A .= Ref(zero(hk))
    D .= Ref(zero(hk))

    A[1, 1] = A[2, 2] = λ * kron(Id, Id)

    return MPO(A, B, C, D)
end
