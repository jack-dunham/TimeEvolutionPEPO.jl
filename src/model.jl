
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


function isingthermal()
    _, _, Z = PAULI

    Id = one(Z)
    hk = kron(Z, Id)
    hb = kron(Id, Z)

    T = BlockMatrix

    A = T(undef, 2, 2)
    B = T(undef, 2, 1)
    C = T(undef, 1, 2)
    D = T(undef, 1, 1)

    B[1, 1] = C[1, 1] = 1 / 2 * hk
    B[2, 1] = C[1, 2] = hb

    A .= Ref(zero(hk))
    D .= Ref(zero(hk))

    return MPO(A, B, C, D)
end
