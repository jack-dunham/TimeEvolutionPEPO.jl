@with_kw struct TEBD
    state::PEPO
    model
    initial_state::PEPO = state
    timestep::Number = 0.01
    numsteps::Integer = 100
    tol::AbstractFloat = 0
    order::Integer = 1
    truncalg::AbstractTruncationScheme = FET()
    info::ConvergenceInfo = ConvergenceInfo()
    outfile::String = "outfile.jld2"
    outdir::String = ""
end

# Sruct MPOWII
@with_kw struct WII
    state::PEPO
    model
    initial_state::PEPO = state
    timestep::Number = 0.01
    numsteps::Integer = 100
    tol::AbstractFloat = 0
    order::Integer = 1
    truncalg::AbstractTruncationScheme = SU()
    info::ConvergenceInfo = ConvergenceInfo()
    outfile::String = "outfile.jld2"
    outdir::String = ""
end

function simulate!(tebd::TEBD)
    xmodel, ymodel = tebd.model.data_x, tebd.model.data_y

    truncalg = tebd.truncalg
    state = tebd.state

    timestep = tebd.timestep
    numsteps = tebd.numsteps
    tol = tebd.tol
    order = tebd.order

    trotterevolution!(
        state,
        xmodel,
        ymodel;
        alg=truncalg,
        numsteps=numsteps,
        timestep=timestep,
        order=order,
    )
    return nothing
end

function testthermal()
    s2 = ℂ^2

    L = Lattice{2,2,Infinite}([s2 s2; s2 s2])
    # L = Lattice{3,3,Infinite}([s2 s2 s2; s2 s2 s2; s2 s2 s2])

    pepo = PEPO(L, [0, 0, 0], ℂ^2)

    rmul!.(getstate(pepo), 0.5)

    Z = Matrix{ComplexF64}([1 0; 0 -1])
    ZZ = kron(Z, Z)

    βc = log(1 + sqrt(2)) / 2

    model = -1 / 2 * βc * ((kron(ZZ, one(ZZ)) + kron(one(ZZ), ZZ)))

    out = trotterevolution!(pepo, fill(model, L), fill(model, L))

    return out
end

function testising()
    s2 = ℂ^2

    # L = Lattice{1,1,Infinite}([s2;;])
    L = Lattice{2,2,Infinite}([s2 s2; s2 s2], false)
    # L = Lattice{3,3,Infinite}([s2 s2 s2; s2 s2 s2; s2 s2 s2])

    pepo = PEPO(L, [0, 0, -1], ℂ^6)

    Z = Matrix{ComplexF64}([1 0; 0 -1])

    mx, my = @infliouvillian 0.250 * Z[i] * Z[i + 1] 1 / 4 * (Y[i] - im * Z[i])

    out = trotterevolution!(pepo, fill(mx, L), fill(my, L))

    return out
end

function testxyz()
    s2 = ℂ^2

    # L = Lattice{1,1,Infinite}([s2;;])
    L = Lattice{2,2,Infinite}([s2 s2; s2 s2], false)
    # L = Lattice{3,3,Infinite}([s2 s2 s2; s2 s2 s2; s2 s2 s2])

    pepo = PEPO(L, [1, 0, 0], ℂ^2)

    Z = Matrix{ComplexF64}([1 0; 0 -1])

    Jx = 0.5
    Jy = 1.2
    Jz = 1.0
    Γ = 0.25

    mx, my = @infliouvillian Jx * X[i] * X[i +1] + Jy * Y[i] * Y[i + 1] + Jz * Z[i] * Z[i + 1] Γ *
        M[i]

    out = trotterevolution!(pepo, fill(mx, L), fill(my, L))

    return out
end
