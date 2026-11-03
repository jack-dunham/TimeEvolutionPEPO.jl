@testsetup module IterativeSetup
using Reexport
@reexport using TimeEvolutionPEPO
@reexport using TensorKit
@reexport using LinearAlgebra
@reexport using Logging

export TEP, quiet, figmodel, occupation, weightvalues, dense_liouvillian, NZ

const TEP = TimeEvolutionPEPO

quiet(f) = with_logger(f, ConsoleLogger(stderr, Logging.Warn))

# The occupation n = (Z + 1)/2 of the state at basis index 1.
const NZ = ComplexF64.(0.5 * (PAULI[3] + I))

# Kac normalisation of r⁻³ on the square lattice, 4 ζ(3/2) β(3/2).
const NKAC3 = 4 * 2.6123753486854883 * 0.8645026534612020

# Fig. 9 of arXiv 2512.01781: one layer per Gaussian of the fitted r⁻³ coupling, each carrying an
# equal share of the field and the decay.
function figmodel(J)
    gs = collect(TEP.expandterm(quiet(() -> PowOp2d(PAULI[3], PAULI[3], 3.0; gtol = 1e-5))))
    layers = map(gs) do g
        H = zero(Hamiltonian) + (J / NKAC3) * g + (1 / length(gs)) * LocalOp(-PAULI[1])
        return TrotterLayer(H + (1 / length(gs)) * Dissipator(LADDER[1]))
    end
    return Model(layers)
end

function occupation(state, site = (1, 1); χ = 9)
    dm = quiet(() -> DensityMatrix(state, χ))
    return real(expval(partialtrace(dm, site), NZ))
end

weightvalues(w) = real.(diag(convert(Array, w)))

function dense_liouvillian(H::AbstractMatrix, jumps::AbstractVector)
    d = size(H, 1)
    Id = Matrix{ComplexF64}(I, d, d)
    G = -im * (kron(Id, H) - kron(transpose(H), Id))
    for L in jumps
        LdL = L' * L
        G += kron(conj(L), L) - (kron(Id, LdL) + kron(transpose(LdL), Id)) / 2
    end
    return G
end
end # IterativeSetup

@testitem "IterativeSU settings" setup = [IterativeSetup] begin
    @test_throws ArgumentError IterativeSU(; protect = :all)
    @test_throws ArgumentError IterativeSU(; mu = -1.0)
    @test_throws ArgumentError IterativeSU(; maxiter = 0)
    @test DampedUpdate().iterative == IterativeSU()
    @test DampedUpdate(; iterative = IterativeSU(; mu = 3.0)).iterative.mu == 3.0
end

@testitem "Long-range layers in the decomposition" setup = [IterativeSetup] begin
    P, dt = ComplexSpace(2), 1.25e-2
    model = figmodel(1.5)
    decomposition = trotterise(model, P, dt)
    @test length(decomposition) == length(model)
    @test all(entry -> entry isa PEPSOLayer, decomposition)

    # a nearest-neighbour layer next to them keeps its two axis sweeps
    push!(model, TrotterLayer(zero(Hamiltonian) + LocalOp(-PAULI[1]) + Dissipator(LADDER[1])))
    decomposition = trotterise(model, P, dt)
    @test count(entry -> entry isa PEPSOLayer, decomposition) == length(model) - 1
    @test last.(decomposition[(end - 1):end]) == [1, 2]

    # the finite-state machine builds one uniform tensor, so a long-range layer cannot vary by site
    g = model.layers[1][1, 1]
    @test_throws ArgumentError trotterise(TrotterLayer(fill(g, 2, 1)), P, dt)

    state = productpepo(ComplexF64[0 0; 0 1], 1)
    sim = Simulation(state, figmodel(1.5); timestep = dt)
    @test_throws ArgumentError simulate!(sim; numsteps = 1, zeta = 10.0, verbose = false)
end

@testitem "Long-range layer without coupling against the one-site map" setup = [IterativeSetup] begin
    # With J = 0 every site evolves alone under H = -X and decay. `fsmpepo` builds a layer carrying
    # a share h L of the one-site generator as 1 + hL + h²L²/2, so that map applied once per layer is exact.
    dt, numsteps = 1.25e-2, 24
    model = figmodel(0.0)
    sim = Simulation(
        productpepo(ComplexF64[0 0; 0 1], 1), model; timestep = dt,
        alg = DampedUpdate(; trunc = truncrank(2))
    )
    quiet(() -> simulate!(sim; numsteps, verbose = false))
    G = (dt / length(model)) * dense_liouvillian(-PAULI[1], [LADDER[1]])
    M = I + G + G^2 / 2
    expected = reshape(M^(length(model) * numsteps) * vec(ComplexF64[0 0; 0 1]), 2, 2)
    # the cut's factors come from square roots of eigenvalues, so they resolve singular values only to
    # about √eps of the largest, and the cut keeps a second direction of that size
    @test occupation(quantumstate(sim)) ≈ real(expected[1, 1]) atol = 1e-8
    # and it is within the expansion's error of the exact exponential
    exact = reshape(exp(numsteps * length(model) * G) * vec(ComplexF64[0 0; 0 1]), 2, 2)
    @test occupation(quantumstate(sim)) ≈ real(exact[1, 1]) atol = 1e-5
end

@testitem "Iterative simple update against the pre-port implementation" setup = [IterativeSetup] begin
    # Reference: `dev/fig9/core.jl` (`itrsu` and `itrsu_trace`, 1×1 cell) from the ground state,
    # D = 3, 16 steps of Δt = 0.0125 with weights converged to 1e-10.
    REF = (
        none = (
            λx = [1.0, 0.0016721985591049264, 0.0016643909008889654],
            λy = [1.0, 0.0016646406240010785, 0.0016561694993428668],
            n = 0.03481878346482034,
        ),
        twosite = (
            λx = [1.0, 0.001666415255514956, 0.0016588961273208472],
            λy = [1.0, 0.0016780121049346603, 0.0016697554727823434],
            n = 0.03481734382415401,
        ),
    )
    for (protect, ref) in pairs(REF)
        iterative = IterativeSU(; tol = 1e-10, maxiter = 200, protect, mu = 10.0)
        sim = Simulation(
            productpepo(ComplexF64[0 0; 0 1], 1), figmodel(1.5); timestep = 1.25e-2,
            alg = DampedUpdate(; trunc = truncrank(3), iterative)
        )
        quiet(() -> simulate!(sim; numsteps = 16, verbose = false))
        w = bondweights(sim)
        @test weightvalues(w[1, 1, 1]) ≈ ref.λx atol = 1e-8
        @test weightvalues(w[2, 1, 1]) ≈ ref.λy atol = 1e-8
        @test occupation(quantumstate(sim)) ≈ ref.n atol = 1e-6
    end
end

@testitem "Uniform model on a 2×2 cell" setup = [IterativeSetup] begin
    # two of the ten layers; the :twosite cut also runs the traced-network BP over the cell
    model = Model(figmodel(1.5).layers[1:2])
    iterative = IterativeSU(; tol = 1e-10, maxiter = 200, protect = :twosite, mu = 10.0)
    alg = DampedUpdate(; trunc = truncrank(3), iterative)
    sims = map(((1, 1), (2, 2))) do cell
        state = productpepo(fill((0.0, 0.0, -1.0), cell...), 1)
        sim = Simulation(state, model; timestep = 1.25e-2, alg)
        return quiet(() -> simulate!(sim; numsteps = 4, verbose = false))
    end
    w1, w2 = bondweights.(sims)
    # the subleading weights are about 3e-7, so each is compared relative to its own size
    for d in 1:2, r in 1:2, c in 1:2
        @test all(isapprox.(weightvalues(w2[d, r, c]), weightvalues(w1[d, 1, 1]); rtol = 1e-6))
    end
    n1 = occupation(quantumstate(sims[1]))
    dm = quiet(() -> DensityMatrix(quantumstate(sims[2]), 9))
    for r in 1:2, c in 1:2
        @test real(expval(partialtrace(dm, (r, c)), NZ)) ≈ n1 atol = 1e-10
    end
end

@testitem "Protected and soft cuts" setup = [IterativeSetup] begin
    using Random
    Random.seed!(7)
    d, D, k = 12, 5, 4
    V = ComplexSpace(d)
    R1 = TensorMap(randn(ComplexF64, d, d), V ← V)
    R2 = TensorMap(randn(ComplexF64, d, d), V ← V)
    wL, wR = randn(ComplexF64, k, d), randn(ComplexF64, d, k)
    R1a, R2a = TEP._matrix(R1), TEP._matrix(R2)
    M = R1a * R2a

    # The cut is M̃ = R1 Π R2 with Π = P1 P2; `change` is the relative change of wL M wR it causes.
    Π(cut) = TEP._matrix(cut[1]) * TEP._matrix(cut[2])
    change(cut) = norm(wL * Π(cut) * wR - wL * wR) / norm(wL * wR)
    # The weights are the cut bond's own: R1 P1 is an isometry times √s, and so is (P2 R2)'.
    function consistent(cut)
        s = weightvalues(cut[3])
        L, R = R1a * TEP._matrix(cut[1]), TEP._matrix(cut[2]) * R2a
        return max(norm(L' * L - Diagonal(s)), norm(R * R' - Diagonal(s))) / norm(s) < 1e-10
    end

    plain = TEP._svdcut(R1, R2; trunc = truncrank(D))
    exact = TEP._protectedcut(R1, R2, wL, wR; D)
    @test change(exact) < 1e-10
    @test consistent(plain) && consistent(exact)
    @test change(plain) > 1e-3

    @test Π(TEP._softcut(R1, R2, wL, wR; D, mu = 0.0)) ≈ Π(plain)
    changes = [change(TEP._softcut(R1, R2, wL, wR; D, mu)) for mu in (1.0, 1e2, 1e4, 1e6)]
    @test issorted(changes; rev = true)
    @test last(changes) < 1e-4
    @test all(mu -> consistent(TEP._softcut(R1, R2, wL, wR; D, mu)), (1.0, 1e4))

    # the truncation error is that of the cut actually made, the SVD's being the smallest
    @test plain[4] ≈ norm(M - R1a * Π(plain) * R2a) / norm(M)
    @test exact[4] ≈ norm(M - R1a * Π(exact) * R2a) / norm(M)
    @test exact[4] > plain[4]
end
