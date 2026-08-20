@testsetup module ModelSetup
using Reexport
@reexport using TimeEvolutionPEPO
@reexport using PEPSKit
@reexport using TensorKit
@reexport using LinearAlgebra
@reexport using Logging

export quiet
quiet(f) = with_logger(f, ConsoleLogger(stderr, Logging.Error))
end # ModelSetup

@testitem "Trotter layers" setup = [ModelSetup] begin
    X, Y, Z = PAULI

    @testset "a bare term is a 1x1 layer" begin
        layer = TrotterLayer(LocalOp(Z, Z))
        @test size(layer) == (1, 1)
        @test layer.axis == 0
        @test layer[1, 1] isa Hamiltonian
        # A 1x1 layer wraps periodically, so it serves any unit cell.
        @test layer[3, 7] === layer[1, 1]
    end

    @testset "a matrix of terms keeps its shape" begin
        data = [LocalOp(Z, Z) LocalOp(X, X); LocalOp(X, X) LocalOp(Z, Z)]
        layer = TrotterLayer(data, 1)
        @test size(layer) == (2, 2)
        @test layer.axis == 1
        @test layer[3, 3] === layer[1, 1]
    end

    @testset "a non-axial term cannot be pinned to an axis" begin
        gauss = quiet() do
            GaussOp2d(Z, Z, 1.0)
        end
        @test_throws ArgumentError TrotterLayer(gauss, 1)
    end

    @testset "a dissipative layer can be pinned to an axis" begin
        layer = TrotterLayer(Liouvillian(zero(Hamiltonian) + LocalOp(Z, Z)), 1)
        @test layer.axis == 1

        layer = TrotterLayer(zero(Hamiltonian) + LocalOp(Z, Z) + Dissipator(Z), 1)
        @test layer.axis == 1
    end

    @testset "a dissipative Liouvillian and an equivalent dissipative Ising agree on pinning" begin
        liouvillian = TrotterLayer(
            Liouvillian(zero(Hamiltonian) + LocalOp(Z, Z), Dissipator(Z)), 1
        )
        ising = TrotterLayer(Ising(LocalOp(Z, Z), Dissipator(Z)), 1)
        @test liouvillian.axis == 1
        @test ising.axis == 1
    end
end

@testitem "Model container" setup = [ModelSetup] begin
    X, Y, Z = PAULI

    @testset "layers are appended, never distributed" begin
        model = Model(LocalOp(X, X), LocalOp(Y, Y))
        @test length(model) == 2
        push!(model, LocalOp(Z, Z))
        @test length(model) == 3
        # push! appends whether or not the term is one-local.
        push!(model, LocalOp(X))
        @test length(model) == 4
    end

    @testset "distribute! spreads over existing layers" begin
        model = Model(LocalOp(X, X), LocalOp(Y, Y))
        distribute!(model, LocalOp(Z))
        # The layer count is unchanged; the numerical share is checked in the next testitem, once
        # trotterise accepts a Model.
        @test length(model) == 2
        @test length(oneterms(model[1][1, 1])) == 1
    end

    @testset "distributing over no layers is an error" begin
        @test_throws ArgumentError distribute!(Model{Hamiltonian}(), LocalOp(Z))
    end

    @testset "a Hamiltonian model rejects dissipation" begin
        model = Model{Hamiltonian}()
        push!(model, LocalOp(X, X))
        @test_throws ArgumentError push!(model, Dissipator(Z))
    end

    @testset "distribute! rejects a term outside the model's family" begin
        model = Model{Hamiltonian}()
        push!(model, LocalOp(X, X))
        @test_throws ArgumentError distribute!(model, Dissipator(Z))
        # The rejected call must not have mutated any layer before throwing.
        @test length(model) == 1
        @test dissipator(model[1][1, 1]) === nothing
    end

    @testset "distribute! leaves the model untouched when a later layer rejects the axial merge" begin
        # `free` has no pinned axis, so merging the non-axial share succeeds; `pinned` is pinned to
        # axis 1, so the same merge makes it non-axial and throws.
        gauss = quiet() do
            GaussOp2d(Z, Z, 1.0)
        end
        free = TrotterLayer(convert(Hamiltonian, LocalOp(Y, Y)))
        pinned = TrotterLayer(convert(Hamiltonian, LocalOp(X, X)), 1)
        model = Model([free, pinned])
        originalfree, originalpinned = model[1], model[2]

        @test_throws ArgumentError distribute!(model, gauss)
        @test model[1] === originalfree
        @test model[2] === originalpinned
    end
end

@testitem "Trotter decomposition of a model" setup = [ModelSetup] begin
    X, Y, Z = PAULI
    P = ComplexSpace(2)

    @testset "a uniform layer yields a 1x1 propagator matrix" begin
        decomposition = trotterise(LocalOp(Z, Z), P, 0.1)
        @test length(decomposition) == 2                   # one layer over both axes
        propagators, axis = first(decomposition)
        @test size(propagators) == (1, 1)
        @test axis == 1
    end

    @testset "a site-dependent layer yields one propagator per site" begin
        data = [LocalOp(Z, Z) 0.5 * LocalOp(Z, Z); 0.5 * LocalOp(Z, Z) LocalOp(Z, Z)]
        decomposition = trotterise(Model([TrotterLayer(data)]), P, 0.1)
        propagators, = first(decomposition)
        @test size(propagators) == (2, 2)
        @test !(propagators[1, 1] ≈ propagators[1, 2])
    end

    @testset "distribute! gives each layer a 1/n share" begin
        # Moved here from Task 6: the check needs trotterise to accept a Model.
        distributed = Model(LocalOp(X, X), LocalOp(Y, Y))
        distribute!(distributed, LocalOp(Z))
        byhand = Model(LocalOp(X, X) + 0.5 * LocalOp(Z), LocalOp(Y, Y) + 0.5 * LocalOp(Z))
        got = trotterise(distributed, P, 0.1)
        want = trotterise(byhand, P, 0.1)
        @test length(got) == length(want)
        @test all(first(g)[1, 1] ≈ first(w)[1, 1] for (g, w) in zip(got, want))
    end

    @testset "the one-local part is halved across the two axes" begin
        # A layer applied on both axes must halve its one-local part; pinned to one axis it must not.
        both = first(first(trotterise(zero(Hamiltonian) + LocalOp(X), P, 0.1)))[1, 1]
        pinned = first(first(trotterise([(zero(Hamiltonian) + LocalOp(X)) => 1], P, 0.1)))[1, 1]
        @test !(both ≈ pinned)
    end
end

@testitem "staggered field against exp(tL)" setup = [ModelSetup] begin
    X, Y, Z = PAULI
    P = ComplexSpace(2)

    # A one-local Lindbladian on one sublattice only. At D = 1 with no coupling every site evolves
    # independently, so both sublattices are checked against dense linear algebra.
    jump = 0.5 * (Y - im * Z)
    v0 = (0.0, 0.0, -1.0)
    # nsteps = 5 (t = 0.1) only separates the sublattices by 0.076; 15 steps clears the 0.1 bar.
    dt, nsteps = 0.02, 15

    active = zero(Hamiltonian) + LocalOp(X) + Dissipator(jump)
    idle = zero(Hamiltonian) + LocalOp(X)   # hamiltonian-only sublattice, no dissipation

    layer = TrotterLayer([active idle; idle active])
    sim = quiet(() -> Simulation(productpepo(fill(v0, 2, 2), 1), Model([layer]); timestep = dt))
    quiet(() -> simulate!(sim; numsteps = nsteps, verbose = false))

    dense(H, jumps) = begin
        d = 2
        Id = Matrix{ComplexF64}(I, d, d)
        G = -im * (kron(Id, H) - kron(transpose(H), Id))
        for L in jumps
            LdL = L' * L
            G += kron(conj(L), L) - (kron(Id, LdL) + kron(transpose(LdL), Id)) / 2
        end
        G
    end

    evolve(G) = reshape(exp(nsteps * dt * G) * vec(bloch(v0)), 2, 2)
    want_active = evolve(dense(Matrix(X), [jump]))
    want_idle = evolve(dense(Matrix(X), Matrix{ComplexF64}[]))

    dm = quiet(() -> DensityMatrix(quantumstate(sim), 4))
    got_active = reshape(convert(Array, partialtrace(dm, (1, 1))), 2, 2)
    got_idle = reshape(convert(Array, partialtrace(dm, (1, 2))), 2, 2)

    @test got_active ≈ want_active atol = 1e-8
    @test got_idle ≈ want_idle atol = 1e-8
    # The two sublattices must actually differ, or the test proves nothing.
    @test norm(want_active - want_idle) > 0.1
end

@testitem "long-range propagator as an InfinitePEPSO" setup = [ModelSetup] begin
    X, Y, Z = PAULI
    P = ComplexSpace(2)
    jump = 0.5 * (Y - im * Z)
    v0 = (0.0, 0.0, -1.0)

    # Dense Lindblad generator on vec(rho), built independently of the package.
    function dense(H, jumps)
        n = size(H, 1)
        Id = Matrix{ComplexF64}(I, n, n)
        G = -im * (kron(Id, H) - kron(transpose(H), Id))
        for L in jumps
            LdL = L' * L
            G += kron(conj(L), L) - (kron(Id, LdL) + kron(transpose(LdL), Id)) / 2
        end
        return G
    end

    @testset "a one-local generator reproduces exp(t L)" begin
        # At bond dimension one the PEPSO carries no coupling, so every site evolves under its own
        # generator and the result is exactly comparable to dense linear algebra. This pins the
        # physical leg convention: the rules are vectorised ket-slowest, so a (bra, ket) ordering
        # would return the transpose of the correct density matrix.
        model = zero(Hamiltonian) + LocalOp(X) + Dissipator(jump)
        G = dense(Matrix(X), [jump])

        residual = map((0.04, 0.02, 0.01)) do dt
            S = fsmpepso(model, P, dt; unitcell = (2, 2))
            evolved, = compress(
                (productpepo(fill(v0, 2, 2), 1), S), LocalTruncation(truncspace(ComplexSpace(1)))
            )
            got = reshape(
                convert(Array, quiet(() -> partialtrace(DensityMatrix(evolved, 4), (1, 1)))), 2, 2
            )
            return norm(got - reshape(exp(dt * G) * vec(bloch(v0)), 2, 2))
        end

        # The finite-state machine builds a symmetric splitting, so the local error is O(dt^3):
        # halving dt must divide the residual by roughly eight. A wrong physical leg ordering gives
        # an O(1) residual instead, and a wrong prefactor an O(dt) one.
        @test all(residual .< 1e-4)
        @test all(@. 6 < residual[1:(end - 1)] / residual[2:end] < 11)
    end

    @testset "a nearest-neighbour coupling reproduces the exact derivative" begin
        # Bond dimension three, so this exercises the virtual legs the one-local case cannot.
        J = 1.5
        model = J / 4 * LocalOp(Z, Z) + Dissipator(jump)
        Zm = Matrix(Z)

        rho = bloch(v0)
        onelocal = real(tr(Zm * reshape(dense(zeros(ComplexF64, 2, 2), [jump]) * vec(rho), 2, 2)))
        pair = reshape(dense((J / 4) * kron(Zm, Zm), Matrix{ComplexF64}[]) * vec(kron(rho, rho)), 4, 4)
        want = onelocal + 4 * real(tr(kron(Zm, Matrix{ComplexF64}(I, 2, 2)) * pair))

        got = map((0.004, 0.002)) do dt
            S = fsmpepso(model, P, dt; unitcell = (2, 2))
            state = productpepo(fill(v0, 2, 2), 1)
            before = real(expval(quiet(() -> partialtrace(DensityMatrix(state, 12), (1, 1))), Zm))
            evolved, = compress((state, S), LocalTruncation(truncspace(ComplexSpace(8))))
            after = real(expval(quiet(() -> partialtrace(DensityMatrix(evolved, 12), (1, 1))), Zm))
            return (after - before) / dt
        end

        @test abs(got[2] - want) < 1e-3
        # Halving dt must halve the finite-difference error, or the agreement is a coincidence.
        @test abs(got[2] - want) < 0.6 * abs(got[1] - want)
    end

    @testset "a long-range term lands on the axes it was given" begin
        # An isotropic Gaussian: the term's own profile is the reference, and it reproduces
        # exp(-mu r^2) exactly, so the sum-of-exponentials fit is not the weak link here.
        term = quiet(() -> GaussOp2d(Z, Z, 0.4))
        @test real(term(1, 0)) ≈ exp(-0.4) rtol = 1e-4
        @test real(term(2, 0)) ≈ exp(-0.4 * 4) rtol = 1e-4
        @test real(term(1, 0)) ≈ real(term(0, 1)) rtol = 1e-4

        S = fsmpepso(term, P, 0.02)
        east = PEPSKit.virtualspace(S[1, 1, 1], PEPSKit.EAST)
        north = PEPSKit.virtualspace(S[1, 1, 1], PEPSKit.NORTH)

        # `fsm_add_gaussian!` gives the east-west chain a state per exponential in each direction
        # and the north-south chain one per exponential, so east carries 2k+1 states against north's
        # k+1. That asymmetry is what pins which virtual leg carries which chain: `fsmpepo` orders
        # its legs east-south-west-north, so dropping the swap in `fsmpepso` puts the doubled chain
        # on north instead and fails this.
        @test dim(east) == 2 * dim(north) - 1
        @test isdual(east) && isdual(north)      # `compress` rejects the propagator otherwise
    end

    @testset "each axis takes its own decay" begin
        # `fsmset!(::GaussOp2d)` reads gx for the east-west chain and gy for north-south, so the two
        # lattice directions may decay differently and need not use the same number of exponentials.
        aniso = quiet(() -> GaussOp2d(Z, Z, 0.4, 1.6))
        nx = length(aniso.gx.terms)
        ny = length(aniso.gy.terms)
        @test nx != ny                                   # the two fits differ, or this proves nothing

        S = fsmpepso(aniso, P, 0.02)
        east = PEPSKit.virtualspace(S[1, 1, 1], PEPSKit.EAST)
        north = PEPSKit.virtualspace(S[1, 1, 1], PEPSKit.NORTH)

        # `fsmset!` calls `fsm_add_gaussian!` once per half of the vectorised commutator, and the
        # east-west chain carries a state per exponential per direction. Reading gx for both axes
        # would give north 1 + 2nx instead.
        @test dim(east) == 1 + 4 * nx
        @test dim(north) == 1 + 2 * ny

        # Both axes reproduce their own exp(-mu r^2), so the anisotropy is real and not a relabelling.
        @test real(aniso(1, 0)) ≈ exp(-0.4) rtol = 1e-4
        @test real(aniso(0, 1)) ≈ exp(-1.6) rtol = 1e-4
        @test real(aniso(2, 0)) ≈ exp(-0.4 * 4) rtol = 1e-4
        @test real(aniso(0, 2)) ≈ exp(-1.6 * 4) rtol = 1e-4
    end

    @testset "the permutation-symmetric build has equal bonds on both axes" begin
        # `symmetric = true` gives each axis two finite-state-machine levels instead of two
        # east-west and one north-south, so an isotropic term gets the same bond dimension both ways.
        iso = quiet(() -> GaussOp2d(Z, Z, 0.4))
        nx = length(iso.gx.terms)

        asym = fsmpepso(iso, P, 0.02)
        sym = fsmpepso(iso, P, 0.02; symmetric = true)
        eastof(S) = dim(PEPSKit.virtualspace(S[1, 1, 1], PEPSKit.EAST))
        northof(S) = dim(PEPSKit.virtualspace(S[1, 1, 1], PEPSKit.NORTH))

        @test eastof(asym) == 2 * northof(asym) - 1        # one level saved on north-south
        @test eastof(sym) == northof(sym) == 1 + 4 * nx    # two levels on each

        # Both must build the same operator. Each off-axis displacement is reached north-first and
        # south-first rather than only north-first, so the turn count doubles while the summed turn
        # weight is unchanged -- that sum is what the halved turn weight has to preserve, and getting
        # it wrong doubles the off-axis couplings while leaving the on-axis profile untouched.
        isturn(k) = ((k[1] > 1) + (k[3] > 1) == 1) && ((k[2] > 1) + (k[4] > 1) == 1)
        function turns(symmetric)
            rules = TimeEvolutionPEPO.FSM(P)
            TimeEvolutionPEPO.fsmset!(rules, iso; symmetric)
            ks = [k for k in keys(rules.A) if isturn(k)]
            return length(ks), sum(abs(rules.A[k][1, 1]) for k in ks)
        end
        na, wa = turns(false)
        ns, ws = turns(true)
        @test ns == 2 * na
        @test ws ≈ wa rtol = 1e-10

        # The role split is what keeps the on-axis pairs counted once: west is only ever entered by
        # a turn, and south only ever leaves one.
        rules = TimeEvolutionPEPO.FSM(P)
        TimeEvolutionPEPO.fsmset!(rules, iso; symmetric = true)
        single(d, p) = any(k -> k[p] > 1 && count(>(1), k) == 1, keys(d))
        @test !single(rules.C, 3)      # no chain starts going west
        @test single(rules.B, 1)       # but one finishes going west
        @test single(rules.C, 2)       # a chain starts going south
        @test !single(rules.B, 4)      # but none finishes going south
    end
end
