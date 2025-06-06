mutable struct FSM
    const A::Dict{NTuple{4,Int},Matrix{ComplexF64}}
    const B::Dict{NTuple{4,Int},Matrix{ComplexF64}}
    const C::Dict{NTuple{4,Int},Matrix{ComplexF64}}
    const D::Dict{NTuple{4,Int},Matrix{ComplexF64}}
    h::Int
    v::Int
end

FSM(space::IndexSpace) = FSM(dim(space))
function FSM(localdim::Int)
    Id = I(localdim)

    T = Dict{NTuple{4,Int},Matrix{ComplexF64}}

    rules = FSM(T(), T(), T(), T(), 1, 1)

    # Every FSM needs at least these set
    rules.A[1, 1, 1, 1] = kron(Id, Id)
    rules.D[1, 1, 1, 1] = zero(kron(Id, Id))

    return rules
end

function fsmadd!(rules, hamiltonian::Hamiltonian; kwargs...)
    for term in oneterms(hamiltonian)
        @assert term isa LocalOp{1}
        fsmadd!(rules, term; kwargs...)
    end

    for term in terms(hamiltonian)
        @assert !(term isa LocalOp{1})
        fsmset!(rules, term; kwargs...)
    end

    return rules
end

function fsmadd!(rules, dissipator::Dissipator)
    for L in dissipator
        anticomm = kron(L' * L, one(L)) + kron(one(L), transpose(L' * L))
        rules.D[1, 1, 1, 1] += kron(L, transpose(L')) - 1 / 2 * anticomm
    end
    return rules
end

function fsmadd!(rules, liouvillian::Liouvillian)
    fsmadd!(rules, dissipator(liouvillian))
    fsmadd!(rules, hamiltonian(liouvillian); thermal=false)

    return rules
end

function fsmadd!(rules, term::LocalOp{1}; thermal=false)
    h = Matrix(term)
    Id = one(h)

    if thermal
        H1 = -0.5 * kron(h, Id)
        H2 = -0.5 * kron(Id, transpose(h))
    else
        H1 = -im * kron(h, Id)
        H2 = +im * kron(Id, transpose(h))
    end

    rules.D[1, 1, 1, 1] += H1
    rules.D[1, 1, 1, 1] += H2

    return rules
end

function fsmset!(rules, term::GaussOp2d; thermal=false)
    ms = map(t -> t.a, terms(term.gx))
    λs = map(t -> t.λ, terms(term.gx))

    a = term.a

    C = term.C
    B = term.B

    if thermal
        C1 = -0.5 * kron(C, one(C))
        C2 = -0.5 * kron(one(C), transpose(C))
    else
        C1 = -im * kron(C, one(C))
        C2 = +im * kron(one(C), transpose(C))
    end

    B1 = kron(B, one(B))
    B2 = kron(one(B), transpose(B))

    @debug "Adding terms with a = $a"

    fsm_add_gaussian!(rules, a, C1, B1, ms, λs)
    fsm_add_gaussian!(rules, a, C2, B2, ms, λs)

    return rules
end

function fsmset!(rules, term::LocalOp{2}; thermal=false)
    a = term.a

    C = term.C
    B = term.B

    if thermal
        @debug "It's thermal"
        C1 = -0.5 * kron(C, one(C))
        C2 = -0.5 * kron(one(C), transpose(C))
    else
        C1 = -im * kron(C, one(C))
        C2 = +im * kron(one(C), transpose(C))
    end

    B1 = kron(B, one(B))
    B2 = kron(one(B), transpose(B))

    @debug "Adding Ising terms with a = $a"

    fsm_add_ising!(rules, a, C1, B1)
    fsm_add_ising!(rules, a, C2, B2)

    return rules
end
function fsm_add_ising!(rules::FSM, a::Number, C::AbstractMatrix, B::AbstractMatrix)
    h = rules.h + 1
    v = rules.v + 1

    rules.C[h, 1, 1, 1] = a * C
    rules.C[1, v, 1, 1] = a * C

    rules.B[1, 1, h, 1] = B
    rules.B[1, 1, 1, v] = B

    rules.h += 1
    rules.v += 1

    return rules
end

function fsmset!(rules, term::J1J2SymmetricOp; thermal=false)
    a = term.a
    k = term.k
    CB = term.CB

    # Put C and B on equal footing
    if thermal
        CB1 = sqrt(-0.5) * kron(CB, one(CB))
        CB2 = sqrt(-0.5) * kron(one(CB), transpose(CB))
    else
        CB1 = sqrt(-im) * kron(CB, one(CB))
        CB2 = sqrt(+im) * kron(one(CB), transpose(CB))
    end

    @debug "Adding symmetric J1-J2 terms with a = $a, k = $k"

    fsm_add_j1j2_symmetric!(rules, a, k, CB1)
    fsm_add_j1j2_symmetric!(rules, a, k, CB2)

    return rules
end
function fsm_add_j1j2_symmetric!(rules::FSM, a::Number, k::Number, CB::AbstractMatrix)
    a = ComplexF64(a)

    h = rules.h + 1
    hu = rules.h + 2

    v = rules.v + 1
    vu = rules.v + 2

    Id = one(CB)

    # Need this for the NN terms only
    if iszero(a)
        @debug "a = 0"
        a = 1
    else
        rules.C[1, 1, hu, 1] = sqrt(a) * CB
        rules.C[1, 1, 1, vu] = sqrt(a) * CB
    end

    # Theres four different ways to generate each diagonal rule
    rules.A[h, v, 1, 1] = k / (4 * a) * Id
    rules.A[1, v, hu, 1] = k / (4 * a) * Id
    rules.A[h, 1, 1, vu] = k / (4 * a) * Id

    # These ONLY couple to the A terms
    rules.C[1, 1, h, 1] = sqrt(a) * CB
    rules.C[1, 1, 1, v] = sqrt(a) * CB
    rules.B[1, 1, h, 1] = sqrt(a) * CB
    rules.B[1, 1, 1, v] = sqrt(a) * CB

    rules.B[hu, 1, 1, 1] = sqrt(a) * CB
    rules.B[1, vu, 1, 1] = sqrt(a) * CB

    rules.h += 2
    rules.v += 2

    return rules
end

#TODO: This is currently not general in the sense of different approximations along different
# lattice directions
function fsm_add_gaussian!(
    rules::FSM, a::Number, C::AbstractMatrix, B::AbstractMatrix, mks, lks
)
    # Convenience
    Id = one(C)

    @debug "Adding Gaussian PEPSO" weight = a mks

    gapprox = zip(mks, lks)

    for (nh, (mk, lk)) in enumerate(gapprox)
        # mklk = mk * lk 

        c = rules.h + 2 * nh - 1
        u = rules.h + 2 * nh

        # Start going east
        rules.C[c, 1, 1, 1] = a * mk * lk * C

        # Finish going east
        rules.B[1, 1, c, 1] = B
        # Finish going west (unconventional)
        rules.B[u, 1, 1, 1] = B

        # Continue going east
        rules.A[c, 1, c, 1] = lk * Id
        # Continue going west (unconventional)
        rules.A[u, 1, u, 1] = lk * Id

        # Direction change rules
        for nv in 1:length(gapprox)
            v = rules.v + nv

            # Change from going north to going east
            rules.A[c, v, 1, 1] = mk * lk * Id
            # Change from going north to going west (unconventional)
            rules.A[1, v, u, 1] = mk * lk * Id
        end
    end
    for (nv, (mk, lk)) in enumerate(gapprox)
        v = rules.v + nv
        # Start going north
        rules.C[1, 1, 1, v] = a * mk * lk * C

        # Finish going north
        rules.B[1, v, 1, 1] = B

        # Continue going north
        rules.A[1, v, 1, v] = lk * Id
    end

    rules.h += 2 * length(gapprox)
    rules.v += length(gapprox)

    return rules
end

# Old
function fsm_add_jl1j2_symmetric(C, B, a, k, space, dt; extend=true, L=LADDER[1])
    # Need two levels per term
    T = Dict{NTuple{4,Int},Matrix{ComplexF64}}
    rules = (A=T(), B=T(), C=T(), D=T())

    D =
        kron(L, transpose(L')) -
        1 / 2 * (kron(L' * L, one(L)) + kron(one(L), transpose(L' * L)))

    X, _, Z = PAULI

    Id2 = one(D)
    # D += -im * 0.5 * (kron(Z, one(Z)) - kron(one(Z), transpose(Z)))

    rules.A[1, 1, 1, 1] = one(D)
    rules.D[1, 1, 1, 1] = D

    i = 1

    f1 = O -> kron(O, one(O))
    f2 = O -> kron(one(O), transpose(O))

    if isempty(L)
        ts = (-1 / 2, -1 / 2)
    else
        ts = (-im, im)
    end

    ts = (-im, im)

    for (ti, f) in zip(ts, (f1, f2))
        t = sqrt(0.5 * ti * a)
        # t = 1 * ti * im * a

        c = i + 1
        u = i + 2

        rules.C[c, 1, 1, 1] = t * f(C)
        rules.C[1, c, 1, 1] = t * f(C)

        rules.C[1, 1, u, 1] = t * f(C)
        rules.C[1, 1, 1, u] = t * f(C)

        rules.B[1, 1, c, 1] = t * f(B)
        rules.B[1, 1, 1, c] = t * f(B)

        rules.B[u, 1, 1, 1] = t * f(B)
        rules.B[1, u, 1, 1] = t * f(B)

        rules.A[u, u, 1, 1] = k / (40 * a) * Id2
        rules.A[1, 1, c, c] = k / (40 * a) * Id2

        rules.A[u, 1, 1, c] = k / (40 * a) * Id2
        rules.A[1, u, c, 1] = k / (40 * a) * Id2

        i += 2
    end

    @info "TNO bond dimension: $((i,i))"

    d = space
    di = dim(d)

    data = zeros(ComplexF64, di, di, di, di, i, i, i, i)

    fsmpepo!(data, rules, dt; extend=extend)

    D = reduce(oplus, fill(oneunit(d), i))

    return TensorMap(data, d * d' * d' * d, D * D * D' * D')
end
function fsmpepo(rules::FSM, space::IndexSpace, timestep::Real; kwargs...)
    d = dim(space)

    h = rules.h
    v = rules.v

    data = zeros(ComplexF64, d, d, d, d, h, v, h, v)

    # Symmetric version
    fsmpepo2!(data, rules, timestep; kwargs...)

    s = space

    Dh = reduce(oplus, fill(oneunit(s), h))
    Dv = reduce(oplus, fill(oneunit(s), v))

    pepo = TensorMap(data, s * s' * s' * s, Dh * Dv * Dh' * Dv')

    return pepo
end

function fsmpepo!(tensor::AbstractArray{<:Number,8}, rules, t::Real; extend=true)
    physdim = map(ind -> size(tensor, ind), (1, 2, 3, 4))

    incr! = (ind, rhs) -> begin
        bondind = CartesianIndex(ind)

        rhsdata = reshape(rhs, physdim)

        for physind in CartesianIndices(rhsdata)
            tensor[physind, bondind] += rhsdata[physind]
        end

        return tensor
    end

    rule_types = (:A, :B, :C, :D)

    if !extend
        for type in rule_types
            if type in (:A, :C)
                pref = 1
            elseif type in (:B, :D)
                pref = t
            end
            for (ind, rule) in getproperty(rules, type)
                incr!(ind, pref * rule)
            end
        end
    else
        for (type1, type2) in Iterators.product(rule_types, rule_types)
            dict1 = getproperty(rules, type1)
            dict2 = getproperty(rules, type2)

            for (kv1, kv2) in Iterators.product(dict1, dict2)
                ind1, rule1 = kv1
                ind2, rule2 = kv2

                any(@. !isone(ind1) && !isone(ind2)) && continue

                ind = ind1 .* ind2

                if type1 in (:A, :C)
                    if type2 in (:A, :C)
                        pref = ind1 == (1, 1, 1, 1) && ind2 == (1, 1, 1, 1) ? 1 : 1 / 2
                    elseif type2 in (:B, :D)
                        pref = t / 2
                    end
                elseif type1 in (:B, :D)
                    if type2 in (:A, :C)
                        pref = t / 2
                    elseif type2 in (:B, :D)
                        pref = t^2 / factorial(2)
                    end
                end

                incr!(ind, pref * rule1 * rule2)
            end
        end
    end

    return tensor
end

function fsmpepo2!(tensor::AbstractArray{<:Number,8}, rules, timestep::Real; extend=true)
    physdim = map(ind -> size(tensor, ind), (1, 2, 3, 4))

    incr! = (ind, rhs) -> begin
        bondind = CartesianIndex(ind)

        rhsdata = reshape(rhs, physdim)

        for physind in CartesianIndices(rhsdata)
            tensor[physind, bondind] += rhsdata[physind]
        end

        return tensor
    end

    rule_types = (:A, :B, :C, :D)

    if !extend
        for type in rule_types
            if type == :A
                pref = 1
            elseif type in (:C, :B)
                pref = sqrt(timestep)
            elseif type == :D
                pref = timestep
            end

            for (ind, rule) in getproperty(rules, type)
                incr!(ind, pref * rule)
            end
        end
    else
        for (type1, type2) in Iterators.product(rule_types, rule_types)
            dict1 = getproperty(rules, type1)
            dict2 = getproperty(rules, type2)

            for (kv1, kv2) in Iterators.product(dict1, dict2)
                ind1, rule1 = kv1
                ind2, rule2 = kv2

                # Skip if overlapping in more than one place
                any(@. !isone(ind1) && !isone(ind2)) && continue

                ind = ind1 .* ind2

                # Check for the special case of product of the identity rule with itself
                if (type1 == :A && type2 == :A)
                    # Two identities. Only occurs once in this loop.
                    if ind1 == (1, 1, 1, 1) && ind2 == (1, 1, 1, 1)
                        @assert isone(rule1)
                        @assert isone(rule2)

                        incr!(ind, rule1 * rule2)

                        continue
                    end
                end

                # See defn of methods for this function for explanations
                pref = fsmprefactor(Val(type1), Val(type2); t=timestep)
                incr!(ind, pref * rule1 * rule2)
            end
        end
    end

    return tensor
end

# Symmetric
fsmprefactor(R1, R2; t) = fsmprefactor(R2, R1; t)

function fsmprefactor(::Val{:A}, ::Val{:A}; t)
    # No starting or finishing of paths, so no t factors. Only need the factorial prefactor.
    # Note; as one of these rules may be the identity, we would usually need to check this
    # as it means a non-overlapping term and trivially needs a prefactor of 1, but as the
    # loop will combo the identity rule with any other rule twice (due to the product iterator),
    # and that 2! = 2, this rule also works here.
    return 1 / factorial(2)
end
function fsmprefactor(::Val{:A}, ::Union{Val{:C},Val{:B}}; t)
    # Starting/finishing a path in a second-order term, so we need 
    # a sqrt(t) and the factor of 2!. The remaining sqrt(2) comes from
    # the complementary operator of the path just started/finished
    return sqrt(t) / factorial(2)
end
function fsmprefactor(::Val{:A}, ::Val{:D}; t)
    # Add in a full t here as a "path" simulatenously starts and finishes.
    return t / factorial(2)
end
function fsmprefactor(::Union{Val{:C},Val{:B}}, ::Union{Val{:C},Val{:B}}; t)
    # Both finish an old path AND start a new path; so two sqrt(t) needed (one from each)
    return t / factorial(2)
end

function fsmprefactor(::Union{Val{:C},Val{:B}}, ::Val{:D}; t)
    # D rules need a full t whereas the C/B rule needs sqrt(t)
    return (t * sqrt(t)) / factorial(2)
end
function fsmprefactor(::Val{:D}, ::Val{:D}; t)
    # A full second-order term; gets the full prefactor
    return t^2 / factorial(2)
end
