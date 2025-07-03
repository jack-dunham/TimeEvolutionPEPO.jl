struct PlaquetteVertex <: AbstractTerm
    p::Float64
    v::Float64
    P::Matrix{ComplexF64}
    V::Matrix{ComplexF64}
end
struct Plaquette <: AbstractTerm
    p
    P
end
struct Vertex <: AbstractTerm
    v
    V
end

function toricop(p, P, v, V)
    subA = PlaquetteVertex(p, v, P, V) + zero(Hamiltonian)
    subB = PlaquetteVertex(v, p, V, P) + zero(Hamiltonian)
    return TrotterLayer([subA subB; subB subA])
end

function fsm_add_toric!(rules, a, b, A, B)
    # For this we need two new levels on both h and v axis

    fsm_add_plaquette!(rules, a, A)
    fsm_add_vertex!(rules, b, B)

    rules.h += 2
    rules.v += 2

    return rules
end
function fsm_add_plaquette!(rules, a, A)
    # For this we need two new levels on both h and v axis
    h1 = rules.h + 1
    h2 = rules.h + 2

    v1 = rules.v + 1
    v2 = rules.v + 2

    rules.C[h1, v1, 1, 1] = sqrt(a) * A
    rules.B[1, 1, h2, v2] = sqrt(a) * A

    return rules
end
function fsm_add_vertex!(rules, a, A)
    # For this we need two new levels on both h and v axis
    h1 = rules.h + 1
    h2 = rules.h + 2

    v1 = rules.v + 1
    v2 = rules.v + 2

    rules.A[h2, 1, 1, v1] = A
    rules.A[1, v2, h1, 1] = A

    return rules
end
