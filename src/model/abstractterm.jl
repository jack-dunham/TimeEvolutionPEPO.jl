abstract type AbstractTerm end
abstract type AbstractSumOfTerms <: AbstractTerm end

abstract type AbstractSingleTerm <: AbstractTerm end
abstract type AbstractAxialTerm <: AbstractSingleTerm end
abstract type AbstractRadialTerm <: AbstractSingleTerm end
abstract type AbstractManhattanTerm <: AbstractSingleTerm end

abstract type AbstractModel end

## AbstractTerm

Base.length(h::AbstractTerm) = length(terms(h))
Base.iterate(h::AbstractTerm, state = 1) = iterate(terms(h), state)

Base.firstindex(h::AbstractTerm) = 1
Base.lastindex(h::AbstractTerm) = length(h)

Base.getindex(h::AbstractTerm, i) = getindex(terms(h),i)

isexpandable(::AbstractTerm) = false

isaxial(::AbstractTerm) = true
isaxial(::AbstractRadialTerm) = false
isaxial(::AbstractManhattanTerm) = false

Base.:+(A::AbstractTerm, B::AbstractTerm) = +(B, A)

struct LocalOp{N} <: AbstractSingleTerm
    a::Float64
    CAB::NTuple{N,Matrix{ComplexF64}}
    function LocalOp(a, CAB::NTuple{N}) where {N}
        c = prod(op -> norm(op, Inf), CAB)
        new{N}(a * c, map(op -> normalize(op, Inf), CAB))
    end
end
terms(t::AbstractSingleTerm) = (t,)
terms(t::LocalOp{1}) = ()
oneterms(t::AbstractTerm) = LocalOp{1}[]
oneterms(t::LocalOp{1}) = LocalOp{1}[t]

function Base.hash(term::LocalOp, h::UInt)
    for op in term.CAB
        h = hash(op, h)
    end
    h = hash(term.a, h)
    return hash(typeof(term), h)
end

Base.:(==)(l1::LocalOp, l2::LocalOp) = hash(l1) == hash(l2)
Base.:(≈)(l1::LocalOp{N}, l2::LocalOp{N}) where {N} = all(l1.CAB .≈ l2.CAB) && l1.a ≈ l2.a

function isaxial(h::AbstractSumOfTerms)
    for term in terms(h)
        if !isaxial(term)
            # Short-circuit if we find a non-axial term
            return false
        end
    end
    return true
end

Base.zero(h::AbstractSumOfTerms) = zero(typeof(h))

"""
$(TYPEDEF)

Concrete type representing a Hamiltonian. Users should construct this object by adding terms
together using `localop`, `expop`, `powop2d` etc, rather than directly.
"""
struct Hamiltonian{T<:Tuple{Vararg{AbstractSingleTerm}}} <: AbstractSumOfTerms
    terms::T
    oneterms::Vector{LocalOp{1}}
end
Hamiltonian(term::AbstractSingleTerm, oneterms) = Hamiltonian((term,),oneterms)

function Base.show(io::IO, ::MIME"text/plain", eqn::Hamiltonian)

    n1 = length(oneterms(eqn))
    n2 = length(terms(eqn))

    println(io, "Hamiltonian containing:")
    println(io, "  $n1 on-site term(s)")
    return print(io, "  $n2 local term(s)")
end

terms(h::Hamiltonian) = h.terms

# Hamiltonians have no lindblad operators associated with them
dissipator(::Hamiltonian) = nothing

Hamiltonian() = Hamiltonian((), LocalOp{1}[])
Base.zero(::Type{<:Hamiltonian}) = Hamiltonian()

oneterms(h::Hamiltonian) = h.oneterms

Hamiltonian(h) = Hamiltonian(h, LocalOp{1}[])
Hamiltonian(h0::LocalOp{1}) = Hamiltonian((), [h0])

scalelocal(h::Hamiltonian, a) = @set h.oneterms .*= a

function Base.:+(A::Hamiltonian, B::AbstractTerm)
    H1 = TupleTools.vcat(terms(A), terms(B))
    H0 = vcat(oneterms(A), oneterms(B))
    return Hamiltonian(H1, H0)
end

struct Dissipator <: AbstractSumOfTerms
    terms::Vector{Matrix{ComplexF64}}
end

Base.:*(a::Number, D::Dissipator) = Dissipator(map(t -> sqrt(a) * t, terms(D)))
Base.:+(D1::Dissipator,D2::Dissipator) = Dissipator(vcat(D1.terms,D2.terms))

Dissipator(L::Vararg{AbstractMatrix}) = Dissipator(collect(L))

terms(D::Dissipator) = D.terms

isaxial(::Dissipator) = false


function vectorise(dissipator::Dissipator)
    anticomm = L -> kron(L' * L, one(L)) + kron(one(L), transpose(L' * L))
    f = twolocal(L -> kron(L, transpose(L')) - 1 / 2 * anticomm(L))
    rv = sum(f, dissipator)
    return rv
end

"""
$(TYPEDEF)

Concrete type representing a Liouvillian super-operator.

## Constructors

    Liouvillian(H::Hamiltonian, Ls::Varargs{AbstractMatrix}; tied = false)

Construct `Liouvillian` from `Hamiltonian` H and lindblad operators Ls. The keyword argument
`tied` defines the trotterisation of the liouvillian. If `true` then all terms in the
Hamiltonian as well as all the lindblad operators are exponentiated together rather than
decomposed using the trotter decomposition. This is useful if one wishes to represent a
strong symmetry. 
"""
struct Liouvillian{T} <: AbstractSumOfTerms
    hamiltonian::Hamiltonian{T}
    dissipator::Dissipator
end

dissipator(L::Liouvillian) = L.dissipator
terms(L::Liouvillian) = terms(L.hamiltonian)
oneterms(L::Liouvillian) = oneterms(L.hamiltonian)

Liouvillian(H) = Liouvillian(H, [])
Liouvillian(H, L) = Liouvillian(H, Dissipator(L))
Liouvillian(H::Hamiltonian, Ls::Vararg{<:Matrix}) = Liouvillian(H, collect(Ls))

Base.convert(::Type{Liouvillian}, H::Hamiltonian) = Liouvillian(H)
Base.convert(::Type{Liouvillian}, term) = Liouvillian(convert(Hamiltonian, term))
Base.convert(::Type{Liouvillian}, term::Dissipator) = Liouvillian(zero(Hamiltonian),term)

function scalelocal(L::Liouvillian, a) 

    @reset L.hamiltonian.oneterms .*= a
    @reset L.dissipator *= a

    return L
end

function Base.show(io::IO, ::MIME"text/plain", eqn::Liouvillian)

    n1 = length(oneterms(eqn))
    n2 = length(terms(eqn))

    nd = length(eqn.dissipator)

    println(io, "Liouvillian containing:")
    println(io, "  $n1 on-site term(s)")
    println(io, "  $n2 local term(s)")
    # println(io, "  $n3 non-local terms")
    return print(io, "  $nd lindblad operator(s)")
end

Base.:+(L::Liouvillian, H::Hamiltonian)= Liouvillian(L.hamiltonian + H, L.dissipator)
Base.:+(H::Hamiltonian, D::Dissipator)= Liouvillian(H, D)
Base.:+(L::Liouvillian, D::Dissipator)= Liouvillian(L.hamiltonian, dissipator(L) + D)

# Temporary solution
struct Ising{HL} <: AbstractSumOfTerms
    a::Float64
    CB::Matrix{ComplexF64}
    oneterms::Vector{LocalOp{1}}
    dissipator::Union{Nothing,Dissipator}
    Ising(a, CB, oneterms) = new{Hamiltonian}(a, CB, oneterms, nothing)
    Ising(a, CB, oneterms, dissipator) = new{Liouvillian}(a, CB, oneterms, dissipator)
end
oneterms(L::Ising) = L.oneterms

function scalelocal(L::Ising, a)

    @reset L.oneterms .*= a
    D = L.dissipator

    if !(D isa Nothing)
        @reset L.dissipator *= a
    end

    return L
end

isaxial(::Ising) = true

layertype(::Type{<:Ising}) = Ising

struct TrotterLayer{T}
    data::Matrix{T}
    axis::Int
    function TrotterLayer(data::Matrix{T}, axis) where {T}
        if any(!isaxial, data) && axis != 0 
            throw(ArgumentError("cannot put non-axial term on axis $axis"))
        else
            return new{layertype(T)}(data, axis)
        end
    end
end
function TrotterLayer(data, axis)
    T = layertype(typeof(data))
    mat = Matrix{T}(undef, 1, 1)
    mat[1,1] = data
    return TrotterLayer(mat, axis)
end

Base.convert(::TrotterLayer, term) = TrotterLayer(term)
function Base.convert(::Type{<:TrotterLayer{T}}, term) where {T}
    newdata = broadcast(t -> convert(T, t), term.data)
    return TrotterLayer(newdata, term.axis)
end

layertype(::Type{<:AbstractTerm}) = Hamiltonian
layertype(::Type{<:Liouvillian}) = Liouvillian
layertype(::Type{LocalOp{1}}) = LocalOp{1}
layertype(::Type{Dissipator}) = Dissipator


Base.:*(l::TrotterLayer, a) = l.data .= l.data .* a
Base.:*(a, l::TrotterLayer) = l.data .= a .* l.data

Base.:+(l::TrotterLayer, h::LocalOp{1}) = TrotterLayer(l.data .+ Ref(h), l.axis)
function Base.:+(l1::TrotterLayer, l2::TrotterLayer)

    if !(l1.axis == 0 && l2.axis == 0)
        if l1.axis != l2.axis
            throw(ArgumentError("cannot combine operators acting on different lattice axis into the same trotter layer"))
        end
    else
        rvdata = l1.data .+ l2.data
    end
    @assert l1.axis == l2.axis
    return TrotterLayer(rvdata, l1.axis)
end

Base.zero(t::TrotterLayer) = TrotterLayer(zero.(t.data), t.axis)

Base.:+(terms::AbstractSumOfTerms, l::TrotterLayer) = +(l, terms)
function Base.:+(l::TrotterLayer, terms::AbstractSumOfTerms)
    newdata = broadcast(i -> i + terms, l.data)
    return TrotterLayer(newdata, l.axis)
end
Base.:+(l::TrotterLayer, term::AbstractTerm) = +(l, TrotterLayer(term))

# linops(l::TrotterLayer{Hamiltonian}) = nothing
# terms(l::TrotterLayer) = TrotterLayer(sum.(terms.(l.data)), l.axis)
# oneterms(l::TrotterLayer) = TrotterLayer(sum.(oneterms.(l.data)), l.axis)

unwrap(model::TrotterLayer) = model.data

TrotterLayer(data) = TrotterLayer(data, 0)
TrotterLayer(data::TrotterLayer) = data

changeaxis(t::TrotterLayer, axis; a = 1.0) = TrotterLayer(scalelocal.(t.data, a), axis)

function broadcastmodel(
    G::Type{<:ITC.AbstractUnitCellGeometry}, model::TrotterLayer, physical, method; inreverse=false,kwargs...
)

    return broadcastmodel(
        G, model.data, physical, method; axis=model.axis, inreverse, kwargs...
    )
end

function applykwargs(S::Type, o::TrotterLayer)
    function rv(lat; kwargs...)
        return S(lat; axis=o.axis, kwargs...)
    end
    return rv
end

# physicaldim(term::AbstractTerm) = map(x -> size(x, 1), term.term)

function Base.hash(ham::Hamiltonian, h::UInt)
    hashes = sort(map(hash, ham))
    onehashes = map(hash, oneterms(ham))
    h = hash(hashes, h)
    h = hash(onehashes, h)
    return h
end
Base.:(==)(h1::Hamiltonian, h2::Hamiltonian) = hash(h1) == hash(h2)

convert_if_identity(I::UniformScaling) = Matrix{ComplexF64}(I, (2, 2))
convert_if_identity(x) = x

function Base.:+(t1::AbstractSingleTerm, t2::AbstractSingleTerm)
    newterms = TupleTools.vcat(terms(t1), terms(t2))
    return Hamiltonian(newterms, vcat(oneterms(t1), oneterms(t2)))
end

# Commutes
Base.:*(A::AbstractTerm, a::Number) = Base.:*(a, A)
Base.:*(a::Number, A::AbstractTerm) = @set A.a *= a

function Base.:*(a::Number, H::Hamiltonian) 
    H1 = map(t -> a * t, terms(H))
    H0 = map(t -> a * t, oneterms(H))
    return Hamiltonian(H1, H0)
end

Base.:-(A::AbstractTerm) = -1 * A
Base.:-(A::AbstractTerm, B::AbstractTerm) = +(A, -B)

vectorise(term::AbstractTerm) = vectorise(typeof(term))(Matrix{ComplexF64}(term))

(T::Type{<:AbstractMatrix})(A::AbstractTerm) = T(term_to_matrix(A))

struct Combine
    model::NTuple{2,TrotterLayer}
end
Combine(a,b) = Combine((a,b))

function Base.:+(c::Combine, h::AbstractTerm) 
    c1,c2 = c.model
    return Combine((c1 + h), (c2 + h))
end

function addlinops(c::Combine, linops)
    c1,c2 = c.model
    rv1 = addlinops(c1, linops)
    rv2 = addlinops(c2, linops)
    return Combine(rv1,rv2)
end

function (model::Combine)(G, physical, method, dt; inreverse=false, kwargs...)
    @info "Combining..."

    @debug "" model.model

    # Swap the order of the axis
    termtuple = inreverse ? reverse(model.model) : model.model

    t1,t2 = termtuple
    @debug "" typeof(t1) typeof(t2)

    tlat1 = broadcastmodel(G, t1, physical, method;  kwargs...)(dt)
    tlat2 = broadcastmodel(G, t2, physical, method;  kwargs...)(dt)

    rv = broadcast(tlat1, tlat2) do ten1, ten2
        @tensoropt out[ok ob ik ib; e s w n] :=
            ten2[ok ob x y; e w] * ten1[x y ik ib; s n]
        return out
    end

    return rv
end

function broadcastmodel(
    G::Type{<:ITC.AbstractUnitCellGeometry}, model::Combine, physical, method; kwargs...
)
    return dt -> onlattice(G, model(G, physical, method, dt; kwargs...))
end
function applykwargs(S::Type, ::Combine)
    function rv(lat; kwargs...)
        return S(lat; kwargs...)
    end
    return rv
end

Base.convert(::Type{Hamiltonian}, term::AbstractTerm) = Hamiltonian(term)

struct Model{HL<:Union{Hamiltonian,Liouvillian,Ising}} <: AbstractModel
    layers::Vector{TrotterLayer{HL}}
    function Model{T}(terms::Vector) where {T}
        layers = map(TrotterLayer,terms)
        return new{T}(layers)
    end
    Model{T}() where {T} = new{T}([])
end

Model(HL::Vararg) = Model(collect(HL))
Model(H::Vector{<:Union{Hamiltonian,Liouvillian}}) = Model(map(h -> h + zero(Hamiltonian), H))

Model(H::T) where {T} = Model{layertype(T)}([H])
Model(H::Vector{T}) where {T} = Model{layertype(T)}(H)

Base.eltype(::Model{HL}) where {HL} = HL

# Model{T}(HL::Vararg) where {T} = Model{T}(collect(HL))

Base.iterate(model::Model, state = 1) = iterate(model.layers, state)
Base.length(model::Model) = length(model.layers)

function Base.show(io::IO, ::MIME"text/plain", model::Model)
    print("$(typeof(model)) with $(length(model)) Trotter layer")
    length(model) > 1 && print("s")
    return nothing
end

Base.sum(t::TrotterLayer) = t


function Base.push!(model::Model{T}, layer) where {T}

    # H0 = zero(eltype(model))
    #
    # sum(H0, )

    push!(model.layers, TrotterLayer(sum(layer)))
    return model
end

function Base.push!(model::Model{T}, layer::TrotterLayer) where {T}
    try
        push!(model.layers, layer)
    catch e
        if e isa MethodError
            if e.f === convert
                type, value = e.args
                if all(S -> S<:TrotterLayer{Hamiltonian}, (type, TrotterLayer{T}))
                    if value isa Union{TrotterLayer{Liouvillian},TrotterLayer{Dissipator}}
                        @error "Failed to `push!` term of type `$(typeof(value))` to `$(typeof(model))`. You should instead construct an instance of the type `Model{Liouvillian}` to add terms of type `Dissipator` and `Liouvillian`."
                    end
                end
            end
        end
        rethrow()
    end
    return model
end
function Base.push!(model::Model, layer::TrotterLayer{<:Union{Dissipator, LocalOp{1}}})

    l = length(model)

    broadcast!(+, model.layers, model.layers, 1 / l * layer)
    return model
end

function Base.append!(m1::Model, m2::Model)
    for layer in m2
        push!(m1, m2)
    end
    return m1
end
function Base.push!(model::Model, term::AbstractTerm) 
    if isexpandable(term)
        for subterm in expandterm(term)
            push!(model, expandterm(subterm))
        end
    else
        push!(model, TrotterLayer(term))
    end
    return model
end
