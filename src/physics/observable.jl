abstract type AbstractObservable{T<:AbstractTensorMap} end

struct TrMetric <: PureTensorStyle end

struct LocalObservable{T<:AbsTen{1,1}} <: AbstractObservable{T}
    data::T
    function LocalObservable(t::T) where {T<:AbsTen{1,1}}
        if !ishermitian(t)
            throw(ArgumentError("Operator is not hermitian"))
        end
        return new{T}(t)
    end
end

struct Trace{A<:AbstractBoundary, Nx, Ny, T<:AbsTen{2,2}}
    data::UnitCell{TrMetric, Nx, Ny, T}
    boundary::BoundaryState{A}
end

function (Tr::Trace)(P::InfinitePEPO)
    Tr_t = Tr.data
    P_l = applytrace.(Tr_t, P)
    return P_l
end


function onelocal(P)
    tra = trace(P)
    P_l = tra(P)

    return P_l ./ tr.(P_l)
end

function LinearAlgebra.normalize(P, Tr::Trace)
    N = Tr(P)
end


function Base.:*(t::LocalObservable, p::AbsTen{3,3})
    tt = t.data
    @tensoropt p_n[1 2 k; 4 5 b] := tt[k; kk] * p[1 2 kk; 4 5 b]
    return p_n
end

function tracephysical(p::AbsTen{3,3})
    @tensoropt p_tr[1 2; 4 5] := p[1 2 x; 4 5 x]
end

function applytrace(Tr::AbsTen{2,2}, p::AbsTen{3,3})
    @tensoropt p_l[k ; b] := Tr[3 4; 1 2] * p[1 2 k; 3 4 b] 
    return p_l
end

function trace(
    P::InfinitePEPO,
    alg=VUMPS::Type{<:AbstractBoundary},
    param=BoundaryParameters(P)::BoundaryParameters,
)
    P_tr = convertstyle(MPO, tracephysical.(P))
    P_tr = P_tr ./ tr.(P_tr)
    boundary = initialise_state(alg, P_tr, param)

    calculate!(boundary)

    tensors = boundary.data
    
    Tr = get_trace(tensors)

    return Trace(Tr , boundary)
end

get_trace(v::VUMPS) = convertstyle(TrMetric, get_vumpstrace(v.A, v.FL, v.FR))

function get_vumpstrace(A, FL, FR)
    AC = centraltensor(A)
    return get_single_vumpstrace.(AC, circshift(AC,(0,-1)), FL, FR)
end

function get_single_vumpstrace(acu, acd, fl , fr)
    @tensoropt Tr[7 4 ; 2 5] := fl[3;1 2]*acu[1 4; 6]*(acd')[8;3 5]*fr[6 7; 8]
    return Tr
end
