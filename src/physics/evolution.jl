function propagate(
    vp, #::TensorMap{S,4,2},
    M::TensorMap{S,3,3},
    n::Int,
    χ::Int,
    ξ=Inf;
    truncalg::AbstractTruncationAlgorithm=FET(),
) where {S<:IndexSpace}
    vp = _propagate(vp, M, n, χ, ξ; truncalg=truncalg)
    return vp
end

function _propagate(
    vp, #::TensorMap{S,4,2}, 
    M::TensorMap{S,3,3},
    n::Int,
    χ::Int,
    ξ=Inf;
    truncalg::FET=FET(),
) where {S<:IndexSpace}
    A = randmps(vp, χ)
    Ax = A
    Ay = A
    for i in 1:n
        vp, Ax, Ay = trotterstep(vp, M, Ax, Ay, ξ; truncalg=truncalg)
    end
    return vp
end

function trotterstep(
    vp, #::TensorMap{S,4,2},
    m::TensorMap{S,3,3},
    Ax::TensorMap{S,2,2},
    Ay::TensorMap{S,2,2},
    ξ=Inf;
    truncalg::FET=FET(),
) where {S<:IndexSpace}
    gauge = truncalg.gauge
    alg = truncalg.alg
    vp, Ax = fetstep(vp, m, Ax, :x, ξ; alg=alg, gauge=gauge)
    vp, Ay = fetstep(vp, m, Ay, :y, ξ; alg=alg, gauge=gauge)
    return vp, Ax, Ay
end

function randmps(vp::TensorMap{S,4,2}, χ::Int) where {S<:IndexSpace}
    unitsp = oneunit(S)
    bondsp = ⊕(fill(unitsp, χ)...)

    sp = codomain(vp)[4]
    A = TensorMap(randn, ComplexF64, sp * bondsp, sp * bondsp)
    return A
end
function randmps(p::TensorMap{S,3,3}, χ::Int) where {S<:IndexSpace}
    unitsp = oneunit(S)
    bondsp = ⊕(fill(unitsp, χ)...)

    sp = codomain(p)[2]
    A = TensorMap(randn, ComplexF64, sp * bondsp, sp * bondsp)
    return A
end
