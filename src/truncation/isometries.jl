function (alg::IsometryOptimisation)(pepo::AbstractPEPO, environment::AbstractEnvironment)
    bonds = getbonds(pepo)
    return alg.(pepo, environment, bonds)
end
function (alg::IsometryOptimisation)(pepo, environment, bond)
    D = domain.(getstate(pepo), 1)[left(bond)]

    tmetric = metric(pepo, environment, bond)
    left0, right0 = isometryinit(tmetric)
    u, s_f64, v = metricoptimise(left0, right0, tmetric, truncspace(D))

    s = similar(s_f64, eltype(u))

    copy!(s, s_f64)

    return TruncationIsometries(u, s, v), tmetric
end

# Only works in sequence
function (alg::TensorOptimisation)(pepo::AbstractPEPO, environment, bond)
    D = domain.(getstate(pepo), 1)[left(bond)]

    p_left, p_right = pepo[bond]

    left0 = _leftrankreduce!(p_left)
    right0 = _rightrankreduce!(p_right)

    tmetric = metric(pepo, environment, bond)

    u, s_f64, v = metricoptimise(left0, right0, tmetric, truncspace(D))

    s = similar(s_f64, numbertype(pepo))

    copy!(s, s_f64)

    return TruncationIsometries(u, s, v), tmetric
end

function _leftrankreduce!(p::AbsTen{2,4})
    Q, R = leftorth(p, (1, 2, 4, 5, 6), (3,))
    permute!(p, Q, (1, 2), (6, 3, 4, 5))
    return R
end
function _rightrankreduce!(p::AbsTen{2,4})
    L, Q = rightorth(p, (5,), (1, 2, 3, 4, 6))
    permute!(p, Q, (2, 3), (4, 5, 1, 6))
    return L
end
