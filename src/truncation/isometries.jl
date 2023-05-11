function (alg::IsometryOptimisation)(args...)
    G = metric(args)
    left0, right0 = isometryinit(G)
    U, S, V = metricoptimise(left0, right0, G; EAT=false)
    return U, S, V
end

# Only works in sequence
function (alg::TensorOptimisation)(pepo, environment, bond)
    pepo[l], left0 = leftrankreduce(pepo[l], bond)
    right0, pepo[r] = rightrankreduce(pepo[r], bond)
    G = metric(pepo, boundary, bond)
    u, s, v = metricoptimise(left0, right0, G; alg.kwargs...)
    return TruncationIsometries(u, s, v)
end
