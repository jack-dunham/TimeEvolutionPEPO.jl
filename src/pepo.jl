function LinearAlgebra.normalize!(pepo::AbstractPEPO, boundary::AbstractBoundary)
    return rmul!.(st, 1 / norm(pepo, boundary))
end
function LinearAlgebra.norm(pepo::AbstractPEPO, boundary::AbstractBoundary)
    st = getstate(pepo)
    norm = tracecontract(boundary, contractphysical.(st))
    return norm
end

function onelocal(pepo::AbstractPEPO, boundary::AbstractBoundaryState; kwargs...)
    return onelocal(pepo, boundary.tensors; kwargs...)
end

function onelocal(pepo::AbstractPEPO, boundary::AbstractBoundary; normalize=false)
    onelocal = onelocalcontract(boundary, getstate(pepo))

    if normalize
        N = norm(pepo, boundary)
        @debug "pepo normalization: $N"
        rmul!.(onelocal, 1 ./ N)
    end

    return onelocal
end

function measure(pepo::AbstractPEPO, boundary, observables...; kwargs...)
    pepo_i = onelocal(pepo; kwargs...)
    return measure(pepo_i, observables...)
end
function measure(pepo_i, observables::Vararg{<:AbstractMatrix})
    return map(x -> measure(pepo_i, x), observables)
end

function measure(pepo_i, observable::AbstractMatrix)
    d = domain.(pepo_i)
    c = codomain.(pepo_i)
    obs_t = TensorMap.(Ref(observable), d, c)
    out_val = tr.(obs_t .* pepo_i)
    @debug "Measurment outcome: $out_val"
    return out_val
end
