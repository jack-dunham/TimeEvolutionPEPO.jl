# function tebdtensor(
#     model::LindbladEquation, p1::IndexSpace, p2::IndexSpace, dt; compresstol=eps()
# )
#
#     # ket1 * ket2 * bra1 * bra2
#     spacecod = p1 * p2 * p1' * p2'
#     spacedom = p1 * p2 * p1' * p2'
#
#     data = zeros(ComplexF64, dim(spacecod), dim(spacedom))
#     symdata = Array{ComplexF64}(undef, dims(spacecod * spacedom))
#
#     if isempty(model.linops)
#         vecf = thermalvectorise
#         dt = dt / 2
#     else
#         vecf = vectorise
#         data += 1 / 2 * sum(vecf, model.linops; init=zero(data))
#     end
#
#     # We divide by half again for local terms as in 2D we effectively apply these twice,
#     data += 1 / 2 * sum(vecf, model.onelocal; init=zero(data))
#
#     data += sum(vecf, model.twolocal; init=zero(data))
#
#     copyto!(symdata, exp(dt * data))
#
#     tensor = TensorMap(symdata, spacecod, spacedom)
#
#     # move site one to codomain and site two to domain
#     tensor = permute(tensor, (1, 3, 5, 7), (2, 4, 6, 8))
#
#     u, s, v, _ = tsvd(tensor; trunc=truncbelow(compresstol))
#
#     return u * sqrt(s), _transpose(sqrt(s) * v)
# end

function construct(model, space, method)
    for t in terms(model)
        if !(t isa LocalOp{2})
            throw(ArgumentError("term of type $(typeof(t)) incompatible with TEBD"))
        end
    end
    return dt -> exact_tebd_time_stepper(model, space ,dt)
end

function construct(model, physical, method::AbstractTEBD; axis, kwargs...)

    for t in terms(model)
        if !(t isa LocalOp{2})
            throw(ArgumentError("term of type $(typeof(t)) incompatible with TEBD"))
        end
    end

    model = unwrap(model)

    rv =
        timestep -> begin
            if axis == 1
                incr = (1, 0)
            elseif axis == 2
                incr = (0, 1)
            end

            p1p2 = broadcast(tuple, physical, circshift(physical, .-incr))

            uvs = broadcast(p1p2, model) do (p1, p2), modeli
                return tebdtensor(modeli, p1, p2, timestep; compresstol=method.compresstol)
            end

            lat = broadcast(tuple, first.(uvs), circshift(last.(uvs), incr))

            if method isa AxialTEBD
                lat = broadcast(lat) do (u, v)
                    @tensoropt out[ok ob ik ib; e w] := u[ok ob x y; e] * v[x y ik ib; w]
                    return out
                end
            end

            return lat
        end

    return rv
end
function tebdtensor(
    model::AbstractTerm, p1::IndexSpace, p2::IndexSpace, dt; compresstol=eps()
)
    # ket1 * ket2 * bra1 * bra2
    spacecod = p1 * p2 * p1' * p2'
    spacedom = p1 * p2 * p1' * p2'

    data = zeros(ComplexF64, dim(spacecod), dim(spacedom))
    symdata = Array{ComplexF64}(undef, dims(spacecod * spacedom))

    if model isa Hamiltonian
        vecf = thermalvectorise
    elseif model isa Liouvillian
        vecf = vectorise
        # data += 1 / 2 * sum(vecf, dissipator(model); init=zero(data))
        D = dissipator(model)
        if length(D) > 0
            data += vectorise(D, 2)
        end
    end

    # We divide by half again for local terms as in 2D we effectively apply these twice,
    if length(oneterms(model)) > 0 
        data += sum(vecf, oneterms(model); init=zero(data))
    end

    if length(terms(model)) > 0 
        data += sum(vecf, terms(model))
    end

    copyto!(symdata, exp(dt * data))

    tensor = TensorMap(symdata, spacecod, spacedom)

    # move site one to codomain and site two to domain
    tensor = permute(tensor, (1, 3, 5, 7), (2, 4, 6, 8))

    trunc = length(model) > 0  ? truncbelow(compresstol) : truncdim(1)
    u, s, v, err = tsvd(tensor; trunc=trunc)

    @info "" err η = dim(domain(s))

    return u * sqrt(s), _transpose(sqrt(s) * v)
end

function broadcastmodel(model::OnAxis)
    axis = model.axis
    return broadcast
end

# function (model::LindbladEquation)(physical, method::AbstractClusterMPO; kwargs...)
#     rv =
#         timestep -> begin
#             mpo = MPO(model)
#
#             data = clustermpo(
#                 mpo,
#                 timestep,
#                 method.order;
#                 method.extend,
#                 method.compress,
#                 method.approx,
#             )
#
#             tensor = clustermpotensor(data, physical)
#
#             return tensor
#         end
#
#     return rv
# end

# function broadcastmodel(
#     ::Type{<:LindbladEquation}, model, physical, method::AbstractTEBD; axis, kwargs...
# )
#     model = unwrap(model)
#
#     rv =
#         timestep -> begin
#             if axis == 1
#                 incr = (1, 0)
#             elseif axis == 2
#                 incr = (0, 1)
#             end
#
#             p1p2 = broadcast(tuple, physical, circshift(physical, .-incr))
#
#             uvs = broadcast(p1p2, model) do (p1, p2), modeli
#                 return tebdtensor(modeli, p1, p2, timestep; compresstol=method.compresstol)
#             end
#
#             lat = broadcast(tuple, first.(uvs), circshift(last.(uvs), incr))
#
#             if method isa AxialTEBD
#                 lat = broadcast(lat) do (u, v)
#                     @tensoropt out[ok ob ik ib; e w] := u[ok ob x y; e] * v[x y ik ib; w]
#                     return out
#                 end
#             end
#
#             return lat
#         end
#
#     return rv
# end
# LindbladEquation(p::Int) = LindbladEquation([], [], [], [], p::Int)

function broadcastmodel(
    ::Type{<:Union{Hamiltonian, Liouvillian, Ising}}, model, physical, method::AbstractTEBD; axis, inreverse, kwargs...
)
    model = unwrap(model)

    rv =
        timestep -> begin
            if axis == 1
                incr = (1, 0)
            elseif axis == 2
                incr = (0, 1)
            end

            p1p2 = broadcast(tuple, physical, circshift(physical, .-incr))

            uvs = broadcast(p1p2, model) do (p1, p2), modeli
                return tebdtensor(modeli, p1, p2, timestep; compresstol=method.compresstol)
            end

            lat = broadcast(tuple, first.(uvs), circshift(last.(uvs), incr))

            if method isa Union{AxialTEBD,GlobalTEBD}
                lat = broadcast(lat) do (u, v)
                    if inreverse
                        @tensoropt out[ok ob ik ib; e w] := v[ok ob x y; w] * u[x y ik ib; e]
                    else
                        @tensoropt out[ok ob ik ib; e w] := u[ok ob x y; e] * v[x y ik ib; w]
                    end
                    return out
                end
            end

            return lat
        end

    return rv
end

function broadcastmodel(::Type{<:Union{Hamiltonian,Liouvillian}}, model, physical, method::ClusterPEPO; kwargs...)
    # kw = paramkwargs(model)

    #TODO, let TrotterLayer be hamiltonain or liouvvilian

    rv =
        timestep -> begin

            tensors = broadcast(model, physical) do liouvillian, space
                rules = FSM(space)

                fsmadd!(rules, liouvillian)

                return fsmpepo(rules, space, timestep; method.extend)
            end

            return tensors
        end

    return rv
end

function (model::Union{Hamiltonian, Liouvillian})(physical, method::ClusterMPO; kwargs...)
    rv =
        timestep -> begin
            mpo = MPO(model)

            data = clustermpo(
                mpo,
                timestep,
                method.order;
                method.extend,
                method.compress,
                method.approx,
            )

            tensor = clustermpotensor(data, physical)

            return tensor
        end
    return rv
end

# function Base.show(io::IO, ::MIME"text/plain", eqn::LindbladEquation)
#     d = physicaldim(eqn)
#
#     n1 = length(eqn.onelocal)
#     n2 = length(eqn.twolocal)
#
#     nl = length(eqn.longrange)
#
#     nd = length(eqn.linops)
#
#     println(io, "LindbladEquation with d = $d containing:")
#     println(io, "  $n1 local terms")
#     println(io, "  $n2 nearest-neighbour terms")
#     println(io, "  $nl long-range terms")
#     return print(io, "  $nd lindblad operators")
# end
