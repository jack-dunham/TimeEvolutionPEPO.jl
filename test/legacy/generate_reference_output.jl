# Reference output for the PEPSKit refactor, task 0.1 of PEPSKIT_REFACTOR_PLAN.md.
#
#   julia> include("test/generate_reference_output.jl")
#   julia> generate()                                    # writes test/reference/
#   julia> generate("test/reference-phase2")              # writes elsewhere
#   julia> compare("test/reference", "test/reference-phase2")
#
# R3 uses `TEBD(; alg = FET(...))` rather than `AxialTEBD`: both reach the environment-based
# truncation, but the `AxialStepper` method at timeevolution.jl:228 prints and calls `similar` on an
# `AxialPEPO`, which has no such method.

using TimeEvolutionPEPO
using TensorKit
using Random
using Logging
using Printf
using DelimitedFiles

const REFERENCE_DIR = joinpath(@__DIR__, "reference")
const SEED = 1234
const COLUMNS = ["step", "time", "expval_x", "expval_z", "correlationlength", "bondentropy"]

quiet(f) = with_logger(f, ConsoleLogger(stderr, Logging.Warn))

_scalar(x::Number) = real(x)
_scalar(x) = maximum(real, x)

function _maxbondentropy(pepo)
    entropies = bondentropy(pepo)
    return maximum(entropies[bond] for bond in TimeEvolutionPEPO.eachbond(entropies))
end

"""
One row of observables: transverse and longitudinal magnetisation from the one-site reduced density
matrix, the boundary correlation length, and the largest bond entropy.
"""
function measure(pepo, densitymatrix, obsalg)
    X, _, Z = PAULI
    dm = quiet(() -> densitymatrix(pepo, obsalg))
    rho = partialtrace(dm, (1, 1))
    return (
        _scalar(expval(rho, X)),
        _scalar(expval(rho, Z)),
        _scalar(correlationlength(dm)),
        _maxbondentropy(pepo),
    )
end

struct Configuration
    name::String
    description::String
    build::Function      # () -> Simulation
    densitymatrix::Type
    obsalg::Any
    numsteps::Int
end

function configurations()
    X, Y, Z = PAULI
    lindblad = () -> Model(1.5 / 4 * LocalOp(Z, Z) + Dissipator(0.5 * (Y - im * Z)))
    hamiltonian = () -> Model(1.5 / 4 * LocalOp(Z, Z) - LocalOp(X))
    downspins = () -> PEPO(fill((0, 0, -1), 2, 2), 2, 4)
    ctmrg = CTMRG(; bonddim=8, tol=1e-8, maxiter=100, verbose=false)
    vumps = VUMPS(; bonddim=8, tol=1e-8, maxiter=100, verbose=false)

    return [
        Configuration(
            "R1",
            "Lindblad TEBD + SimpleUpdate, 2x2 cell, D=4, dt=0.025",
            () -> Simulation(downspins(), lindblad(), TEBD(); timestep=0.025),
            DensityMatrix,
            ctmrg,
            40,
        ),
        Configuration(
            "R2",
            "thermal annealing (Hamiltonian only), 2x2 cell, D=4, dt=0.025",
            () -> Simulation(
                PEPO(fill(ThermalState(), 2, 2), 2, 4), hamiltonian(), TEBD();
                timestep=0.025,
            ),
            PurifiedDensityMatrix,
            vumps,
            40,
        ),
        Configuration(
            "R3",
            "Lindblad TEBD + FET, 2x2 cell, D=4, dt=0.025",
            () -> Simulation(
                downspins(), lindblad(), TEBD(; alg=FET(; alg=ctmrg)); timestep=0.025
            ),
            DensityMatrix,
            ctmrg,
            10,
        ),
    ]
end

function record(config::Configuration; record_every::Int=1)
    Random.seed!(SEED)
    sim = quiet(config.build)

    rows = Vector{Vector{Float64}}()
    callback = function (sim)
        info = simulationinfo(sim)
        info.iterations % record_every == 0 || return nothing
        x, z, xi, s = measure(quantumstate(sim), config.densitymatrix, config.obsalg)
        push!(rows, [Float64(info.iterations), info.simtime, x, z, xi, s])
        @info @sprintf("%s step %3d  x=% .6f  z=% .6f  xi=%.6f  S=%.6f",
            config.name, info.iterations, x, z, xi, s)
        return nothing
    end

    quiet(
        () -> simulate!(
            callback, sim; numsteps=config.numsteps, maxshots=config.numsteps, verbose=false
        ),
    )

    return reduce(hcat, rows)'
end

function generate(dir::AbstractString=REFERENCE_DIR; record_every::Int=1, only=nothing)
    mkpath(dir)
    failures = String[]

    for config in configurations()
        isnothing(only) || config.name in only || continue
        path = joinpath(dir, "$(config.name).dat")
        try
            data = record(config; record_every)
            open(path, "w") do io
                println(io, "# ", config.name, ": ", config.description)
                println(io, "# seed=", SEED, " columns=", join(COLUMNS, ","))
                writedlm(io, data)
            end
            @info "wrote $path ($(size(data, 1)) rows)"
        catch e
            push!(failures, "$(config.name): $(sprint(showerror, e))")
            @error "configuration $(config.name) failed" exception = e
        end
    end

    isempty(failures) || @warn "failed configurations" failures
    return dir
end

"""
Largest elementwise deviation per configuration between two output directories. Files whose shapes
disagree are reported as `missing`, since a changed row count is not a numerical deviation.
"""
function compare(dir1::AbstractString, dir2::AbstractString)
    deviations = Dict{String,Any}()

    for config in configurations()
        f1 = joinpath(dir1, "$(config.name).dat")
        f2 = joinpath(dir2, "$(config.name).dat")
        (isfile(f1) && isfile(f2)) || continue

        d1 = readdlm(f1; comments=true)
        d2 = readdlm(f2; comments=true)

        deviations[config.name] =
            size(d1) == size(d2) ? maximum(abs.(d1 .- d2)) : missing
    end

    for (name, dev) in sort(collect(deviations); by=first)
        @info "$name: maximum deviation $dev"
    end

    return deviations
end
