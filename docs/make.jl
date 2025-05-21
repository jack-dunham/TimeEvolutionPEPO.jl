using Documenter
using TimeEvolutionPEPO

makedocs(
    sitename="TimeEvolutionPEPO.jl",
    pages=[
        "Home" => "index.md",
        "Manual" => [
            "man/tutorial.md",
            "man/states.md",
            "man/models.md",
        ],
        "Library" => [
            "lib/models.md",
            "lib/observables.md",
            "lib/simulations.md",
        ],
        "Index" => "lib/index.md",
    ],
)

deploydocs(
    repo="github.com/jack-dunham/TimeEvolutionPEPO.jl.git",
    devbranch="dev",
)
