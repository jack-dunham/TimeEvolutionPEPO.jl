using Documenter
using TimeEvolutionPEPO

makedocs(
    sitename="TimeEvolutionPEPO.jl",
    pages=[
        "Home" => "index.md",
        "Manual" => [
            "man/tutorial.md",
            "man/tensormap.md",
            "man/states.md",
            "man/models.md",
        ],
        "Library" => [
            "lib/states.md",
            "lib/simulations.md",
            "lib/models.md",
            "lib/observables.md",
            "lib/methods.md",
        ],
        "Interfaces" => "interfaces.md",
        "Index" => "lib/index.md",
    ],
)

deploydocs(
    repo="github.com/jack-dunham/TimeEvolutionPEPO.jl.git",
    devbranch="dev",
)
