cd(@__DIR__)
using SpeedMapping
using Documenter
using Literate

Literate.markdown(
    joinpath(@__DIR__, "src", "tutorial.jl"), joinpath(@__DIR__, "src");
    credit = false
)
makedocs(
         sitename = "SpeedMapping.jl",
         format=Documenter.HTML(;
            canonical="https://NicolasL-S.github.io/SpeedMapping.jl",
            edit_link="main",
            assets=String[],
        ),
         pages=[
                "Introduction" => "index.md",
                "Tutorial" => "tutorial.md",
                "API" => "api.md",
                "Benchmarks" => "benchmarks.md"
               ])
deploydocs(;
    repo="github.com/NicolasL-S/SpeedMapping.jl",
    devbranch = "MajorRefactor",
)