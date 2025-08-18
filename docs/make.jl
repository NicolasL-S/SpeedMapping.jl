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
         pages=[
                "Introduction" => "index.md",
                "Tutorial" => "tutorial.md",
                "API" => "api.md",
                "Benchmarks" => "benchmarks.md"
               ])
deploydocs(;
    repo="github.com/NicolasL-S/SpeedMapping.jl",
    devbranch = "main",
)