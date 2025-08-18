cd(@__DIR__)
using SpeedMapping
using Documenter
using Literate

DocMeta.setdocmeta!(SpeedMapping, :DocTestSetup, :(using SpeedMapping); recursive=true)

Literate.markdown(
    joinpath(@__DIR__, "src", "tutorial.jl"), joinpath(@__DIR__, "src");
    credit = false
)
makedocs(
    sitename = "SpeedMapping.jl",
    format = Documenter.HTML(
        canonical = "https://juliadata.github.io/SpeedMapping.jl/stable/",
        edit_link = "main",
    ),
    pages=[
        "Introduction" => "index.md",
        "Tutorial" => "tutorial.md",
        "API" => "api.md",
        "Benchmarks" => "benchmarks.md"
    ]
)
deploydocs(;
    repo="github.com/NicolasL-S/SpeedMapping.jl.git",
    devbranch = "main",
)