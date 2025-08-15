push!(LOAD_PATH,"../src/")
using SpeedMapping
using Documenter
using Literate

cd(@__DIR__)
Literate.markdown("./src/tutorial.jl")
makedocs(
         sitename = "SpeedMapping.jl",
         pages=[
                "Introduction" => "index.md",
                "Tutorial" => "tutorial.md",
                "API" => "api.md"
               ])
#deploydocs(;
#    repo="github.com/NicolasL-S/SpeedMapping.jl.git",
#)