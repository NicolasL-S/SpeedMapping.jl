push!(LOAD_PATH,"../src/")
using SpeedMapping
using Documenter
using Literate

cd(@__DIR__)
Literate.markdown("./src/tutorial.jl")
makedocs(
         sitename = "SpeedMapping.jl",
         modules  = [SpeedMapping],
         pages=[
                "Introduction" => "index.md",
                "Tutorial" => "tutorial.md",
                "API" => "api.md"
               ])
makedocs(;pages, modules = [SpeedMapping])
#deploydocs(;
#    repo="github.com/NicolasL-S/SpeedMapping.jl.git",
)