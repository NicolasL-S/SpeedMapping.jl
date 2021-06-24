push!(LOAD_PATH,"../src/")
using SpeedMapping
using Documenter
makedocs(
         sitename = "SpeedMapping.jl",
         modules  = [SpeedMapping],
         pages=[
                "Home" => "index.md"
               ])
deploydocs(;
    repo="github.com/NicolasL-S/SpeedMapping.jl.git",
)