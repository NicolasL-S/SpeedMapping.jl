"""
$(read(joinpath((@__DIR__)[1:end-4], "README.md"), String))
"""
module SpeedMapping

	# --- using ---
	using MaybeInplace
	using LinearAlgebra
	using StaticArrays: FieldVector
	using ForwardDiff
	#using AccurateArithmetic: dot_oro
	
	# --- include ---
	include("SpeedMapping_structs.jl")
	include("SpeedMapping_common.jl")
	include("acx.jl")
	include("aa.jl")

	# --- exports ---
	export speedmapping
	export AcxState
	export AcxCache
	export AaCache
	export AaState
	export SpeedMappingResult
end