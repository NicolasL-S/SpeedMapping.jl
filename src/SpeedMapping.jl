module SpeedMapping

	# --- using ---
	using MaybeInplace
	using LinearAlgebra
	using StaticArrays: FieldVector
	using ForwardDiff
	using AccurateArithmetic: dot_oro
	
	# --- include ---
	include("SpeedMapping_structs.jl")
	include("acx.jl")
	include("aa.jl")
	include("SpeedMapping_common.jl")

	# --- exports ---
	export speedmapping
end