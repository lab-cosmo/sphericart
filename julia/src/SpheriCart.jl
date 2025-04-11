module SpheriCart

using StaticArrays, OffsetArrays, Bumper

export SolidHarmonics, 
       compute, 
       compute!, 
       compute_with_gradients,
       compute_with_gradients!

include("indexing.jl")
include("normalisations.jl")
include("api.jl")
include("generated_kernels.jl")
include("batched_kernels.jl")
include("spherical.jl")

include("ka_kernels.jl")

end

