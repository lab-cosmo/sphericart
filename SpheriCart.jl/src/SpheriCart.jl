module SpheriCart

using StaticArrays, OffsetArrays, ObjectPools

export SolidHarmonics, 
       compute, 
       compute!

include("indexing.jl")
include("normalisations.jl")
include("api.jl")
include("generated_kernels.jl")
include("batched_kernels.jl")

end

