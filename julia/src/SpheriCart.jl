module SpheriCart

using StaticArrays

import LuxCore
using LuxCore: AbstractLuxLayer
using Random: AbstractRNG

export SolidHarmonics,
       compute,
       compute!,
       compute_with_gradients,
       compute_with_gradients!

include("indexing.jl")
include("normalisations.jl")
include("api.jl")
include("generated_kernels.jl")
include("ka_kernels.jl")
include("spherical.jl")
include("complex.jl")
include("luxcore.jl")

end
