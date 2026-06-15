
"""
`struct SolidHarmonics` : datatype representing a solid harmonics basis. 

### Constructor to generate the basis object
```julia
basis = SolidHarmonic(L::Integer; kwargs...)
```

### Keyword arguments:
* `normalisation = :L2` : choose the normalisation of the basis, default is to
   make it orthonormal on the unit sphere.
* `static = (L <= 6)` : if `true`, single-point evaluation uses fully unrolled,
generated code returning an `SVector` (fastest for a single input, but with a
compile-time and stack footprint that grows with `L`). If `false`, single-point
evaluation reuses the batched KernelAbstractions kernel. Batched evaluation
always uses the KernelAbstractions kernel regardless of this flag. The default
keeps the unrolled path for the common small-`L` regime; raise it if you call
single-point evaluation at larger `L` in a hot loop.
* `T = Float64` : datatype in which basis parameters are stored. The output type
is inferred at runtime, but the rule of thumb is to use `T = FloatX` for
`FloatX` output.

### Usage example: 
```julia
using StaticArrays, SpheriCart
basis = SolidHarmonics(4)
# evaluate basis with single input 
𝐫 = @SVector randn(3)
Z = basis(𝐫)
Z = compute(basis, 𝐫)
# evaluate basis with multiple inputs (batching)
R = [ @SVector randn(3) for _ = 1:32 ]
Z = basis(Rs)
Z = compute(basis, Rs)
# evaluate basis with gradients
Z, ∇Z = compute_with_gradients(basis, 𝐫) # or Rs

# to be implented: (simply use ForwardDiff)
# Z, ∇Z, ∇²Z = compute_with_hessian(basis, 𝐫)
```
See documentation for more details.
"""
struct SolidHarmonics{L, NORM, STATIC, TF}
   Flm::TF
end

function SolidHarmonics(L::Integer;
                        normalisation = :L2,
                        static = (L <= 6),
                        T = Float64)
   Flm = generate_Flms(L; normalisation = normalisation, T = T)
   @assert eltype(Flm) == T
   # always store Flm as an (isbits) SMatrix: it is small for the supported
   # range of L, and being isbits it can be passed by value into the GPU kernel
   # irrespective of `static`.
   Flm = SMatrix{size(Flm, 1), size(Flm, 2)}(Flm)
   SolidHarmonics{L, normalisation, static, typeof(Flm)}(Flm)
end

@inline (basis::SolidHarmonics)(args...) = compute(basis, args...)


@inline function compute(basis::SolidHarmonics{L, NORM, true}, 𝐫::SVector{3}
                 ) where {L, NORM}
   return static_solid_harmonics(Val{L}(), 𝐫, basis.Flm)
end

function compute(basis::SolidHarmonics{L, NORM, false}, 𝐫::SVector{3, T}
         ) where {L, NORM, T}
   Z = zeros(T, sizeY(L))
   Zmat = reshape(Z, 1, :)   # this is a view, not a copy!
   compute!(Zmat, basis, SA[𝐫,])
   return Z 
end 

function compute(basis::SolidHarmonics{L, NORM, STATIC}, 
                  Rs::AbstractVector{SVector{3, T}}
                  ) where {L, NORM, STATIC, T}
   # note here we are NOT using the type of the Flm. If the Flm type  is 
   # different from the Rs type then there will be an implicit conversion 
   Z = similar(Rs, T, (length(Rs), sizeY(L))) # we could make this cached as well 
   compute!(Z, basis, Rs)
   return Z
end


function compute!(Z::AbstractMatrix,
                  basis::SolidHarmonics{L, NORM, STATIC},
                  Rs::AbstractVector{<: SVector{3}}
                  ) where {L, NORM, STATIC}
   # single batched path for all backends: KernelAbstractions, one point per
   # work-item; the CPU backend auto-multithreads across the batch.
   ka_solid_harmonics!(Z, nothing, Val{L}(), Rs, basis.Flm)
   return Z
end


# ---------- gradients 

function compute_with_gradients(basis::SolidHarmonics{L, NORM, false}, 
                                𝐫::SVector{3, T}
                               ) where {L, NORM, T}
   Z = zeros(T, sizeY(L))
   dZ = zeros(SVector{3, T}, sizeY(L))
   Zmat = reshape(Z, 1, :)   # this is a view, not a copy!
   dZmat = reshape(dZ, 1, :)
   compute_with_gradients!(Zmat, dZmat, basis, SA[𝐫,])
   return Z, dZ 
end 

function compute_with_gradients(basis::SolidHarmonics{L, NORM, true},
                                𝐫::SVector{3, T}
                               ) where {L, NORM, T}
   return static_solid_harmonics_with_grads(Val{L}(), 𝐫, basis.Flm)
end


function compute_with_gradients(basis::SolidHarmonics{L, NORM, STATIC}, 
                                Rs::AbstractVector{SVector{3, T}}
                                ) where {L, NORM, STATIC, T}
   Z = similar(Rs, T, (length(Rs), sizeY(L))) # we could make this cached as well 
   dZ = similar(Rs, SVector{3, T}, (length(Rs), sizeY(L))) 
   compute_with_gradients!(Z, dZ, basis, Rs)
   return Z, dZ 
end



function compute_with_gradients!(
            Z::AbstractMatrix,
            dZ::AbstractMatrix,
            basis::SolidHarmonics{L, NORM, STATIC},
            Rs::AbstractVector{<: SVector{3}}
            ) where {L, NORM, STATIC}
   ka_solid_harmonics!(Z, dZ, Val{L}(), Rs, basis.Flm)
   return Z, dZ
end
