
export SphericalHarmonics

using LinearAlgebra: norm, dot 

"""
`struct SphericalHarmonics` : datatype representing a real spherical harmonics basis. 

### Constructor to generate the basis object
```julia
basis = SphericalHarmonics(L::Integer; kwargs...)
```

### Keyword arguments:
* `normalisation = :L2` : choose the normalisation of the basis, default is to 
   make it orthonoormal on the unit sphere. 
* `static = (L<=15)` : decide whether to use a generated code that outputs an 
`SVector` but has a larger compiler and stack footprint
* `T = Float64` : datatype in which basis parameters are stored. The output type 
is inferred at runtime, but the rule of thumb is to use `T = FloatX` for 
`FloatX` output.

### Usage example: 
```julia
using StaticArrays, SpheriCart
basis = SphericalHarmonics(4)
# evaluate basis with single input 
𝐫 = @SVector randn(3)
Z = basis(𝐫)
Z = compute(basis, 𝐫)
# evaluate basis with multiple inputs (batching)
R = [ @SVector randn(3) for _ = 1:32 ]
Z = basis(Rs)
Z = compute(basis, Rs)

# to be implented: 
# Z, ∇Z = compute_and_gradients(basis, 𝐫)
# Z, ∇Z, ∇²Z = compute_and_hessian(basis, 𝐫)
```
See documentation for more details.
"""
struct SphericalHarmonics{L, NORM, STATIC, T1}
   solids::SolidHarmonics{L, NORM, STATIC, T1}
   # Flm::OffsetMatrix{T1, Matrix{T1}}
   cache::TSafe{ArrayPool{FlexArrayCache}}
end

SphericalHarmonics(L::Integer; kwargs...) = 
      SphericalHarmonics(SolidHarmonics(L; kwargs...), 
                         TSafe(ArrayPool(FlexArrayCache)))


@inline (basis::SphericalHarmonics)(args...) = compute(basis, args...)

@inline function compute(basis::SphericalHarmonics, 𝐫::SVector{3})
   𝐫̂ = 𝐫 / norm(𝐫)                        
   return compute(basis.solids, 𝐫̂)
end 

@inline function compute_with_gradients(basis::SphericalHarmonics, 𝐫::SVector{3})
   r = norm(𝐫)
   𝐫̂ = 𝐫 / r
   Y, ∇Z = compute_with_gradients(basis.solids, 𝐫̂)

   ∇Y = map(dZ -> (dz = dZ / r; dz - dot(𝐫̂, dz) * 𝐫̂), ∇Z)

   # @inbounds @simd ivdep for i = 1:length(Y)
   #    dz = ∇Y[i] / r
   #    ∇Y[i] = dz - dot(𝐫̂, dz) * 𝐫̂
   # end

   return Y, ∇Y
end 

# --------------------- 
#  batched api 

function _normalise_Rs!(basis::SphericalHarmonics, 
                        Rs::AbstractVector{SVector{3, T1}}) where {T1}
   nX = length(Rs) 
   rs = acquire!(basis.cache, :rs, (nX, ), T1)
   Rs_norm = acquire!(basis.cache, :Rs_norm, (nX, ), SVector{3, T1})
   @inbounds @simd ivdep for i = 1:nX
      rs[i] = norm(Rs[i])
      Rs_norm[i] = Rs[i] / rs[i]
   end
   return rs, Rs_norm 
end

function _rescale_∇Z2∇Y!(∇Z::AbstractMatrix, Rs_norm, rs)
   nX = length(rs)
   @inbounds for i = 1:size(∇Z, 2)
      @simd ivdep for j = 1:nX
         dzj = ∇Z[j, i] / rs[j]
         𝐫̂j = Rs_norm[j]
         ∇Z[j, i] = dzj - dot(𝐫̂j, dzj) * 𝐫̂j
      end
   end
end

function compute(basis::SphericalHarmonics, 
                         Rs::AbstractVector{<: SVector{3}})
   rs, Rs_norm = _normalise_Rs!(basis, Rs)
   Y = compute(basis.solids, Rs_norm)
   release!(Rs_norm)
   release!(rs)
   return Y
end

function compute!(Y, basis::SphericalHarmonics, 
                     Rs::AbstractVector{<: SVector{3}})
   rs, Rs_norm = _normalise_Rs!(basis, Rs)
   compute!(Y, basis.solids, Rs_norm)
   release!(Rs_norm)
   release!(rs)
   return Y
end


function compute_with_gradients(basis::SphericalHarmonics, 
                           Rs::AbstractVector{<: SVector{3}})
   rs, Rs_norm = _normalise_Rs!(basis, Rs)
   Y, ∇Z = compute_with_gradients(basis.solids, Rs_norm)
   _rescale_∇Z2∇Y!(∇Z, Rs_norm, rs)

   release!(Rs_norm)
   release!(rs)

   return Y, ∇Z
end 

function compute_with_gradients!(Y, ∇Y, basis::SphericalHarmonics, 
                                 Rs::AbstractVector{<: SVector{3}})
   rs, Rs_norm = _normalise_Rs!(basis, Rs)
   compute_with_gradients!(Y, ∇Y, basis.solids, Rs_norm)
   _rescale_∇Z2∇Y!(∇Y, Rs_norm, rs)

   release!(Rs_norm)
   release!(rs)

   return Y, ∇Y
end 
