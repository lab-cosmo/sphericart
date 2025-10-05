
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
struct SphericalHarmonics{L, NORM, STATIC, TF}
   solids::SolidHarmonics{L, NORM, STATIC, TF}
end

SphericalHarmonics(L::Integer; kwargs...) = 
      SphericalHarmonics(SolidHarmonics(L; kwargs...))

Base.getproperty(basis::SphericalHarmonics, prop::Symbol) = (
      prop == :solids ? getfield(basis, :solids) 
                      : getfield(getfield(basis, :solids), prop) )

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

function _normalise_Rs!(rs, Rs_norm, 
                        basis::SphericalHarmonics, 
                        Rs::AbstractVector{SVector{3, T1}}) where {T1}
   nX = length(Rs) 
   @assert length(rs) == length(Rs_norm) == nX
   @inbounds @simd ivdep for i = 1:nX
      rs[i] = norm(Rs[i])
      Rs_norm[i] = Rs[i] / rs[i]
   end
   return nothing 
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
                  Rs::AbstractVector{<: SVector{3, T1}}
                  ) where {T1}  
   @no_escape begin     
      nX = length(Rs)              
      rs = @alloc(T1, nX)
      Rs_norm = @alloc(SVector{3, T1}, nX)
      _normalise_Rs!(rs, Rs_norm, basis, Rs)
      Y = compute(basis.solids, Rs_norm)
   end
   return Y
end


function compute!(Y, basis::SphericalHarmonics, 
                  Rs::AbstractVector{<: SVector{3, T1}}
                  ) where {T1}
   @no_escape begin       
      nX = length(Rs)            
      rs = @alloc(T1, nX)
      Rs_norm = @alloc(SVector{3, T1}, nX)
      _normalise_Rs!(rs, Rs_norm, basis, Rs)
      compute!(Y, basis.solids, Rs_norm)
      nothing 
   end
   return Y
end


function compute_with_gradients(basis::SphericalHarmonics, 
                  Rs::AbstractVector{<: SVector{3, T1}}
                  ) where {T1} 
   @no_escape begin     
      nX = length(Rs)              
      rs = @alloc(T1, nX)
      Rs_norm = @alloc(SVector{3, T1}, nX)
      _normalise_Rs!(rs, Rs_norm, basis, Rs)
      Y, ∇Z = compute_with_gradients(basis.solids, Rs_norm)
      _rescale_∇Z2∇Y!(∇Z, Rs_norm, rs)
      nothing 
   end 
   return Y, ∇Z
end 

function compute_with_gradients!(Y, ∇Y, basis::SphericalHarmonics, 
                        Rs::AbstractVector{<: SVector{3, T1}}
                        ) where {T1}
   @no_escape begin      
      nX = length(Rs)             
      rs = @alloc(T1, nX)
      Rs_norm = @alloc(SVector{3, T1}, nX)
      _normalise_Rs!(rs, Rs_norm, basis, Rs)
      compute_with_gradients!(Y, ∇Y, basis.solids, Rs_norm)
      _rescale_∇Z2∇Y!(∇Y, Rs_norm, rs)
      nothing 
   end 
   return Y, ∇Y
end 


# --------------------- 
#  KernelAbstractions api 

using KernelAbstractions
using GPUArraysCore: AbstractGPUVector, AbstractGPUMatrix

function compute(basis::SphericalHarmonics, 
                  Rs::AbstractGPUVector{<: SVector{3, T1}}
                  ) where {T1}
   nX = length(Rs)              
   Rs_norm = map(𝐫 -> 𝐫 / norm(𝐫), Rs)
   Y = compute(basis.solids, Rs_norm)
   return Y
end


@kernel function _ka_rescale_∇Z2∇Y!(
                  ∇Z, @Const(Rs_norm), @Const(rs))
   j, i = @index(Global, NTuple)
   dzj = ∇Z[j, i] / rs[j]
   𝐫̂j = Rs_norm[j]
   ∇Z[j, i] = dzj - dot(𝐫̂j, dzj) * 𝐫̂j
   nothing 
end

function _rescale_∇Z2∇Y!(∇Z::AbstractGPUMatrix, Rs_norm, rs)
   backend = KernelAbstractions.get_backend(∇Z)
   kernel! = _ka_rescale_∇Z2∇Y!(backend)
   nRs, nZ = size(∇Z)
   kernel!(∇Z, Rs_norm, rs; ndrange = (nRs, nZ))
   synchronize(backend)
end


function compute_with_gradients(basis::SphericalHarmonics, 
                  Rs::AbstractGPUVector{<: SVector{3, T1}}
                  ) where {T1} 
   nX = length(Rs)              
   rs = map(𝐫 -> norm(𝐫), Rs)
   Rs_norm = map(𝐫 -> 𝐫 / norm(𝐫), Rs)
   Y, ∇Z = compute_with_gradients(basis.solids, Rs_norm)
   # ∇Y = map( (dz, 𝐫̂, r) -> (dz - dot(𝐫̂, dz) * 𝐫̂) / r, 
   #           ∇Z, Rs_norm, rs)
   # @show size(∇Y)             
   _rescale_∇Z2∇Y!(∇Z, Rs_norm, rs)
   return Y, ∇Z
end 
