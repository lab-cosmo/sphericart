
"""
`struct SolidHarmonics` : datatype representing a solid harmonics basis. 

### Constructor to generate the basis object
```julia
basis = SolidHarmonic(L::Integer; kwargs...)
```

### Keyword arguments:
* `normalisation = :L2` : choose the normalisation of the basis, default is to 
   make it orthonormal on the unit sphere. 
* `static = (L<=15)` : decide whether to use a generated code that outputs an 
`SVector` but has a larger compiler and stack footprint
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
                        static = (L <= 15), 
                        T = Float64) 
   Flm = generate_Flms(L; normalisation = normalisation, T = T)
   @assert eltype(Flm) == T   
   SolidHarmonics{L, normalisation, static, typeof(Flm)}(Flm)
end

@inline (basis::SolidHarmonics)(args...) = compute(basis, args...)


@inline function compute(basis::SolidHarmonics{L, NORM, true}, 𝐫::SVector{3}
                 ) where {L, NORM} 
   return static_solid_harmonics(Val{L}(), 𝐫, Val{NORM}())
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
   Z = zeros(T, length(Rs), sizeY(L)) # we could make this cached as well 
   compute!(Z, basis, Rs)
   return Z
end

function compute!(Z::AbstractMatrix, 
                  basis::SolidHarmonics{L, NORM, STATIC}, 
                  Rs::AbstractVector{SVector{3, T}}
                  ) where {L, NORM, STATIC, T}

   nX = length(Rs)

   @no_escape begin 

      # allocate temporary arrays from an array cache 
      temps = (x = @alloc(T, nX), 
               y = @alloc(T, nX),
               z = @alloc(T, nX), 
              r² = @alloc(T, nX),
               s = @alloc(T, nX, L+1), 
               c = @alloc(T, nX, L+1),
               Q = @alloc(T, nX, sizeY(L)),
             Flm = basis.Flm )

      # the actual evaluation kernel 
      solid_harmonics!(Z, Val{L}(), Rs, temps)

      nothing
   end # @no_escape 

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
   return static_solid_harmonics_with_grads(Val{L}(), 𝐫, Val{NORM}())
end 


function compute_with_gradients(basis::SolidHarmonics{L, NORM, STATIC}, 
                                Rs::AbstractVector{SVector{3, T}}
                                ) where {L, NORM, STATIC, T}
   Z = zeros(T, length(Rs), sizeY(L)) # we could make this cached as well 
   dZ = zeros(SVector{3, T}, length(Rs), sizeY(L)) 
   compute_with_gradients!(Z, dZ, basis, Rs)
   return Z, dZ 
end



function compute_with_gradients!(
            Z::AbstractMatrix, 
            dZ::AbstractMatrix,
            basis::SolidHarmonics{L, NORM, STATIC}, 
            Rs::AbstractVector{SVector{3, T}}
            ) where {L, NORM, STATIC, T}

   nX = length(Rs)

   @no_escape begin 

      # allocate temporary arrays from an array cache 
      temps = (x = @alloc(T, nX),    
               y = @alloc(T, nX),    
               z = @alloc(T, nX),    
              r² = @alloc(T, nX),    
               s = @alloc(T, nX, L+1), 
               c = @alloc(T, nX, L+1), 
               Q = @alloc(T, nX, sizeY(L)), 
             Flm = basis.Flm )

      # the actual evaluation kernel 
      solid_harmonics_with_grad!(Z, dZ, Val{L}(), Rs, temps)
      nothing
   end

   return Z 
end 
