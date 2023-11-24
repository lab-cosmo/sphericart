"""
`struct SolidHarmonics` : datatype representing a solid harmonics basis. 

### Constructor to generate the basis object
```julia
basis = SolidHarmonic(L::Integer; kwargs...)
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
basis = SolidHarmonics(4)
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
struct SolidHarmonics{L, NORM, STATIC, T1}
   Flm::OffsetMatrix{T1, Matrix{T1}}
   cache::TSafe{ArrayPool{FlexArrayCache}}
end

function SolidHarmonics(L::Integer; 
                        normalisation = :L2, 
                        static = (L <= 15), 
                        T = Float64) 
   Flm = generate_Flms(L; normalisation = normalisation, T = T)
   @assert eltype(Flm) == T   
   SolidHarmonics{L, normalisation, static, T}(Flm, TSafe(ArrayPool(FlexArrayCache)))
end

@inline (basis::SolidHarmonics)(args...) = compute(basis, args...)


@inline function compute(basis::SolidHarmonics{L, NORM, true}, 𝐫::SVector{3}
                 ) where {L, NORM} 
   return static_solid_harmonics(Val{L}(), 𝐫, Val{NORM}())
end 

function compute(basis::SolidHarmonics{L, NORM, false, T1}, 𝐫::SVector{3, T2}
         ) where {L, NORM, T1, T2}
   T = promote_type(T1, T2)
   Z = zeros(T, sizeY(L))
   Zmat = reshape(Z, 1, :)   # this is a view, not a copy!
   compute!(Zmat, basis, SA[𝐫,])
   return Z 
end 

function compute(basis::SolidHarmonics{L, NORM, STATIC, T1}, 
                  Rs::AbstractVector{SVector{3, T2}}
                  ) where {L, NORM, STATIC, T1, T2}
   T = promote_type(T1, T2)
   Z = zeros(T, length(Rs), sizeY(L)) # we could make this cached as well 
   compute!(Z, basis, Rs)
   return Z
end

function compute!(Z::AbstractMatrix, 
                  basis::SolidHarmonics{L, NORM, STATIC, T1}, 
                  Rs::AbstractVector{SVector{3, T2}}
                  ) where {L, NORM, STATIC, T1, T2}

   nX = length(Rs)
   T = promote_type(T1, T2)

   # allocate temporary arrays from an array cache 
   temps = (x = acquire!(basis.cache, :x,  (nX, ),    T),
            y = acquire!(basis.cache, :y,  (nX, ),    T),
            z = acquire!(basis.cache, :z,  (nX, ),    T), 
           r² = acquire!(basis.cache, :r2, (nX, ),    T),
            s = acquire!(basis.cache, :s,  (nX, L+1), T),
            c = acquire!(basis.cache, :c,  (nX, L+1), T),
            Q = acquire!(basis.cache, :Q,  (nX, sizeY(L)), T), 
            Flm = basis.Flm )

   # the actual evaluation kernel 
   solid_harmonics!(Z, Val{L}(), Rs, temps)

   # release the temporary arrays back into the cache
   # (don't release Flm!!)
   release!(temps.x)
   release!(temps.y)
   release!(temps.z)
   release!(temps.r²)
   release!(temps.s)
   release!(temps.c)
   release!(temps.Q)

   return Z 
end 


# ---------- gradients 

function compute_with_gradients(basis::SolidHarmonics{L, NORM, false, T1}, 
                                𝐫::SVector{3, T2}
                               ) where {L, NORM, T1, T2}
   T = promote_type(T1, T2)
   Z = zeros(T, sizeY(L))
   dZ = zeros(SVector{3, T}, sizeY(L))
   Zmat = reshape(Z, 1, :)   # this is a view, not a copy!
   dZmat = reshape(dZ, 1, :)
   compute_with_gradients!(Zmat, dZmat, basis, SA[𝐫,])
   return Z, dZ 
end 

function compute_with_gradients(basis::SolidHarmonics{L, NORM, true, T1}, 
                                𝐫::SVector{3, T2}
                               ) where {L, NORM, T1, T2}
   return static_solid_harmonics_with_grads(Val{L}(), 𝐫, Val{NORM}())
end 


function compute_with_gradients(basis::SolidHarmonics{L, NORM, STATIC, T1}, 
                                Rs::AbstractVector{SVector{3, T2}}
                                ) where {L, NORM, STATIC, T1, T2}
   T = promote_type(T1, T2)
   Z = zeros(T, length(Rs), sizeY(L)) # we could make this cached as well 
   dZ = zeros(SVector{3, T}, length(Rs), sizeY(L)) 
   compute_with_gradients!(Z, dZ, basis, Rs)
   return Z, dZ 
end



function compute_with_gradients!(
            Z::AbstractMatrix, 
            dZ::AbstractMatrix,
            basis::SolidHarmonics{L, NORM, STATIC, T1}, 
            Rs::AbstractVector{SVector{3, T2}}
            ) where {L, NORM, STATIC, T1, T2}

   nX = length(Rs)
   T = promote_type(T1, T2)

   # allocate temporary arrays from an array cache 
   temps = (x = acquire!(basis.cache, :x,  (nX, ),    T),
            y = acquire!(basis.cache, :y,  (nX, ),    T),
            z = acquire!(basis.cache, :z,  (nX, ),    T), 
           r² = acquire!(basis.cache, :r2, (nX, ),    T),
            s = acquire!(basis.cache, :s,  (nX, L+1), T),
            c = acquire!(basis.cache, :c,  (nX, L+1), T),
            Q = acquire!(basis.cache, :Q,  (nX, sizeY(L)), T),
            Flm = basis.Flm )

   # the actual evaluation kernel 
   solid_harmonics_with_grad!(Z, dZ, Val{L}(), Rs, temps)

   # release the temporary arrays back into the cache
   # (don't release Flm!!)
   release!(temps.x)
   release!(temps.y)
   release!(temps.z)
   release!(temps.r²)
   release!(temps.s)
   release!(temps.c)
   release!(temps.Q)

   return Z 
end 
