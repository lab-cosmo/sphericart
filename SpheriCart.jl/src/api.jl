
struct SolidHarmonics{L, NORM, STATIC, T1}
   Flm::OffsetMatrix{T1, Matrix{T1}}
   cache::ArrayPool{FlexArrayCache}
end

function SolidHarmonics(L::Integer; 
                        normalisation = :L2, 
                        static = (L <= 15)) 
   Flm = generate_Flms(L; normalisation = normalisation)
   SolidHarmonics{L, normalisation, static, eltype(Flm)}(Flm, ArrayPool(FlexArrayCache))
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
   Zmat = Z'   # this is a view, not a copy!
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
