
struct SolidHarmonics{L, T1}
   Flm::OffsetMatrix{T1, Matrix{T1}}
   cache::ArrayPool{FlexArrayCache}
end

function SolidHarmonics(L::Integer) 
   Flm = generate_Flms(L)
   SolidHarmonics{L, eltype(Flm)}(Flm, ArrayPool(FlexArrayCache))
end

function compute(basis::SolidHarmonics{L, T1}, 
                 Rs::AbstractVector{SVector{3, T2}}) where {L, T1, T2}
   T = promote_type(T1, T2)                 
   Z = zeros(T, length(Rs), sizeY(L)) # we could make this cached as well 
   compute!(Z, basis, Rs)
   return Z
end

function compute!(Z::AbstractMatrix, 
                  basis::SolidHarmonics{L, T1}, 
                  Rs::AbstractVector{SVector{3, T2}}) where {L, T1, T2}

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
