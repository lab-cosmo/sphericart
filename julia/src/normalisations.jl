


# Generates the `F[l, m]` values exactly as described in the 
# `sphericart` publication.  This gives L2-orthonormality. 
function _generate_Flms(L::Integer, ::Union{Val{:sphericart}, Val{:L2}}, T=Float64)
   Flm = OffsetMatrix(zeros(L+1, L+1), (-1, -1))
   for l = 0:L
      Flm[l, 0] = sqrt((2*l+1)/(2 * π))
      for m = 1:l 
         Flm[l, m] = - Flm[l, m-1] / sqrt((l+m) * (l+1-m))
      end
   end
   return Flm
end

function _generate_Flms(L::Integer, ::Val{:racah}, T=Float64)
   Flm = _generate_Flms(L, Val(:L2), T)
   for l = 0:L 
      for m = 0:l
         Flm[l, m] = Flm[l, m] * sqrt(4*pi/(2*l+1))
      end
   end
   return Flm
end


"""
```
generate_Flms(L; normalisation = :L2, T = Float64)
```
generate the `F[l,m]` prefactors in the definitions of the solid harmonics; 
see `sphericart` publication for details. The default normalisation generates 
a basis that is L2-orthonormal on the unit sphere. Other normalisations: 
- `:sphericart` the same as `:L2`, gives L2-orthonormality, i.e. ∫ |Ylm|² = 1
- `:racah` gives Racah normalization for which ∫ |Ylm|² = 4π/(2l+1).
"""
generate_Flms(L::Integer; normalisation = :L2, T = Float64) = 
      _generate_Flms(L, Val(normalisation), T)
