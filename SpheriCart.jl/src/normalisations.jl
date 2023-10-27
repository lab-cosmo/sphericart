


# Generates the `F[l, m]` values exactly as described in the 
# `sphericart` publication. 
function _generate_Flms(L::Integer, ::Val{:sphericart})
   Flm = OffsetMatrix(zeros(L+1, L+1), (-1, -1))
   for l = 0:L
      Flm[l, 0] = sqrt((2*l+1)/(2 * π))
      for m = 1:l 
         Flm[l, m] = - Flm[l, m-1] / sqrt((l+m) * (l+1-m))
      end
   end
   return Flm
end

# alias for :sphericart
_generate_Flms(L::Integer, ::Val{:L2}) = 
      _generate_Flms(L, Val{:sphericart}())


function _generate_Flms(L::Integer, ::Val{:p4ml})
   Flm = OffsetMatrix(zeros(L+1, L+1), (-1, -1))
   for l = 0:L
      Flm[l, 0] = sqrt((2*l+1)/(2 * π))
      for m = 1:l
         Flm[l, m] = (-1)^m * sqrt((2*l+1)/(4*π) )
      end
   end
   return Flm
end

"""
```
generate_Flms(L; normalisation = :L2)
```
generate the `F[l,m]` prefactors in the definitions of the solid harmonics; 
see `sphericart` publication for details. The default normalisation generates 
a basis that is L2-orthonormal on the unit sphere. Other normalisations: 
- `:sphericart` the same as `:L2`
- `:p4ml` same normalisation as used in `Polynomials4ML.jl`
"""
generate_Flms(L::Integer; normalisation = :L2) = 
      _generate_Flms(L, Val(normalisation))
