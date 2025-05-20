
#  This is a generated code for best possible performance with a single input

import ForwardDiff
using ForwardDiff: Dual 

_codegen_Zlm(L, T::Dual, normalisation) = 
   _codegen_Zlm(L, ForwardDiff.valtype(T), normalisation)

function _codegen_Zlm(L, T, normalisation) 
   Flm = generate_Flms(L; normalisation = normalisation, T = T)
   len = sizeY(L)
   rt2 = sqrt(T(2)) 

   code = Expr[] 
   push!(code, :(r² = x^2 + y^2 + z^2))

   # c_m and s_m 
   push!(code, :(s_0 = zero($T)))
   push!(code, :(c_0 = one($T)))
   for m = 1:L 
      push!(code, Meta.parse("s_$m = s_$(m-1) * x + c_$(m-1) * y"))
      push!(code, Meta.parse("c_$m = c_$(m-1) * x - s_$(m-1) * y"))
   end

   # redefine c_0 = 1/√2 => this allows us to avoid special casing m=0
   push!(code, Meta.parse("c_0 = one($T)/$rt2"))

   # Q_0^0 and Y_0^0
   push!(code, Meta.parse("Q_0_0 = one($T)"))
   push!(code, Meta.parse("Z_1 = $(Flm[1+0,1+0]/rt2) * Q_0_0"))

   for l = 1:L 
      # Q_l^l and Y_l^l
      # m = l 
      push!(code, Meta.parse("Q_$(l)_$(l)  = - $(2*l-1) * Q_$(l-1)_$(l-1)"))
      push!(code, Meta.parse("Z_$(lm2idx(l, l))  = $(Flm[1+l,1+l]) * Q_$(l)_$(l) * c_$(l)"))
      push!(code, Meta.parse("Z_$(lm2idx(l, -l)) = $(Flm[1+l,1+ l]) * Q_$(l)_$(l) * s_$(l)"))
      # Q_l^l-1 and Y_l^l-1
      # m = l-1 
      push!(code, Meta.parse("Q_$(l)_$(l-1)  = $(2*l-1) * z * Q_$(l-1)_$(l-1)"))
      push!(code, Meta.parse("Z_$(lm2idx(l, -l+1)) = $(Flm[1+l, 1+ l-1]) * Q_$(l)_$(l-1) * s_$(l-1)"))
      push!(code, Meta.parse("Z_$(lm2idx(l, l-1) ) = $(Flm[1+l, 1+ l-1]) * Q_$(l)_$(l-1) * c_$(l-1)" )) # overwrite if m = 0 -> ok 
      # now we can go to the second recursion 
      for m = l-2:-1:0
         push!(code, Meta.parse("Q_$(l)_$(m)  = $((2*l-1)/(l-m)) * z * Q_$(l-1)_$m - $((l+m-1)/(l-m)) * r² * Q_$(l-2)_$(m)"))
         push!(code, Meta.parse("Z_$(lm2idx(l,-m)) = $(Flm[1+l, 1+ m]) * Q_$(l)_$(m) * s_$(m)"))
         push!(code, Meta.parse("Z_$(lm2idx(l,m) ) = $(Flm[1+l, 1+ m]) * Q_$(l)_$(m) * c_$(m)"))
      end
   end

   # finally generate an svector output
   push!(code, Meta.parse("return SVector{$len, $T}(" * 
                join( ["Z_$i, " for i = 1:len], ) * ")"))
end


function _codegen_Zlm_grads(L, T, normalisation) 
   Flm = generate_Flms(L; normalisation = normalisation, T = T)
   len = sizeY(L)
   rt2 = sqrt(T(2)) 

   code = Expr[] 
   push!(code, :(r² = x^2 + y^2 + z^2))

   # c_m and s_m 
   push!(code, :(s_0 = zero($T)))
   push!(code, :(c_0 = one($T)))
   for m = 1:L 
      push!(code, Meta.parse("s_$m = s_$(m-1) * x + c_$(m-1) * y"))
      push!(code, Meta.parse("c_$m = c_$(m-1) * x - s_$(m-1) * y"))
   end

   # l = 0 
   # Q_0^0 and Y_0^0
   push!(code, Meta.parse("Q_0_0 = one($T)"))
   push!(code, Meta.parse("Z_1 = $(Flm[1+0,1+0]/rt2) * Q_0_0"))

   # gradients
   push!(code, Meta.parse("dZ_1 = zero(SVector{3, $T})"))


   # l = 1 special case 
   # Q_1^1 => Y_1^1, Y_1^-1
   push!(code, Meta.parse("Q_1_1  = - Q_0_0"))
   push!(code, Meta.parse("Z_$(lm2idx(1,  1)) = $(-Flm[1+1, 1+1]) * c_1"))
   push!(code, Meta.parse("Z_$(lm2idx(1, -1)) = $(-Flm[1+1, 1+1]) * s_1"))
   # Q_1^0 and Y_1^0
   push!(code, Meta.parse("Q_1_0  = z"))
   push!(code, Meta.parse("Z_$(lm2idx(1, 0)) = $(Flm[1+1, 1+ 0]/rt2) * Q_1_0 * c_0"))

   # gradients       
   push!(code, Meta.parse("dZ_$(lm2idx(1,  1)) = SA[ $(-Flm[1+1, 1+1]), zero($T), zero($T) ]"))
   push!(code, Meta.parse("dZ_$(lm2idx(1, -1)) = SA[ zero($T), $(-Flm[1+1, 1+1]), zero($T) ]"))
   push!(code, Meta.parse("dZ_$(lm2idx(1,  0)) = SA[ zero($T), zero($T), $(Flm[1+1, 1+ 0]/rt2) ]"))

   for l = 2:L 
      # Q_l^l => Y_l^l, Y_l^-l
      push!(code, Meta.parse("Q_$(l)_$(l)  = - $(2*l-1) * Q_$(l-1)_$(l-1)"))
      push!(code, Meta.parse("Z_$(lm2idx(l, l))  = $(Flm[1+l, 1+l]) * Q_$(l)_$(l) * c_$(l)"))
      push!(code, Meta.parse("Z_$(lm2idx(l, -l)) = $(Flm[1+l, 1+ l]) * Q_$(l)_$(l) * s_$(l)"))
      # Q_l^l-1 => Y_l^l-1, Y_l^-l+1
      push!(code, Meta.parse("Q_$(l)_$(l-1)  = $(2*l-1) * z * Q_$(l-1)_$(l-1)"))
      push!(code, Meta.parse("Z_$(lm2idx(l, -l+1)) = $(Flm[1+l, 1+ l-1]) * Q_$(l)_$(l-1) * s_$(l-1)"))
      push!(code, Meta.parse("Z_$(lm2idx(l, l-1) ) = $(Flm[1+l, 1+ l-1]) * Q_$(l)_$(l-1) * c_$(l-1)" )) # overwrite if m = 0 -> ok 

      # gradients 
      push!(code, Meta.parse("dZ_$(lm2idx(l, l)) = $(Flm[1+l, 1+l]) * Q_$(l)_$(l) * SA[ $l * c_$(l-1), - $l * s_$(l-1), zero($T) ]"))
      push!(code, Meta.parse("dZ_$(lm2idx(l, -l)) = $(Flm[1+l, 1+l]) * Q_$(l)_$(l) * SA[ $l * s_$(l-1),  $l * c_$(l-1), zero($T) ]"))
      push!(code, Meta.parse("""dZ_$(lm2idx(l, -l+1)) = $(Flm[1+l, 1+ l-1]) * SA[ Q_$(l)_$(l-1) * $(l-1) * s_$(l-2), 
                                                                             Q_$(l)_$(l-1) * $(l-1) * c_$(l-2), 
                                                                             $(2*l-1) * Q_$(l-1)_$(l-1) * s_$(l-1) ]"""))
      push!(code, Meta.parse("""dZ_$(lm2idx(l,  l-1)) = $(Flm[1+l, 1+ l-1]) * SA[ Q_$(l)_$(l-1) * $(l-1) * c_$(l-2), 
                                                                             Q_$(l)_$(l-1) * $(-l+1) * s_$(l-2), 
                                                                             $(2*l-1) * Q_$(l-1)_$(l-1) * c_$(l-1) ]"""))

      # now we can go to the second recursion 
      for m = l-2:-1:1
         push!(code, Meta.parse("Q_$(l)_$(m)  = $((2*l-1)/(l-m)) * z * Q_$(l-1)_$m - $((l+m-1)/(l-m)) * r² * Q_$(l-2)_$(m)"))
         push!(code, Meta.parse("Z_$(lm2idx(l,-m)) = $(Flm[1+l, 1+ m]) * Q_$(l)_$(m) * s_$(m)"))
         push!(code, Meta.parse("Z_$(lm2idx(l,m) ) = $(Flm[1+l, 1+ m]) * Q_$(l)_$(m) * c_$(m)"))

         # gradients 
         push!(code, Meta.parse(""" 
            dZ_$(lm2idx(l, -m)) = $(Flm[1+l, 1+ m]) * SA[ Q_$(l)_$(m) * $(m) * s_$(m-1) + x * Q_$(l-1)_$(m+1) * s_$(m), 
                                                     Q_$(l)_$(m) * $(m) * c_$(m-1) + y * Q_$(l-1)_$(m+1) * s_$(m), 
                                                     $(l+m) * Q_$(l-1)_$(m) * s_$m ]"""))
         push!(code, Meta.parse("""
            dZ_$(lm2idx(l, m)) = $(Flm[1+l, 1+ m]) * SA[ Q_$(l)_$(m) * $(m) * c_$(m-1) + x * Q_$(l-1)_$(m+1) * c_$(m), 
                                                    Q_$(l)_$(m) * $(-m) * s_$(m-1) + y * Q_$(l-1)_$(m+1) * c_$(m), 
                                                    $(l+m) * Q_$(l-1)_$(m) * c_$m ]"""))
      end

      # special-case m = 0 
      if l >= 2 
         push!(code, Meta.parse("Q_$(l)_0  = $((2*l-1)/l) * z * Q_$(l-1)_0 - $((l-1)/(l)) * r² * Q_$(l-2)_0"))
         push!(code, Meta.parse("Z_$(lm2idx(l,0) ) = $(Flm[1+l, 1+ 0] / rt2) * Q_$(l)_0"))

         # gradients
         # dZ[j, il0] = F_l_0_f * cj * SA[Q_l0_x, Q_l0_y, Q_l0_z ]                                                 
         push!(code, Meta.parse("dZ_$(lm2idx(l, 0)) = $(Flm[1+l, 1+ 0] / rt2) * SA[ Q_$(l-1)_1 * x, Q_$(l-1)_1 * y, $(l) * Q_$(l-1)_0 ]"))
      end
   end

   # finally generate an svector output
   push!(code, Meta.parse("Z = SVector{$len, $T}(" * 
                join( ["Z_$i, " for i = 1:len], ) * ")"))

   push!(code, Meta.parse("dZ = SVector{$len, SVector{3, $T}}(" * 
                join( ["dZ_$i, " for i = 1:len] ) * ")"))

   push!(code, Meta.parse("return Z, dZ"))
end



"""
`static_solid_harmonics`: evaluate the solid harmonics basis for a single 
input point. The code is fully generated and unrolled. The return value 
is an `SVector{LEN, T}` where `LEN` is the length of the basis and `T` the 
element type of the input point. 

Usage: e.g. for `L = 4`
```julia 
valL = Val{4}() 
𝐫 = @SVector randn(3) 
Z = static_solid_harmonics(valL, 𝐫) 
x, y, z = tuple(𝐫...)
Z = static_solid_harmonics(valL, x, y, z) 
```

Once can also specify the normalisation convention, e.g., 
```julia
Z = static_solid_harmonics(valL, 𝐫, Val{:L2}()) 
```
which would be the default behaviour.
"""
static_solid_harmonics(valL::Val{L}, 𝐫::SVector{3}, 
                     valNorm = Val{:sphericart}()) where {L} = 
      static_solid_harmonics(valL, 𝐫[1], 𝐫[2], 𝐫[3], valNorm)

@generated function static_solid_harmonics(::Val{L}, x::T, y::T, z::T, 
                     valNorm::Val{NORM} = Val{:sphericart}()
                     ) where {L, T, NORM}
   code = _codegen_Zlm(L, T, NORM)
   return quote
      $(Expr(:block, code...))
   end
end


static_solid_harmonics_with_grads(valL::Val{L}, 𝐫::SVector{3}, 
                     valNorm = Val{:sphericart}()) where {L} = 
         static_solid_harmonics_with_grads(valL, 𝐫[1], 𝐫[2], 𝐫[3], valNorm)


@generated function static_solid_harmonics_with_grads(::Val{L}, x::T, y::T, z::T, 
                     valNorm::Val{NORM} = Val{:sphericart}()
                     ) where {L, T, NORM}
   code = _codegen_Zlm_grads(L, T, NORM)
   return quote
      $(Expr(:block, code...))
   end
end

