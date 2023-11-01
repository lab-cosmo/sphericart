
#  This is a generated code for best possible performance with a single input

function _codegen_Zlm(L, T, normalisation) 
   Flm = generate_Flms(L; normalisation = normalisation)
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
   push!(code, Meta.parse("Z_1 = $(Flm[0,0]/rt2) * Q_0_0"))

   for l = 1:L 
      # Q_l^l and Y_l^l
      # m = l 
      push!(code, Meta.parse("Q_$(l)_$(l)  = - $(2*l-1) * Q_$(l-1)_$(l-1)"))
      push!(code, Meta.parse("Z_$(lm2idx(l, l))  = $(Flm[l,l]) * Q_$(l)_$(l) * c_$(l)"))
      push!(code, Meta.parse("Z_$(lm2idx(l, -l)) = $(Flm[l, l]) * Q_$(l)_$(l) * s_$(l)"))
      # Q_l^l-1 and Y_l^l-1
      # m = l-1 
      push!(code, Meta.parse("Q_$(l)_$(l-1)  = $(2*l-1) * z * Q_$(l-1)_$(l-1)"))
      push!(code, Meta.parse("Z_$(lm2idx(l, -l+1)) = $(Flm[l, l-1]) * Q_$(l)_$(l-1) * s_$(l-1)"))
      push!(code, Meta.parse("Z_$(lm2idx(l, l-1) ) = $(Flm[l, l-1]) * Q_$(l)_$(l-1) * c_$(l-1)" )) # overwrite if m = 0 -> ok 
      # now we can go to the second recursion 
      for m = l-2:-1:0
         push!(code, Meta.parse("Q_$(l)_$(m)  = $((2*l-1)/(l-m)) * z * Q_$(l-1)_$m - $((l+m-1)/(l-m)) * r² * Q_$(l-2)_$(m)"))
         push!(code, Meta.parse("Z_$(lm2idx(l,-m)) = $(Flm[l, m]) * Q_$(l)_$(m) * s_$(m)"))
         push!(code, Meta.parse("Z_$(lm2idx(l,m) ) = $(Flm[l, m]) * Q_$(l)_$(m) * c_$(m)"))
      end
   end

   # finally generate an svector output
   push!(code, Meta.parse("return SVector{$len, $T}(" * 
                join( ["Z_$i, " for i = 1:len], ) * ")"))
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
                     ) where {L, T <: AbstractFloat, NORM}
   code = _codegen_Zlm(L, T, NORM)
   return quote
      $(Expr(:block, code...))
   end
end

