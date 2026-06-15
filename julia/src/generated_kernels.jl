
#  Generated, fully-unrolled single-point evaluation of the solid-harmonics
#  recursion (values + gradients), returning an `SVector`.
#
#  The normalisation prefactors `Flm` are READ FROM A PASSED ARGUMENT (an
#  `SMatrix`), not baked into the code. This keeps the *generator* free of any
#  element-type method calls: it only ever emits `Flm[i,j]` reads, integer
#  coefficients, and a runtime `rt2 = sqrt(T(2))`. Consequences:
#    * type stability for any `T` (no Float64 literals leaking into Float32);
#    * no world-age hazard -- the generator never calls `T`'s constructors, so it
#      works for element types loaded after this package (e.g. Quadmath Float128);
#    * smaller generated code (constants live in the data, not the instructions).
#  Benchmarks show this is performance-neutral in the small-L regime where the
#  unrolled path is used.
#
#  Gradient identities (cf. Eq. (10) in the sphericart publication):
#    ∂x c^m = m c^{m-1},  ∂y c^m = -m s^{m-1},  ∂x s^m = m s^{m-1},  ∂y s^m = m c^{m-1}
#    ∂x Q_l^m = x Q_{l-1}^{m+1},  ∂y Q_l^m = y Q_{l-1}^{m+1},  ∂z Q_l^m = (l+m) Q_{l-1}^m

import ForwardDiff
using ForwardDiff: Dual
using StaticArrays: StaticMatrix

"""
`_codegen(L, T; grad)`: build the list of expressions evaluating the solid
harmonics basis (and, if `grad`, its gradients) up to degree `L` for element
type `T`, reading the prefactors from a variable `Flm` and the coordinates from
`x, y, z`. Returns a `Vector{Expr}`.
"""
function _codegen(L, T; grad::Bool)
   len = sizeY(L)
   F(l, m) = "Flm[$(1+l), $(1+m)]"   # read prefactor F_l^m from the passed matrix
   # bake a real constant typed via the input `x` (the type witness): the
   # generator only does Float64 arithmetic + string building, never a `T`
   # method call (so it is world-age safe and `T` need not be in scope), while
   # the runtime sees a `T`-typed compile-time constant. `irt2` = 1/√2 baked,
   # so the m=0 / l=0,1 prefactors are a multiply, not a runtime division.
   lit(v) = "oftype(x, $(Float64(v)))"
   irt2 = lit(1 / sqrt(2))

   code = Expr[]
   push!(code, :(r² = x^2 + y^2 + z^2))

   # c_m and s_m  (c_0 = 1; m = 0 is special-cased with an explicit *1/√2)
   push!(code, :(s_0 = zero(x)))
   push!(code, :(c_0 = one(x)))
   for m = 1:L
      push!(code, Meta.parse("s_$m = s_$(m-1) * x + c_$(m-1) * y"))
      push!(code, Meta.parse("c_$m = c_$(m-1) * x - s_$(m-1) * y"))
   end

   # l = 0 :  Q_0^0 and Z_0^0
   push!(code, Meta.parse("Q_0_0 = one(x)"))
   push!(code, Meta.parse("Z_1 = $(F(0,0)) * $irt2 * Q_0_0"))
   if grad
      push!(code, Meta.parse("dZ_1 = zero(SVector{3, typeof(x)})"))
   end

   # l = 1 : special case (generic formulas below assume l >= 2)
   if L >= 1
      push!(code, Meta.parse("Q_1_1 = - Q_0_0"))
      push!(code, Meta.parse("Z_$(lm2idx(1, 1))  = - $(F(1,1)) * c_1"))
      push!(code, Meta.parse("Z_$(lm2idx(1,-1))  = - $(F(1,1)) * s_1"))
      push!(code, Meta.parse("Q_1_0 = z"))
      push!(code, Meta.parse("Z_$(lm2idx(1, 0))  = $(F(1,0)) * $irt2 * Q_1_0 * c_0"))
      if grad
         push!(code, Meta.parse("dZ_$(lm2idx(1, 1)) = SA[ - $(F(1,1)), zero(x), zero(x) ]"))
         push!(code, Meta.parse("dZ_$(lm2idx(1,-1)) = SA[ zero(x), - $(F(1,1)), zero(x) ]"))
         push!(code, Meta.parse("dZ_$(lm2idx(1, 0)) = SA[ zero(x), zero(x), $(F(1,0)) * $irt2 ]"))
      end
   end

   for l = 2:L
      # m = l
      push!(code, Meta.parse("Q_$(l)_$(l) = - $(2*l-1) * Q_$(l-1)_$(l-1)"))
      push!(code, Meta.parse("Z_$(lm2idx(l, l))  = $(F(l,l)) * Q_$(l)_$(l) * c_$(l)"))
      push!(code, Meta.parse("Z_$(lm2idx(l,-l))  = $(F(l,l)) * Q_$(l)_$(l) * s_$(l)"))
      # m = l-1
      push!(code, Meta.parse("Q_$(l)_$(l-1) = $(2*l-1) * z * Q_$(l-1)_$(l-1)"))
      push!(code, Meta.parse("Z_$(lm2idx(l,-l+1)) = $(F(l,l-1)) * Q_$(l)_$(l-1) * s_$(l-1)"))
      push!(code, Meta.parse("Z_$(lm2idx(l, l-1)) = $(F(l,l-1)) * Q_$(l)_$(l-1) * c_$(l-1)"))
      if grad
         push!(code, Meta.parse("dZ_$(lm2idx(l, l)) = $(F(l,l)) * Q_$(l)_$(l) * SA[ $l * c_$(l-1), - $l * s_$(l-1), zero(x) ]"))
         push!(code, Meta.parse("dZ_$(lm2idx(l,-l)) = $(F(l,l)) * Q_$(l)_$(l) * SA[ $l * s_$(l-1),  $l * c_$(l-1), zero(x) ]"))
         push!(code, Meta.parse("""dZ_$(lm2idx(l,-l+1)) = $(F(l,l-1)) * SA[ Q_$(l)_$(l-1) * $(l-1) * s_$(l-2),
                                                                       Q_$(l)_$(l-1) * $(l-1) * c_$(l-2),
                                                                       $(2*l-1) * Q_$(l-1)_$(l-1) * s_$(l-1) ]"""))
         push!(code, Meta.parse("""dZ_$(lm2idx(l, l-1)) = $(F(l,l-1)) * SA[ Q_$(l)_$(l-1) * $(l-1) * c_$(l-2),
                                                                      Q_$(l)_$(l-1) * $(-l+1) * s_$(l-2),
                                                                      $(2*l-1) * Q_$(l-1)_$(l-1) * c_$(l-1) ]"""))
      end

      # second recursion: m = l-2 down to 1  (recursion coeffs baked via lit)
      for m = l-2:-1:1
         push!(code, Meta.parse("Q_$(l)_$(m) = $(lit((2*l-1)/(l-m))) * z * Q_$(l-1)_$m - $(lit((l+m-1)/(l-m))) * r² * Q_$(l-2)_$(m)"))
         push!(code, Meta.parse("Z_$(lm2idx(l,-m)) = $(F(l,m)) * Q_$(l)_$(m) * s_$(m)"))
         push!(code, Meta.parse("Z_$(lm2idx(l, m)) = $(F(l,m)) * Q_$(l)_$(m) * c_$(m)"))
         if grad
            push!(code, Meta.parse("""dZ_$(lm2idx(l,-m)) = $(F(l,m)) * SA[ Q_$(l)_$(m) * $(m) * s_$(m-1) + x * Q_$(l-1)_$(m+1) * s_$(m),
                                                                     Q_$(l)_$(m) * $(m) * c_$(m-1) + y * Q_$(l-1)_$(m+1) * s_$(m),
                                                                     $(l+m) * Q_$(l-1)_$(m) * s_$m ]"""))
            push!(code, Meta.parse("""dZ_$(lm2idx(l, m)) = $(F(l,m)) * SA[ Q_$(l)_$(m) * $(m) * c_$(m-1) + x * Q_$(l-1)_$(m+1) * c_$(m),
                                                                    Q_$(l)_$(m) * $(-m) * s_$(m-1) + y * Q_$(l-1)_$(m+1) * c_$(m),
                                                                    $(l+m) * Q_$(l-1)_$(m) * c_$m ]"""))
         end
      end

      # m = 0
      push!(code, Meta.parse("Q_$(l)_0 = $(lit((2*l-1)/l)) * z * Q_$(l-1)_0 - $(lit((l-1)/l)) * r² * Q_$(l-2)_0"))
      push!(code, Meta.parse("Z_$(lm2idx(l,0)) = $(F(l,0)) * $irt2 * Q_$(l)_0"))
      if grad
         push!(code, Meta.parse("dZ_$(lm2idx(l, 0)) = $(F(l,0)) * $irt2 * SA[ Q_$(l-1)_1 * x, Q_$(l-1)_1 * y, $(l) * Q_$(l-1)_0 ]"))
      end
   end

   # finalisation (element type inferred from the args -> no `T` name needed)
   if grad
      push!(code, Meta.parse("Z = SVector{$len}(" * join( ["Z_$i, " for i = 1:len] ) * ")"))
      push!(code, Meta.parse("dZ = SVector{$len}(" * join( ["dZ_$i, " for i = 1:len] ) * ")"))
      push!(code, :(return Z, dZ))
   else
      push!(code, Meta.parse("return SVector{$len}(" * join( ["Z_$i, " for i = 1:len] ) * ")"))
   end

   return code
end


"""
`static_solid_harmonics`: evaluate the solid harmonics basis for a single input
point. The code is fully generated and unrolled; the return value is an
`SVector{LEN, T}`. The normalisation prefactors are read from `Flm` (e.g.
`basis.Flm`); a convenience method without `Flm` builds them from a normalisation
symbol.

Usage, e.g. for `L = 4`:
```julia
basis = SolidHarmonics(4)
𝐫 = @SVector randn(3)
Z = static_solid_harmonics(Val(4), 𝐫, basis.Flm)
Z = static_solid_harmonics(Val(4), 𝐫)                 # builds Flm for :L2
Z = static_solid_harmonics(Val(4), 𝐫, Val{:L2}())
```
"""
static_solid_harmonics(valL::Val{L}, 𝐫::SVector{3}, Flm::StaticMatrix) where {L} =
      static_solid_harmonics(valL, 𝐫[1], 𝐫[2], 𝐫[3], Flm)

@generated function static_solid_harmonics(::Val{L}, x::T, y::T, z::T,
                                           Flm::StaticMatrix) where {L, T}
   code = _codegen(L, T; grad = false)
   return quote
      $(Expr(:block, code...))
   end
end

# convenience: build Flm from a normalisation symbol (slower; for one-off use)
function static_solid_harmonics(valL::Val{L}, 𝐫::SVector{3, T},
                     valNorm::Val{NORM} = Val{:sphericart}()) where {L, T, NORM}
   Tc = (T <: Dual) ? ForwardDiff.valtype(T) : T
   Flm = SMatrix{L+1, L+1}(generate_Flms(L; normalisation = NORM, T = Tc))
   return static_solid_harmonics(valL, 𝐫, Flm)
end


static_solid_harmonics_with_grads(valL::Val{L}, 𝐫::SVector{3}, Flm::StaticMatrix
                                  ) where {L} =
      static_solid_harmonics_with_grads(valL, 𝐫[1], 𝐫[2], 𝐫[3], Flm)

@generated function static_solid_harmonics_with_grads(::Val{L}, x::T, y::T, z::T,
                                                      Flm::StaticMatrix) where {L, T}
   code = _codegen(L, T; grad = true)
   return quote
      $(Expr(:block, code...))
   end
end

function static_solid_harmonics_with_grads(valL::Val{L}, 𝐫::SVector{3, T},
                     valNorm::Val{NORM} = Val{:sphericart}()) where {L, T, NORM}
   Tc = (T <: Dual) ? ForwardDiff.valtype(T) : T
   Flm = SMatrix{L+1, L+1}(generate_Flms(L; normalisation = NORM, T = Tc))
   return static_solid_harmonics_with_grads(valL, 𝐫, Flm)
end
