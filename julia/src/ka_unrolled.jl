
#
# Unrolled / @generated per-thread recursion for the batched KA kernel.
#
# This is an EXPLORATION of whether fully unrolling the per-thread solid
# harmonics recursion for small L (Z and dZ stored directly into the output
# arrays, "store sink") gains performance over the looped KA kernel in
# `ka_kernels.jl`.
#
# CRITICAL DESIGN CONSTRAINT (world-age safety inside KA @kernel + @generated):
# The code GENERATOR must never call a method of the element type `T`.
#   - numeric constants are baked via `oftype(x, c)` using the input
#     coordinate `x` as a type witness (NOT `T(c)` and NOT `$T`);
#   - prefactors are READ at runtime from the passed `Flm` array as
#     `Flm[i, j]` (NOT baked as literal Float values);
#   - `zero(x)` / `one(x)` are used where a 0/1 of the right type is needed.
# This is exactly the pattern needed so the same generated kernel works for
# Float32, Float64 AND Quadmath.Float128 (proving world-age safety) and on the
# CUDA backend.
#
# The per-thread Q values are kept as scalar locals (registers) -- there is NO
# @localmem Q buffer, which is a potential GPU win (no shared memory traffic).
#

using KernelAbstractions, GPUArraysCore
using StaticArrays: SA, SVector
using LinearAlgebra: dot

# threshold: L <= UNROLL_LMAX uses the unrolled kernel, larger L falls back to
# the looped kernel in ka_kernels.jl. Easy to change.
const UNROLL_LMAX = 6

# helpers -- symbols for the unrolled scalar locals
_Qsym(l, m) = Symbol("Q_$(l)_$(m)")
_ssym(m) = Symbol("s_$(m)")
_csym(m) = Symbol("c_$(m)")

# bake a numeric constant of the coordinate type via the type witness `x`
_oft(c) = :( oftype(x, $(c)) )


# Build the unrolled store-sink body as a Vector{Expr}.
#  - `x, y, z`     : the (already normalised, if SH) coordinates (locals)
#  - `r²`          : x²+y²+z² (local)
#  - `Flm`         : runtime array of prefactors, read as Flm[i,j]
#  - `Z`, `dZ`     : output arrays, written as Z[j,idx] / dZ[j,idx]
#  - `j`           : the global index local variable
# `grad` toggles gradient emission. Constants baked via oftype(x, c).
function _codegen_store(L; grad::Bool)
   code = Expr[]

   push!(code, :( r² = x*x + y*y + z*z ))

   # c_m and s_m
   push!(code, :( $(_ssym(0)) = zero(x) ))
   push!(code, :( $(_csym(0)) = one(x) ))
   for m = 1:L
      push!(code, :( $(_ssym(m)) = $(_ssym(m-1)) * x + $(_csym(m-1)) * y ))
      push!(code, :( $(_csym(m)) = $(_csym(m-1)) * x - $(_ssym(m-1)) * y ))
   end

   push!(code, :( rt2 = sqrt(oftype(x, 2)) ))

   # ---- l = 0 ----
   push!(code, :( $(_Qsym(0,0)) = one(x) ))
   push!(code, :( Z[j, $(lm2idx(0,0))] = (Flm[1,1] / rt2) * $(_Qsym(0,0)) ))
   if grad
      push!(code, :( dZ[j, $(lm2idx(0,0))] = zero(SVector{3, eltype(Z)}) ))
   end

   if L >= 1
      # ---- l = 1 special case ----
      push!(code, :( F_1_1 = Flm[2, 2] ))   # Flm[1+1, 1+1]
      push!(code, :( F_1_0 = Flm[2, 1] ))   # Flm[1+1, 1+0]

      push!(code, :( $(_Qsym(1,1)) = - $(_Qsym(0,0)) ))
      push!(code, :( Z[j, $(lm2idx(1, 1))] = - F_1_1 * $(_csym(1)) ))
      push!(code, :( Z[j, $(lm2idx(1,-1))] = - F_1_1 * $(_ssym(1)) ))
      push!(code, :( $(_Qsym(1,0)) = z ))
      push!(code, :( Z[j, $(lm2idx(1, 0))] = (F_1_0 / rt2) * $(_Qsym(1,0)) * $(_csym(0)) ))

      if grad
         push!(code, :( dZ[j, $(lm2idx(1, 1))] = SA[ -F_1_1, zero(x), zero(x) ] ))
         push!(code, :( dZ[j, $(lm2idx(1,-1))] = SA[ zero(x), -F_1_1, zero(x) ] ))
         push!(code, :( dZ[j, $(lm2idx(1, 0))] = SA[ zero(x), zero(x), F_1_0 / rt2 ] ))
      end
   end

   for l = 2:L
      Fll   = Symbol("F_$(l)_$(l)")
      Fll1  = Symbol("F_$(l)_$(l-1)")
      push!(code, :( $Fll  = Flm[$(l+1), $(l+1)] ))
      push!(code, :( $Fll1 = Flm[$(l+1), $(l)]   ))

      Qll   = _Qsym(l, l)
      Qll1  = _Qsym(l, l-1)
      Ql1l1 = _Qsym(l-1, l-1)

      # Q_l^l = -(2l-1) Q_{l-1}^{l-1}  => Y_l^l, Y_l^-l
      push!(code, :( $Qll = - $(_oft(2*l-1)) * $Ql1l1 ))
      push!(code, :( Z[j, $(lm2idx(l, l))] = $Fll * $Qll * $(_csym(l)) ))
      push!(code, :( Z[j, $(lm2idx(l,-l))] = $Fll * $Qll * $(_ssym(l)) ))

      # Q_l^{l-1} = (2l-1) z Q_{l-1}^{l-1}  => Y_l^{l-1}, Y_l^{-l+1}
      push!(code, :( $Qll1 = $(_oft(2*l-1)) * z * $Ql1l1 ))
      push!(code, :( Z[j, $(lm2idx(l,-l+1))] = $Fll1 * $Qll1 * $(_ssym(l-1)) ))
      push!(code, :( Z[j, $(lm2idx(l, l-1))] = $Fll1 * $Qll1 * $(_csym(l-1)) ))

      if grad
         # l = m : Q_l^l const in x,y
         push!(code, :( dZ[j, $(lm2idx(l, l))] = $Fll * $Qll *
                        SA[ $(_oft(l)) * $(_csym(l-1)), - $(_oft(l)) * $(_ssym(l-1)), zero(x) ] ))
         push!(code, :( dZ[j, $(lm2idx(l,-l))] = $Fll * $Qll *
                        SA[ $(_oft(l)) * $(_ssym(l-1)),  $(_oft(l)) * $(_csym(l-1)), zero(x) ] ))
         # m = l-1
         push!(code, :( dZ[j, $(lm2idx(l,-l+1))] = $Fll1 *
                        SA[ $Qll1 * $(_oft(l-1)) * $(_ssym(l-2)),
                            $Qll1 * $(_oft(l-1)) * $(_csym(l-2)),
                            $(_oft(2*l-1)) * $Ql1l1 * $(_ssym(l-1)) ] ))
         push!(code, :( dZ[j, $(lm2idx(l, l-1))] = $Fll1 *
                        SA[ $Qll1 * $(_oft(l-1)) * $(_csym(l-2)),
                            $Qll1 * $(_oft(-l+1)) * $(_ssym(l-2)),
                            $(_oft(2*l-1)) * $Ql1l1 * $(_csym(l-1)) ] ))
      end

      # second recursion, m = l-2 down to 1
      for m = l-2:-1:1
         Flm_sym = Symbol("F_$(l)_$(m)")
         push!(code, :( $Flm_sym = Flm[$(l+1), $(m+1)] ))
         Qlm   = _Qsym(l, m)
         Ql1m  = _Qsym(l-1, m)
         Ql2m  = _Qsym(l-2, m)
         Ql1m1 = _Qsym(l-1, m+1)
         push!(code, :( $Qlm = $(_oft((2*l-1)//(l-m))) * z * $Ql1m
                              - $(_oft((l+m-1)//(l-m))) * r² * $Ql2m ))
         push!(code, :( Z[j, $(lm2idx(l,-m))] = $Flm_sym * $Qlm * $(_ssym(m)) ))
         push!(code, :( Z[j, $(lm2idx(l, m))] = $Flm_sym * $Qlm * $(_csym(m)) ))

         if grad
            push!(code, :( dZ[j, $(lm2idx(l,-m))] = $Flm_sym *
                  SA[ $Qlm * $(_oft(m)) * $(_ssym(m-1)) + x * $Ql1m1 * $(_ssym(m)),
                      $Qlm * $(_oft(m)) * $(_csym(m-1)) + y * $Ql1m1 * $(_ssym(m)),
                      $(_oft(l+m)) * $Ql1m * $(_ssym(m)) ] ))
            push!(code, :( dZ[j, $(lm2idx(l, m))] = $Flm_sym *
                  SA[ $Qlm * $(_oft(m)) * $(_csym(m-1)) + x * $Ql1m1 * $(_csym(m)),
                      $Qlm * $(_oft(-m)) * $(_ssym(m-1)) + y * $Ql1m1 * $(_csym(m)),
                      $(_oft(l+m)) * $Ql1m * $(_csym(m)) ] ))
         end
      end

      # special case m = 0
      Fl0_sym = Symbol("F_$(l)_0")
      push!(code, :( $Fl0_sym = Flm[$(l+1), 1] / rt2 ))
      Ql0  = _Qsym(l, 0)
      Ql10 = _Qsym(l-1, 0)
      Ql20 = _Qsym(l-2, 0)
      Ql11 = _Qsym(l-1, 1)
      push!(code, :( $Ql0 = $(_oft((2*l-1)//l)) * z * $Ql10
                          - $(_oft((l-1)//l)) * r² * $Ql20 ))
      push!(code, :( Z[j, $(lm2idx(l, 0))] = $Fl0_sym * $Ql0 ))
      if grad
         # cj = c_0 = 1 (m=0). dZ = Fl0 * SA[Q_{l-1}^1 x, Q_{l-1}^1 y, l Q_{l-1}^0]
         push!(code, :( dZ[j, $(lm2idx(l, 0))] = $Fl0_sym *
               SA[ $Ql11 * x, $Ql11 * y, $(_oft(l)) * $Ql10 ] ))
      end
   end

   return code
end


#
# The @generated unrolled kernel. Mirrors the SH semantics of the looped
# kernel exactly:
#  - load x,y,z for thread j
#  - if SH: normalise onto unit sphere, remember r
#  - run the unrolled store-sink body
#  - if SH && grad: project each gradient onto the sphere
#
@kernel function _ka_solidh_unrolled!(
               Z, dZ, ::Val{L}, ::Val{SH},
               @Const(Rs),
               @Const(Flm), ) where {L, SH}

   j = @index(Global, Linear)
   _ka_unrolled_body!(Z, dZ, Val{L}(), Val{SH}(), Rs, Flm, j)
end

# function barrier so the @generated body sees concrete types and we can
# special-case grad vs no-grad via dispatch on dZ.
@generated function _ka_unrolled_body!(Z, dZ::Nothing, ::Val{L}, ::Val{SH},
                              Rs, Flm, j) where {L, SH}
   body = _codegen_store(L; grad = false)
   pre = quote
      𝐫 = Rs[j]
      x = 𝐫[1]; y = 𝐫[2]; z = 𝐫[3]
   end
   if SH
      pre = quote
         $pre
         r = sqrt(x*x + y*y + z*z)
         x = x / r; y = y / r; z = z / r
      end
   end
   return quote
      $pre
      $(Expr(:block, body...))
      nothing
   end
end

@generated function _ka_unrolled_body!(Z, dZ, ::Val{L}, ::Val{SH},
                              Rs, Flm, j) where {L, SH}
   body = _codegen_store(L; grad = true)
   pre = quote
      𝐫 = Rs[j]
      x = 𝐫[1]; y = 𝐫[2]; z = 𝐫[3]
   end
   if SH
      pre = quote
         $pre
         r = sqrt(x*x + y*y + z*z)
         x = x / r; y = y / r; z = z / r
      end
   end
   # gradient projection for spherical harmonics
   post = SH ? quote
      r̂ = SA[x, y, z]
      @inbounds for i = 1:$(sizeY(L))
         dz = dZ[j, i] / r
         dZ[j, i] = dz - dot(r̂, dz) * r̂
      end
   end : :( nothing )
   return quote
      $pre
      $(Expr(:block, body...))
      $post
      nothing
   end
end
