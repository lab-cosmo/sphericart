
#  Single source of truth for the solid-harmonics recursion (values + gradients).
#
#  `_codegen` emits the straight-line recursion once; the only thing that varies
#  is the *sink*, i.e. where each output Zlm / ∇Zlm is written:
#    - `sink = :svector` : assign to scalars `Z_<i>` / `dZ_<i>` and collect into an
#                          `SVector` at the end  (used by the single-point kernels)
#    - `sink = :store`   : stream straight into `Z[j, <i>]` / `dZ[j, <i>]`
#                          (used by the KA kernel; one thread per point `j`)
#  In both cases the intermediates (`c_m`, `s_m`, `Q_l_m`) stay scalar => registers.
#
#  Gradient identities used (cf. Eq. (10) in the sphericart publication):
#    ∂x c^m = m c^{m-1},  ∂y c^m = -m s^{m-1},  ∂x s^m = m s^{m-1},  ∂y s^m = m c^{m-1}
#    ∂x Q_l^m = x Q_{l-1}^{m+1},  ∂y Q_l^m = y Q_{l-1}^{m+1},  ∂z Q_l^m = (l+m) Q_{l-1}^m

import ForwardDiff
using ForwardDiff: Dual

"""
`_codegen(L, T, normalisation; grad, sink, jvar)`: build the list of expressions
that evaluate the solid harmonics basis (and, if `grad`, its gradients) up to
degree `L`, with element type `T`. See the comment block at the top of this file
for the meaning of `sink`. Returns a `Vector{Expr}`.
"""
function _codegen(L, T, normalisation; grad::Bool, sink::Symbol, jvar = :j)
   # `Tc` is the real type in which the constant prefactors are baked. For a
   # ForwardDiff.Dual input we bake them in the underlying value type (a `Dual`
   # times a real constant is again a `Dual`); the *containers* still use `T`.
   Tc = (T <: Dual) ? ForwardDiff.valtype(T) : T
   Flm = generate_Flms(L; normalisation = normalisation, T = Tc)
   len = sizeY(L)
   rt2 = sqrt(Tc(2))

   # emit a numeric constant as a `Tc`-typed literal. Interpolating a Float32
   # through a string drops its type (Meta.parse reads it back as Float64),
   # which would silently promote the whole evaluation to Float64 -- fatal on
   # GPUs without Float64 (e.g. Metal). Wrapping in `Tc(...)` keeps it in `Tc`.
   lit(v) = "$(Tc)($(Float64(v)))"

   # sink-dependent left-hand sides for the outputs (anonymous bindings, so that
   # the two branches don't register conflicting methods during precompilation)
   if sink == :svector
      Zlhs  = i -> "Z_$i"
      dZlhs = i -> "dZ_$i"
   elseif sink == :store
      Zlhs  = i -> "Z[$jvar, $i]"
      dZlhs = i -> "dZ[$jvar, $i]"
   else
      error("`_codegen`: unknown sink = $sink")
   end

   code = Expr[]
   push!(code, :(r² = x^2 + y^2 + z^2))

   # c_m and s_m  (note c_0 stays = 1; m = 0 is special-cased with an explicit /√2)
   push!(code, :(s_0 = zero($T)))
   push!(code, :(c_0 = one($T)))
   for m = 1:L
      push!(code, Meta.parse("s_$m = s_$(m-1) * x + c_$(m-1) * y"))
      push!(code, Meta.parse("c_$m = c_$(m-1) * x - s_$(m-1) * y"))
   end

   # l = 0 :  Q_0^0 and Z_0^0
   push!(code, Meta.parse("Q_0_0 = one($T)"))
   push!(code, Meta.parse("$(Zlhs(1)) = $(lit(Flm[1,1]/rt2)) * Q_0_0"))
   if grad
      push!(code, Meta.parse("$(dZlhs(1)) = zero(SVector{3, $T})"))
   end

   # l = 1 : special case (generic formulas below assume l >= 2)
   if L >= 1
      push!(code, Meta.parse("Q_1_1 = - Q_0_0"))
      push!(code, Meta.parse("$(Zlhs(lm2idx(1, 1)))  = $(lit(-Flm[2,2])) * c_1"))
      push!(code, Meta.parse("$(Zlhs(lm2idx(1,-1)))  = $(lit(-Flm[2,2])) * s_1"))
      push!(code, Meta.parse("Q_1_0 = z"))
      push!(code, Meta.parse("$(Zlhs(lm2idx(1, 0)))  = $(lit(Flm[2,1]/rt2)) * Q_1_0 * c_0"))
      if grad
         push!(code, Meta.parse("$(dZlhs(lm2idx(1, 1))) = SA[ $(lit(-Flm[2,2])), zero($T), zero($T) ]"))
         push!(code, Meta.parse("$(dZlhs(lm2idx(1,-1))) = SA[ zero($T), $(lit(-Flm[2,2])), zero($T) ]"))
         push!(code, Meta.parse("$(dZlhs(lm2idx(1, 0))) = SA[ zero($T), zero($T), $(lit(Flm[2,1]/rt2)) ]"))
      end
   end

   for l = 2:L
      # m = l
      push!(code, Meta.parse("Q_$(l)_$(l) = - $(2*l-1) * Q_$(l-1)_$(l-1)"))
      push!(code, Meta.parse("$(Zlhs(lm2idx(l, l)))  = $(lit(Flm[1+l,1+l])) * Q_$(l)_$(l) * c_$(l)"))
      push!(code, Meta.parse("$(Zlhs(lm2idx(l,-l)))  = $(lit(Flm[1+l,1+l])) * Q_$(l)_$(l) * s_$(l)"))
      # m = l-1
      push!(code, Meta.parse("Q_$(l)_$(l-1) = $(2*l-1) * z * Q_$(l-1)_$(l-1)"))
      push!(code, Meta.parse("$(Zlhs(lm2idx(l,-l+1))) = $(lit(Flm[1+l,1+l-1])) * Q_$(l)_$(l-1) * s_$(l-1)"))
      push!(code, Meta.parse("$(Zlhs(lm2idx(l, l-1))) = $(lit(Flm[1+l,1+l-1])) * Q_$(l)_$(l-1) * c_$(l-1)"))
      if grad
         push!(code, Meta.parse("$(dZlhs(lm2idx(l, l))) = $(lit(Flm[1+l,1+l])) * Q_$(l)_$(l) * SA[ $l * c_$(l-1), - $l * s_$(l-1), zero($T) ]"))
         push!(code, Meta.parse("$(dZlhs(lm2idx(l,-l))) = $(lit(Flm[1+l,1+l])) * Q_$(l)_$(l) * SA[ $l * s_$(l-1),  $l * c_$(l-1), zero($T) ]"))
         push!(code, Meta.parse("""$(dZlhs(lm2idx(l,-l+1))) = $(lit(Flm[1+l,1+l-1])) * SA[ Q_$(l)_$(l-1) * $(l-1) * s_$(l-2),
                                                                                        Q_$(l)_$(l-1) * $(l-1) * c_$(l-2),
                                                                                        $(2*l-1) * Q_$(l-1)_$(l-1) * s_$(l-1) ]"""))
         push!(code, Meta.parse("""$(dZlhs(lm2idx(l, l-1))) = $(lit(Flm[1+l,1+l-1])) * SA[ Q_$(l)_$(l-1) * $(l-1) * c_$(l-2),
                                                                                       Q_$(l)_$(l-1) * $(-l+1) * s_$(l-2),
                                                                                       $(2*l-1) * Q_$(l-1)_$(l-1) * c_$(l-1) ]"""))
      end

      # second recursion: m = l-2 down to 1
      for m = l-2:-1:1
         push!(code, Meta.parse("Q_$(l)_$(m) = $(lit((2*l-1)/(l-m))) * z * Q_$(l-1)_$m - $(lit((l+m-1)/(l-m))) * r² * Q_$(l-2)_$(m)"))
         push!(code, Meta.parse("$(Zlhs(lm2idx(l,-m))) = $(lit(Flm[1+l,1+m])) * Q_$(l)_$(m) * s_$(m)"))
         push!(code, Meta.parse("$(Zlhs(lm2idx(l, m))) = $(lit(Flm[1+l,1+m])) * Q_$(l)_$(m) * c_$(m)"))
         if grad
            push!(code, Meta.parse("""$(dZlhs(lm2idx(l,-m))) = $(lit(Flm[1+l,1+m])) * SA[ Q_$(l)_$(m) * $(m) * s_$(m-1) + x * Q_$(l-1)_$(m+1) * s_$(m),
                                                                                     Q_$(l)_$(m) * $(m) * c_$(m-1) + y * Q_$(l-1)_$(m+1) * s_$(m),
                                                                                     $(l+m) * Q_$(l-1)_$(m) * s_$m ]"""))
            push!(code, Meta.parse("""$(dZlhs(lm2idx(l, m))) = $(lit(Flm[1+l,1+m])) * SA[ Q_$(l)_$(m) * $(m) * c_$(m-1) + x * Q_$(l-1)_$(m+1) * c_$(m),
                                                                                    Q_$(l)_$(m) * $(-m) * s_$(m-1) + y * Q_$(l-1)_$(m+1) * c_$(m),
                                                                                    $(l+m) * Q_$(l-1)_$(m) * c_$m ]"""))
         end
      end

      # special case m = 0
      push!(code, Meta.parse("Q_$(l)_0 = $(lit((2*l-1)/l)) * z * Q_$(l-1)_0 - $(lit((l-1)/l)) * r² * Q_$(l-2)_0"))
      push!(code, Meta.parse("$(Zlhs(lm2idx(l,0))) = $(lit(Flm[1+l,1+0]/rt2)) * Q_$(l)_0"))
      if grad
         push!(code, Meta.parse("$(dZlhs(lm2idx(l, 0))) = $(lit(Flm[1+l,1+0]/rt2)) * SA[ Q_$(l-1)_1 * x, Q_$(l-1)_1 * y, $(l) * Q_$(l-1)_0 ]"))
      end
   end

   # finalisation for the :svector sink (the :store sink writes in place)
   if sink == :svector
      if grad
         push!(code, Meta.parse("Z = SVector{$len, $T}(" * join( ["Z_$i, " for i = 1:len] ) * ")"))
         push!(code, Meta.parse("dZ = SVector{$len, SVector{3, $T}}(" * join( ["dZ_$i, " for i = 1:len] ) * ")"))
         push!(code, :(return Z, dZ))
      else
         push!(code, Meta.parse("return SVector{$len, $T}(" * join( ["Z_$i, " for i = 1:len] ) * ")"))
      end
   end

   return code
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
   code = _codegen(L, T, NORM; grad = false, sink = :svector)
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
   code = _codegen(L, T, NORM; grad = true, sink = :svector)
   return quote
      $(Expr(:block, code...))
   end
end
