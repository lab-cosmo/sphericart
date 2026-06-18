
#  Single batched evaluation path for all backends (CPU + every GPU) and all
#  element types (incl. Float128 etc.).
#
#  One KernelAbstractions kernel runs one point per work-item; on the CPU backend
#  it auto-multithreads across the `ndrange`. The kernel is an ordinary looped
#  `@kernel` (NOT `@generated`) so it works for element types that are only
#  loaded after this package (a `@generated` body would hit a world-age error,
#  e.g. with Quadmath's Float128).
#
#  The associated-Legendre table `Q` only ever needs rows `l, l-1, l-2`, so it is
#  stored in a *rolling* 3-row buffer of width `L+1` (column `_qcol(L,l,m)`),
#  instead of the full `(L+1)^2`. This keeps the group-shared memory at O(L)
#  rather than O(L^2) -- the previous full-table version exceeded the 48 KB
#  shared-memory limit on NVIDIA already at L≈15.
#
#  `Val{SH}` selects spherical (`SH = true`) vs solid (`SH = false`) harmonics:
#  for `SH` the input is rescaled to the unit sphere and the gradient is
#  projected onto the sphere afterwards.

using KernelAbstractions
using KernelAbstractions: @kernel, @index, @Const, @localmem, @uniform,
                          @groupsize, get_backend, synchronize
using StaticArrays: SMatrix, SVector, SA
using LinearAlgebra: dot


# column into the rolling 3-row Q buffer for (l, m), m = 0:l, 1-based.
@inline _qcol(L, l, m) = (l % 3) * (L + 1) + m + 1


"""
```
ka_solid_harmonics!(Z, dZ, ::Val{L}, [::Val{SH}], Rs, Flm, GRPSZ = 32)
```
KernelAbstractions launcher for the (solid or spherical) harmonics basis on a
batch of points. Works for any backend (CPU + GPU) inferred from `Z`, and any
element type.
* `Z, dZ` : output arrays; if `dZ === nothing` only `Z` is evaluated.
* `L`     : maximum degree.
* `SH`    : `true` for spherical, `false` for solid harmonics (default `false`).
* `Rs`    : input vector of `SVector{3}`.
* `Flm`   : normalisation prefactors (`basis.Flm`).
"""
function ka_solid_harmonics!(Z, dZ, ::Val{L}, ::Val{SH},
                  Rs::AbstractVector{<: SVector{3}}, Flm,
                  GRPSZ = 32) where {L, SH}
   nRs = length(Rs)
   len = sizeY(L)
   @assert size(Z, 1) >= nRs
   @assert size(Z, 2) >= len
   if !isnothing(dZ)
      @assert size(dZ, 1) >= nRs
      @assert size(dZ, 2) >= len
   end

   backend = get_backend(Z)   # assume same backend for dZ
   solidh_main! = _ka_solidh_main!(backend, (GRPSZ,))
   solidh_main!(Z, dZ, Val{L}(), Val{SH}(), Rs, Flm; ndrange = (nRs,))
   synchronize(backend)

   return nothing
end

# convenience: solid harmonics (SH = false)
ka_solid_harmonics!(Z, dZ, ::Val{L},
                    Rs::AbstractVector{<: SVector{3}}, Flm, GRPSZ = 32) where {L} =
   ka_solid_harmonics!(Z, dZ, Val{L}(), Val{false}(), Rs, Flm, GRPSZ)

# convenience: values only (no gradients), solid harmonics
ka_solid_harmonics!(Z, ::Val{L},
                    Rs::AbstractVector{<: SVector{3}}, Flm, GRPSZ = 32) where {L} =
   ka_solid_harmonics!(Z, nothing, Val{L}(), Val{false}(), Rs, Flm, GRPSZ)

# alias kept for backwards compatibility
ka_solid_harmonics_with_grad!(Z, dZ, ::Val{L},
                    Rs::AbstractVector{<: SVector{3}}, Flm, GRPSZ = 32) where {L} =
   ka_solid_harmonics!(Z, dZ, Val{L}(), Val{false}(), Rs, Flm, GRPSZ)


@kernel function _ka_solidh_main!(Z, dZ, ::Val{L}, ::Val{SH},
                                  @Const(Rs), @Const(Flm)) where {L, SH}
   j  = @index(Global, Linear)
   jl = @index(Local, Linear)

   @uniform len_grp = prod(@groupsize())
   @uniform WITHGRAD = !isnothing(dZ)
   @uniform T = eltype(Z)
   @uniform rt2 = sqrt(T(2))
   @uniform _1 = one(T)
   @uniform _0 = zero(T)

   # group-shared scratch
   x  = @localmem T (len_grp,)
   y  = @localmem T (len_grp,)
   z  = @localmem T (len_grp,)
   r² = @localmem T (len_grp,)
   r  = @localmem T (len_grp,)
   s  = @localmem T (len_grp, L+1)
   c  = @localmem T (len_grp, L+1)
   Q  = @localmem T (len_grp, 3*(L+1))   # rolling 3-row buffer (rows l, l-1, l-2)

   # ---- load coordinates
   x[jl], y[jl], z[jl] = Rs[j].data
   r²[jl] = x[jl]*x[jl] + y[jl]*y[jl] + z[jl]*z[jl]

   # ---- normalise the input onto the unit sphere for spherical harmonics
   if SH
      r[jl] = sqrt(r²[jl])
      x[jl] /= r[jl]
      y[jl] /= r[jl]
      z[jl] /= r[jl]
      r²[jl] = _1
   end

   # ---- sines / cosines:  s_0 = 0, c_0 = 1; recurse m -> m+1
   s[jl, 1] = _0
   c[jl, 1] = _1
   @inbounds for m = 1:L
      s[jl, m+1] = s[jl, m] * x[jl] + c[jl, m] * y[jl]
      c[jl, m+1] = c[jl, m] * x[jl] - s[jl, m] * y[jl]
   end

   # ---- l = 0
   i00 = lm2idx(0, 0)
   Q[jl, _qcol(L, 0, 0)] = _1
   Z[j, i00] = (Flm[1,1]/rt2)
   if WITHGRAD
      dZ[j, i00] = zero(SVector{3, T})
   end

   # ---- l = 1 (treated separately; generic formulas below assume l >= 2)
   if L >= 1
      F_1_1 = Flm[2, 2]
      F_1_0 = Flm[2, 1]
      # Q_1^1, Z_1^{±1}
      Q[jl, _qcol(L, 1, 1)] = - _1
      Z[j, lm2idx(1,  1)] = - F_1_1 * c[jl, 2]
      Z[j, lm2idx(1, -1)] = - F_1_1 * s[jl, 2]
      # Q_1^0, Z_1^0
      Q_j_10 = z[jl]
      Q[jl, _qcol(L, 1, 0)] = Q_j_10
      Z[j, lm2idx(1, 0)] = F_1_0 * Q_j_10 * c[jl, 1] / rt2
      if WITHGRAD
         dZ[j, lm2idx(1,  1)] = SA[- F_1_1, _0, _0]
         dZ[j, lm2idx(1, -1)] = SA[_0, - F_1_1, _0]
         dZ[j, lm2idx(1,  0)] = SA[_0, _0, F_1_0 / rt2]
      end
   end

   @inbounds for l = 2:L
      F_l_l   = Flm[l+1, l+1]
      F_l_l⁻¹ = Flm[l+1, l]

      # m = l
      Q_j_l⁻¹l⁻¹ = Q[jl, _qcol(L, l-1, l-1)]
      Q_j_ll = - (2*l-1) * Q_j_l⁻¹l⁻¹
      Q[jl, _qcol(L, l, l)] = Q_j_ll
      Z[j, lm2idx(l,  l)] = F_l_l * Q_j_ll * c[jl, l+1]
      Z[j, lm2idx(l, -l)] = F_l_l * Q_j_ll * s[jl, l+1]

      # m = l-1
      Q_j_ll⁻¹ = (2*l-1) * z[jl] * Q_j_l⁻¹l⁻¹
      Q[jl, _qcol(L, l, l-1)] = Q_j_ll⁻¹
      Z[j, lm2idx(l, -l+1)] = F_l_l⁻¹ * Q_j_ll⁻¹ * s[jl, l]
      Z[j, lm2idx(l,  l-1)] = F_l_l⁻¹ * Q_j_ll⁻¹ * c[jl, l]

      if WITHGRAD
         dZ[j, lm2idx(l,  l)] = F_l_l * Q_j_ll * SA[l * c[jl, l], -l * s[jl, l], _0]
         dZ[j, lm2idx(l, -l)] = F_l_l * Q_j_ll * SA[l * s[jl, l],  l * c[jl, l], _0]
         dZ[j, lm2idx(l, -l+1)] = F_l_l⁻¹ * SA[Q_j_ll⁻¹ * (l-1) * s[jl, l-1],
                                               Q_j_ll⁻¹ * (l-1) * c[jl, l-1],
                                               (2*l-1) * Q_j_l⁻¹l⁻¹ * s[jl, l] ]
         dZ[j, lm2idx(l,  l-1)] = F_l_l⁻¹ * SA[Q_j_ll⁻¹ * (l-1)  * c[jl, l-1],
                                               Q_j_ll⁻¹ * (-l+1) * s[jl, l-1],
                                               (2*l-1) * Q_j_l⁻¹l⁻¹ * c[jl, l] ]
      end

      # second recursion: m = l-2 down to 1
      for m = l-2:-1:1
         F_l_m = Flm[l+1, m+1]
         cj = c[jl, m+1]; sj = s[jl, m+1]
         Q_lm = ((2*l-1) * z[jl] * Q[jl, _qcol(L, l-1, m)]
                 - (l+m-1) * r²[jl] * Q[jl, _qcol(L, l-2, m)]) / (l-m)
         Q[jl, _qcol(L, l, m)] = Q_lm
         Z[j, lm2idx(l, -m)] = F_l_m * Q_lm * sj
         Z[j, lm2idx(l,  m)] = F_l_m * Q_lm * cj
         if WITHGRAD
            Q_lm_x = x[jl] * Q[jl, _qcol(L, l-1, m+1)]
            Q_lm_y = y[jl] * Q[jl, _qcol(L, l-1, m+1)]
            Q_lm_z = (l+m) * Q[jl, _qcol(L, l-1, m)]
            s_x =  m * s[jl, m]; s_y =  m * c[jl, m]
            c_x =  m * c[jl, m]; c_y = -m * s[jl, m]
            dZ[j, lm2idx(l, -m)] = F_l_m * SA[Q_lm * s_x + Q_lm_x * sj,
                                              Q_lm * s_y + Q_lm_y * sj,
                                                           Q_lm_z * sj ]
            dZ[j, lm2idx(l,  m)] = F_l_m * SA[Q_lm * c_x + Q_lm_x * cj,
                                              Q_lm * c_y + Q_lm_y * cj,
                                                           Q_lm_z * cj ]
         end
      end

      # m = 0
      F_l_0_f = Flm[l+1, 1] / rt2
      cj = c[jl, 1]
      Q_l0 = ((2*l-1) * z[jl] * Q[jl, _qcol(L, l-1, 0)]
              - (l-1) * r²[jl] * Q[jl, _qcol(L, l-2, 0)]) / l
      Q[jl, _qcol(L, l, 0)] = Q_l0
      Z[j, lm2idx(l, 0)] = F_l_0_f * Q_l0 * cj
      if WITHGRAD
         Q_l0_x = x[jl] * Q[jl, _qcol(L, l-1, 1)]
         Q_l0_y = y[jl] * Q[jl, _qcol(L, l-1, 1)]
         Q_l0_z = l * Q[jl, _qcol(L, l-1, 0)]
         dZ[j, lm2idx(l, 0)] = F_l_0_f * cj * SA[Q_l0_x, Q_l0_y, Q_l0_z]
      end
   end

   # ---- spherical harmonics: project the gradient onto the unit sphere
   if SH && WITHGRAD
      @inbounds for α = 1:size(dZ, 2)
         dzj = dZ[j, α] / r[jl]
         𝐫̂j = SA[x[jl], y[jl], z[jl]]
         dZ[j, α] = dzj - dot(𝐫̂j, dzj) * 𝐫̂j
      end
   end

   nothing
end
