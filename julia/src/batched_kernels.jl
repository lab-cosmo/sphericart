
# SIMD vectorized computational kernel for moderately many inputs. 
# (for MANY inputs we should in addition multi-thread it)
# 

function solid_harmonics!(Z::AbstractMatrix, ::Val{L}, 
                          Rs::AbstractVector{SVector{3, T}}, 
                          temps::NamedTuple 
                           ) where {L, T}
   nX = length(Rs)
   len = sizeY(L)

   x = temps.x
   y = temps.y
   z = temps.z
   r² = temps.r²
   s = temps.s
   c = temps.c
   Q = temps.Q
   Flm = temps.Flm

   # size checks to make sure the inbounds macro can be used safely. 
   @assert length(y) == length(z) == nX 
   @assert length(r²) >= nX
   @assert size(Z, 1) >= nX && size(s, 1) >= nX  && size(c, 1) >= nX && size(Q, 1) >= nX
   @assert size(Z, 2) >= len && size(Q, 2) >= len 
   @assert size(s, 2) >= L+1 && size(c, 2) >= L+1

   rt2 = sqrt(T(2)) 
   
   @inbounds @simd ivdep for j = 1:nX
      𝐫 = Rs[j] 
      xj, yj, zj = 𝐫[1], 𝐫[2], 𝐫[3]
      x[j] = xj
      y[j] = yj
      z[j] = zj
      r²[j] = xj^2 + yj^2 + zj^2
      # c_m and s_m, m = 0 
      s[j, 1] = zero(T)    # 0 -> 1
      c[j, 1] = one(T)     # 0 -> 1
   end

   # c_m and s_m continued 
   @inbounds for m = 1:L 
      @simd ivdep for j = 1:nX
         # m -> m+1 and  m-1 -> m
         s[j, m+1] = s[j, m] * x[j] + c[j, m] * y[j]
         c[j, m+1] = c[j, m] * x[j] - s[j, m] * y[j]
      end
   end

   # change c[0] to 1/rt2 to avoid a special case l-1=m=0 later 
   i00 = lm2idx(0, 0)

   @inbounds @simd ivdep for j = 1:nX
      c[j, 1] = one(T)/rt2

      # fill Q_0^0 and Z_0^0 
      Q[j, i00] = one(T)
      Z[j, i00] = (Flm[0,0]/rt2) * Q[j, i00]
   end

   @inbounds for l = 1:L 
      ill = lm2idx(l, l)
      il⁻l = lm2idx(l, -l)
      ill⁻¹ = lm2idx(l, l-1)
      il⁻¹l⁻¹ = lm2idx(l-1, l-1)
      il⁻l⁺¹ = lm2idx(l, -l+1)
      F_l_l = Flm[l,l]
      F_l_l⁻¹ = Flm[l,l-1]
      @simd ivdep for j = 1:nX 
         # Q_l^l and Y_l^l
         # m = l 
         Q[j, ill]   = - (2*l-1) * Q[j, il⁻¹l⁻¹]
         Z[j, ill]   = F_l_l * Q[j, ill] * c[j, l+1]  # l -> l+1
         Z[j, il⁻l] = F_l_l * Q[j, ill] * s[j, l+1]  # l -> l+1
         # Q_l^l-1 and Y_l^l-1
         # m = l-1 
         Q[j, ill⁻¹]  = (2*l-1) * z[j] * Q[j, il⁻¹l⁻¹]
         Z[j, il⁻l⁺¹] = F_l_l⁻¹ * Q[j, ill⁻¹] * s[j, l]  # l-1 -> l
         Z[j, ill⁻¹]  = F_l_l⁻¹ * Q[j, ill⁻¹] * c[j, l]  # l-1 -> l
         # overwrite if m = 0 -> ok 
      end

      # now we can go to the second recursion 
      for m = l-2:-1:0 
         ilm = lm2idx(l, m)
         il⁻m = lm2idx(l, -m)
         il⁻¹m = lm2idx(l-1, m)
         il⁻²m = lm2idx(l-2, m)
         F_l_m = Flm[l,m]
         @simd ivdep for j = 1:nX 
            Q[j, ilm] = ((2*l-1) * z[j] * Q[j, il⁻¹m] - (l+m-1) * r²[j] * Q[j, il⁻²m]) / (l-m)
            Z[j, il⁻m] = F_l_m * Q[j, ilm] * s[j, m+1]   # m -> m+1
            Z[j, ilm] = F_l_m * Q[j, ilm] * c[j, m+1]    # m -> m+1
         end
      end
   end

   return Z 
end



# notes on gradients - identities this code uses 
# 
#  ∂x c^m = m c^{m-1}  
#  ∂y c^m = - m s^{m-1}
#  ∂x s^m = m s^{m-1}
#  ∂y s^m = m c^{m-1}
#
#  ∂x Q_l^m = x Q_{l-1}^{m+1} 
#  ∂y Q_l^m = y Q_{l-1}^{m+1} 
#  ∂z Q_l^m = (l+m) Q_{l-1}^m
#
# cf. Eq. (10) in the sphericart publication.
# these identities are the reason we don't need additional 
# temporary arrays 

function solid_harmonics_with_grad!(
               Z::AbstractMatrix, 
               dZ::AbstractMatrix,
               ::Val{L}, 
               Rs::AbstractVector{SVector{3, T}}, 
               temps::NamedTuple 
               ) where {L, T}
   nX = length(Rs)
   len = sizeY(L)

   x = temps.x
   y = temps.y
   z = temps.z
   r² = temps.r²
   s = temps.s
   c = temps.c
   Q = temps.Q
   Flm = temps.Flm

   # size checks to make sure the inbounds macro can be used safely. 
   @assert length(y) == length(z) == nX 
   @assert length(r²) >= nX
   @assert size(Z, 1) >= nX && size(s, 1) >= nX  && size(c, 1) >= nX && size(Q, 1) >= nX
   @assert size(Z, 2) >= len && size(Q, 2) >= len 
   @assert size(s, 2) >= L+1 && size(c, 2) >= L+1
   @assert size(dZ, 1) >= nX
   @assert size(dZ, 2) >= len

   rt2 = sqrt(T(2)) 
   _0 = zero(T)
   _1 = one(T)
   
   @inbounds @simd ivdep for j = 1:nX
      𝐫 = Rs[j] 
      xj, yj, zj = 𝐫[1], 𝐫[2], 𝐫[3]
      x[j] = xj
      y[j] = yj
      z[j] = zj
      r²[j] = xj^2 + yj^2 + zj^2
      # c_m and s_m, m = 0 
      s[j, 1] = zero(T)    # 0 -> 1
      c[j, 1] = one(T)     # 0 -> 1
   end

   # c_m and s_m continued 
   @inbounds for m = 1:L 
      @simd ivdep for j = 1:nX
         # m -> m+1 and  m-1 -> m
         s[j, m+1] = s[j, m] * x[j] + c[j, m] * y[j]
         c[j, m+1] = c[j, m] * x[j] - s[j, m] * y[j]
      end
   end

   i00 = lm2idx(0, 0)
   @inbounds @simd ivdep for j = 1:nX
      # fill Q_0^0 and Z_0^0 
      Q[j, i00] = one(T)
      Z[j, i00] = (Flm[0,0]/rt2) * Q[j, i00]

      # gradients 
      dZ[j, i00] = zero(SVector{3, T})
   end

   # need to treat l = 1 separately 
   i11 = lm2idx(1, 1)
   i1⁻1 = lm2idx(1, -1)
   i00 = lm2idx(0, 0)
   i10 = lm2idx(1, 0)
   F_1_1 = Flm[1,1]
   F_1_0 = Flm[1,0]

   @inbounds @simd ivdep for j = 1:nX 
      # Q_1^1, Y_1^1, Y_1^-1
      # Q_j_00 = _1
      Q[j, i11] = - _1
      Z[j, i11] = - F_1_1 * c[j, 2]    # 2 => l = 1
      Z[j, i1⁻1] = - F_1_1 * s[j, 2]

      # Q_l^l-1 and Y_l^l-1
      # m = l-1 
      Q[j, i10]  = Q_j_10 = z[j]
      Z[j, i10]  = F_1_0 * Q_j_10 * c[j, 1] / rt2    # l-1 -> l

      # gradients 
      dZ[j, i11]  = SA[- F_1_1, _0, _0]
      dZ[j, i1⁻1] = SA[_0, - F_1_1, _0]
      dZ[j, i10]  = SA[_0, _0,  F_1_0 / rt2 ]                                   
   end

   # now from l = 2 onwards 
   @inbounds for l = 2:L 
      ill = lm2idx(l, l)
      il⁻l = lm2idx(l, -l)
      ill⁻¹ = lm2idx(l, l-1)
      il⁻¹l⁻¹ = lm2idx(l-1, l-1)
      il⁻l⁺¹ = lm2idx(l, -l+1)
      il⁻¹l = lm2idx(l-1, l)
      
      F_l_l = Flm[l,l]
      F_l_l⁻¹ = Flm[l,l-1]

      @simd ivdep for j = 1:nX 
         # Q_l^l and Y_l^l
         # m = l 
         Q_j_l⁻¹l⁻¹ = Q[j, il⁻¹l⁻¹]
         Q[j, ill] = Q_j_ll = - (2*l-1) * Q_j_l⁻¹l⁻¹
         Z[j, ill] = F_l_l * Q_j_ll * c[j, l+1]  # l -> l+1
         Z[j, il⁻l] = F_l_l * Q_j_ll * s[j, l+1]  # l -> l+1

         # Q_l^l-1 and Y_l^l-1
         # m = l-1 
         Q_j_il⁻¹l⁻¹ = Q[j, il⁻¹l⁻¹]
         Q[j, ill⁻¹]  = Q_j_ll⁻¹ = (2*l-1) * z[j] * Q_j_l⁻¹l⁻¹
         Z[j, il⁻l⁺¹] = F_l_l⁻¹ * Q_j_ll⁻¹ * s[j, l]  # l-1 -> l
         Z[j, ill⁻¹]  = F_l_l⁻¹ * Q_j_ll⁻¹ * c[j, l]  # l-1 -> l

         # gradients 

         # l = m 
         # Q_j_ll = const => ∇Q_j_ll = 0
         dZ[j, ill]  = F_l_l * Q_j_ll * SA[l * c[j, l], -l * s[j, l], _0]
         dZ[j, il⁻l] = F_l_l * Q_j_ll * SA[l * s[j, l],  l * c[j, l], _0]

         # m = l-1
         # Q_j_l⁻¹l⁻¹ = const => ∇_{xy}Q_j_l⁻¹l⁻¹ = 0
         dZ[j, il⁻l⁺¹] = F_l_l⁻¹ * SA[Q_j_ll⁻¹ * (l-1) * s[j, l-1], 
                                      Q_j_ll⁻¹ * (l-1) * c[j, l-1], 
                                      (2*l-1) * Q_j_l⁻¹l⁻¹ * s[j, l] ]
         dZ[j, ill⁻¹]  = F_l_l⁻¹ * SA[Q_j_ll⁻¹ * (l-1) * c[j, l-1],
                                      Q_j_ll⁻¹ * (-l+1) * s[j, l-1], 
                                      (2*l-1) * Q_j_l⁻¹l⁻¹ * c[j, l] ]
      end

      # now we can go to the second recursion 
      # unfortunately we have to treat m = 0 separately again 
      for m = l-2:-1:1 
         ilm = lm2idx(l, m)
         il⁻m = lm2idx(l, -m)
         il⁻¹m = lm2idx(l-1, m)
         il⁻²m = lm2idx(l-2, m)
         il⁻¹m⁺¹ = lm2idx(l-1, m+1)
         _f = (m == 0) ? _1/rt2 : _1

         F_l_m = Flm[l,m]
         F_l_m_f = F_l_m * _f

         @simd ivdep for j = 1:nX 
            cj = c[j, m+1]; sj = s[j, m+1]   # m -> m+1
            Q[j, ilm] = Q_lm = ((2*l-1) * z[j] * Q[j, il⁻¹m] - (l+m-1) * r²[j] * Q[j, il⁻²m]) / (l-m)
            Z[j, il⁻m] = F_l_m * Q_lm * sj   
            Z[j, ilm] = F_l_m_f * Q_lm * cj 

            # gradients
            Q_lm_x = x[j] * Q[j, il⁻¹m⁺¹]
            Q_lm_y = y[j] * Q[j, il⁻¹m⁺¹]
            Q_lm_z = (l+m) * Q[j, il⁻¹m]
            s_x = m * s[j, m]
            s_y = m * c[j, m]
            c_x = m * c[j, m]
            c_y = -m * s[j, m]

            dZ[j, il⁻m] = F_l_m * SA[Q_lm * s_x + Q_lm_x * sj, 
                                     Q_lm * s_y + Q_lm_y * sj, 
                                                  Q_lm_z * sj ]
            dZ[j, ilm] = F_l_m_f * SA[Q_lm * c_x + Q_lm_x * cj,
                                    Q_lm * c_y + Q_lm_y * cj, 
                                                 Q_lm_z * cj ]                                                 
         end
      end

      # special case m = 0: only if l = 2 or larger. 
      # for l = 1 it is already taken care of above. 
      if l >= 2 
         # m = 0 
         il0 = lm2idx(l, 0)
         il⁻¹0 = lm2idx(l-1, 0)
         il⁻²0 = lm2idx(l-2, 0)
         il⁻¹1 = lm2idx(l-1, 1)

         F_l_0_f = Flm[l,0] / rt2

         @simd ivdep for j = 1:nX 
            cj = c[j, 1]; sj = s[j, 1]   # 1 => m = 0
            Q[j, il0] = Q_l0 = ((2*l-1) * z[j] * Q[j, il⁻¹0] - (l-1) * r²[j] * Q[j, il⁻²0]) / l
            Z[j, il0] = F_l_0_f * Q_l0 * cj

            # gradients
            Q_l0_x = x[j] * Q[j, il⁻¹1]
            Q_l0_y = y[j] * Q[j, il⁻¹1]
            Q_l0_z = l * Q[j, il⁻¹0]

            dZ[j, il0] = F_l_0_f * cj * SA[Q_l0_x, Q_l0_y, Q_l0_z ]                                                 
         end
      end

   end

   return nothing 
end
