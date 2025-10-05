using KernelAbstractions, GPUArraysCore
using StaticArrays: SMatrix 


function compute!(Z::AbstractGPUMatrix, 
                  basis::SolidHarmonics{L, NORM, STATIC}, 
                  Rs::AbstractGPUVector{SVector{3, T}}
                  ) where {L, NORM, STATIC, T}

   if !STATIC 
      error("GPU evaluation of SolidHarmonics is only implemented for the static basis.")
   end

   nX = length(Rs)
   @assert size(Z, 1) >= nX

   solid_harmonics!(Z, Val{L}(), Rs, basis.Flm)

   return Z 
end 

function compute_with_gradients!(Z::AbstractGPUMatrix, dZ::AbstractGPUMatrix,
                  basis::SolidHarmonics{L, NORM, STATIC}, 
                  Rs::AbstractGPUVector{SVector{3, T}}
                  ) where {L, NORM, STATIC, T}

   if !STATIC 
      error("GPU evaluation of SolidHarmonics is only implemented for the static basis.")
   end

   nX = length(Rs)
   @assert size(Z, 1) >= nX
   @assert size(dZ, 1) >= nX

   solid_harmonics_with_grad!(Z, dZ, Val{L}(), Rs, basis.Flm)

   return Z, dZ  
end 



#
# splitting off this function barrier, to allow experimenting with 
# KA for CPU via calling ka_solid_harmonics directly. 
#
function solid_harmonics!(
               Z::AbstractGPUArray, 
               ::Val{L}, 
               Rs::AbstractGPUArray, 
               Flm::Union{AbstractGPUArray, SMatrix}, 
               GRPSZ = 32) where {L}
   ka_solid_harmonics!(Z, nothing, Val{L}(), Rs, Flm, GRPSZ)
end


function solid_harmonics_with_grad!(
               Z::AbstractGPUArray, dZ::AbstractGPUArray, 
               ::Val{L}, 
               Rs::AbstractGPUArray, 
               Flm::Union{AbstractGPUArray, SMatrix}, 
               GRPSZ = 32) where {L} 
   ka_solid_harmonics!(Z, dZ, Val{L}(), Rs, Flm, GRPSZ)
end

function ka_solid_harmonics!(
               Z::AbstractArray, 
               ::Val{L}, 
               Rs::AbstractArray, 
               Flm::Union{AbstractArray, SMatrix}, 
               GRPSZ = 32) where {L}
   ka_solid_harmonics!(Z, nothing, Val{L}(), Rs, Flm, GRPSZ)
end


function ka_solid_harmonics_with_grad!(
               Z::AbstractArray, dZ::AbstractArray, 
               ::Val{L}, 
               Rs::AbstractArray, 
               Flm::Union{AbstractArray, SMatrix}, 
               GRPSZ = 32) where {L} 
   ka_solid_harmonics!(Z, dZ, Val{L}(), Rs, Flm, GRPSZ)
end



"""
```
      ka_solid_harmonics!(Z, dZ, ::Val{L}, [::Val{SH}],
                  Rs::AbstractVector{<: SVector{3}}, Flm, 
                  GRPSZ = 32) where {L}
```
KernelAbstractions.jl kernel launcher for evaluating solid harmonics 
on a batch of input points. If `dZ == nothing` then only `Z will be 
evaluated, otherwise, both `Z` and `dZ`.
* `Z, dZ` : output arrays, if `isnothing(dZ)` then gradients are not computed 
* `L` : max degree / angular momentum number 
* `SH` : `SH = true` for spherical harmonics, `SH = false` for solid harmonics, default is `SH = false`
* `Rs` : input; Vector or 3-vectors
* `Flm` : precomputed normalization factors
"""
function ka_solid_harmonics!(Z, dZ, ::Val{L}, ::Val{SH}, 
                  Rs::AbstractVector{<: SVector{3}}, Flm, 
                  GRPSZ = 32) where {L, SH}

   # check sizes to make sure the inbounds macro can be used safely.
   nRs = size(Rs, 2) 
   @assert size(Z, 1) >= nRs 
   len = sizeY(L)
   @assert size(Z, 2) >= len 

   if !isnothing(dZ)   # IF GRADIENTS 
      @assert size(dZ, 1) >= nRs
      @assert size(dZ, 2) >= len
   end

   backend = KernelAbstractions.get_backend(Z)  # assume same for dZ 
   solidh_main! = _ka_solidh_main!(backend, (GRPSZ,))

   # call the kernels 
   nRs = length(Rs)
   solidh_main!(Z, dZ, Val{L}(), Val{SH}(), Rs, Flm; ndrange = (nRs,))
   synchronize(backend)

   nothing               
end

ka_solid_harmonics!(Z, dZ, ::Val{L}, # ::Val{SH}, (skip this argument)
                    Rs::AbstractVector{<: SVector{3}}, Flm, 
                    GRPSZ = 32) where {L} = 
      ka_solid_harmonics!(Z, dZ, Val{L}(), Val{false}(), Rs, Flm, GRPSZ)


@kernel function _ka_solidh_main!(
               Z, dZ, ::Val{L}, ::Val{SH}, 
               @Const(Rs), 
               @Const(Flm), ) where {L, SH}

   j = @index(Global, Linear)
   jl = @index(Local, Linear)
   @uniform len_grp = prod(@groupsize())
   @uniform WITHGRAD = !isnothing(dZ)
   
   @uniform T = eltype(Z)
   @uniform len = sizeY(L)
   @uniform rt2 = sqrt(T(2))
   @uniform _1 = one(T)
   @uniform _0 = zero(T)

   # allocations 
   x = @localmem T (len_grp,)
   y = @localmem T (len_grp,)
   z = @localmem T (len_grp,)
   r² = @localmem T (len_grp,)
   r = @localmem T (len_grp,)
   s = @localmem T (len_grp, L+1)
   c = @localmem T (len_grp, L+1)
   Q = @localmem T (len_grp, len)   # nb : len ≈ L^2

   # ------------------------------------------------------------------
   # STAGE 1a: load the coordinates into more convenient local variables 

   # TODO: unclear to me why I have to allocate these arrays rather than 
   #       simply working in thread-private variables. (I tried but only 
   #       got lots of unexplained errors)

   x[jl], y[jl], z[jl] = Rs[j].data  # Rs[j]::SVector{3}
   r²[jl] = x[jl]*x[jl] + y[jl]*y[jl] + z[jl]*z[jl] 

   # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   #  normalize the input if we want spherical harmonics
   if SH 
      r[jl] = sqrt(r²[jl])
      x[jl] /= r[jl]
      y[jl] /= r[jl]
      z[jl] /= r[jl]
      r²[jl] = _1
   end
   # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


   # ------------------------------------------------------------------
   # STAGE 1b: evaluate sines and cosines 

   # initialise sin(0*θ), cos(0*θ)
   s[jl, 1] = _0    # 0 -> 1 (1-based indexing)
   c[jl, 1] = _1
   # construct sin(mθ), cos(mθ) recursively {m -> m+1 and  m-1 -> m}
   @inbounds for m = 1:L
      s[jl, m+1] = s[jl, m] * x[jl] + c[jl, m] * y[jl]
      c[jl, m+1] = c[jl, m] * x[jl] - s[jl, m] * y[jl]
   end

   # ------------------------------------------------------------------
   # STAGE 2: the actual computation ... 
   
   # fill Q_0^0 and Z_0^0 
   i00 = lm2idx(0, 0)
   Q[jl, i00] = _1
   Z[j, i00] = (Flm[1,1]/rt2)   # Q[jl, i00]

   # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   if WITHGRAD
      dZ[j, i00] = zero(SVector{3, T})
   end
   # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   # treat l = 1 separately 
   i11 = lm2idx(1, 1)
   i1⁻1 = lm2idx(1, -1)
   i00 = lm2idx(0, 0)
   i10 = lm2idx(1, 0)
   F_1_1 = Flm[2, 2]  # 1-based indexing
   F_1_0 = Flm[2, 1]

   # Q_1^1, Y_1^1, Y_1^-1
   # Q_j_00 = _1
   Q[jl, i11] = - _1
   Z[j, i11]  = - F_1_1 * c[jl, 2]    # 2 => l = 1
   Z[j, i1⁻1] = - F_1_1 * s[jl, 2]

   # Q_l^l-1 and Y_l^l-1
   # m = l-1 
   Q[jl, i10] = Q_j_10 = z[jl]
   Z[j,  i10] = F_1_0 * Q_j_10 * c[jl, 1] / rt2    # l-1 -> l

   # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   if WITHGRAD
      dZ[j, i11]  = SA[- F_1_1, _0, _0]
      dZ[j, i1⁻1] = SA[_0, - F_1_1, _0]
      dZ[j, i10]  = SA[_0, _0,  F_1_0 / rt2 ]                                   
   end
   # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


   @inbounds for l = 2:L 
      ill     = lm2idx(l,     l)
      il⁻l    = lm2idx(l,    -l)
      ill⁻¹   = lm2idx(l,   l-1)
      il⁻¹l⁻¹ = lm2idx(l-1, l-1)
      il⁻l⁺¹  = lm2idx(l,  -l+1)
      il⁻l⁺¹  = lm2idx(l,  -l+1)
      il⁻¹l   = lm2idx(l-1,   l)
 
      F_l_l   = Flm[l+1,l+1]
      F_l_l⁻¹ = Flm[l+1,l]

      # ----- inner j-loop -----
      # Q_l^l and Y_l^l
      # m = l 
      Q_j_l⁻¹l⁻¹ = Q[jl, il⁻¹l⁻¹]
      Q[jl, ill] = Q_j_ll = - (2*l-1) * Q_j_l⁻¹l⁻¹
      Z[j, ill]    = F_l_l * Q[jl, ill] * c[jl, l+1]  # l -> l+1
      Z[j, il⁻l]   = F_l_l * Q[jl, ill] * s[jl, l+1]  # l -> l+1

      # Q_l^l-1 and Y_l^l-1
      # m = l-1   (NB: this deals with l = 1, m = 0 => special-case below)
      Q_j_il⁻¹l⁻¹ = Q[jl, il⁻¹l⁻¹]
      Q[jl, ill⁻¹] = Q_j_ll⁻¹ = (2*l-1) * Q_j_il⁻¹l⁻¹ * z[jl]
      Z[j, il⁻l⁺¹] = F_l_l⁻¹ * Q_j_ll⁻¹ * s[jl, l]    # l-1 -> l
      Z[j, ill⁻¹]  = F_l_l⁻¹ * Q_j_ll⁻¹ * c[jl, l]    # l-1 -> l

      # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      if WITHGRAD
         # l = m 
         # Q_j_ll = const => ∇Q_j_ll = 0
         dZ[j, ill]  = F_l_l * Q_j_ll * SA[l * c[jl, l], -l * s[jl, l], _0]
         dZ[j, il⁻l] = F_l_l * Q_j_ll * SA[l * s[jl, l],  l * c[jl, l], _0]

         # m = l-1
         # Q_j_l⁻¹l⁻¹ = const => ∇_{xy}Q_j_l⁻¹l⁻¹ = 0
         dZ[j, il⁻l⁺¹] = F_l_l⁻¹ * SA[Q_j_ll⁻¹ * (l-1) * s[jl, l-1], 
                                      Q_j_ll⁻¹ * (l-1) * c[jl, l-1], 
                                      (2*l-1) * Q_j_l⁻¹l⁻¹ * s[jl, l] ]
         dZ[j, ill⁻¹]  = F_l_l⁻¹ * SA[Q_j_ll⁻¹ * (l-1)  * c[jl, l-1],
                                      Q_j_ll⁻¹ * (-l+1) * s[jl, l-1], 
                                      (2*l-1) * Q_j_l⁻¹l⁻¹ * c[jl, l] ]
      end
      # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

      # now we can go to the second recursion 
      for m = l-2:-1:1 
         ilm = lm2idx(l, m)
         il⁻m = lm2idx(l, -m)
         il⁻¹m = lm2idx(l-1, m)
         il⁻²m = lm2idx(l-2, m)
         il⁻¹m⁺¹ = lm2idx(l-1, m+1)

         F_l_m = Flm[l+1,m+1]

         # ----- inner j-loop ----- 
         cj = c[jl, m+1]; sj = s[jl, m+1]   # m -> m+1
         Q[jl, ilm]  = Q_lm = ((2*l-1) * z[jl] * Q[jl, il⁻¹m] - (l+m-1) * r²[jl] * Q[jl, il⁻²m]) / (l-m)
         Z[j,  il⁻m] = F_l_m * Q[jl, ilm] * sj
         Z[j,  ilm]  = F_l_m * Q[jl, ilm] * cj

         if WITHGRAD 
            # first compute a few partial derivatives of the auxiliary variables 
            Q_lm_x = x[jl] * Q[jl, il⁻¹m⁺¹]
            Q_lm_y = y[jl] * Q[jl, il⁻¹m⁺¹]
            Q_lm_z = (l+m) * Q[jl, il⁻¹m]
            s_x = m * s[jl, m]; s_y =  m * c[jl, m]
            c_x = m * c[jl, m]; c_y = -m * s[jl, m]

            dZ[j, il⁻m] = F_l_m * SA[Q_lm * s_x + Q_lm_x * sj, 
                                     Q_lm * s_y + Q_lm_y * sj, 
                                                  Q_lm_z * sj ]
            dZ[j,  ilm] = F_l_m * SA[Q_lm * c_x + Q_lm_x * cj,
                                     Q_lm * c_y + Q_lm_y * cj, 
                                                  Q_lm_z * cj ]
         end
      end  # end for m ...:1

      # Note l = 1 is treated above the l-loop, so here we always 
      # have l >= 2. (a previous version made a case distinction here)

      # m = 0 
      il0   = lm2idx(l,   0)
      il⁻¹0 = lm2idx(l-1, 0)
      il⁻²0 = lm2idx(l-2, 0)
      il⁻¹1 = lm2idx(l-1, 1)

      F_l_0_f = Flm[l+1, 1] / rt2

      cj = c[jl, 1]; sj = s[jl, 1]   # 1 => m = 0
      Q[jl, il0] = Q_l0 = ((2*l-1) * z[jl] * Q[jl, il⁻¹0] - (l-1) * r²[jl] * Q[jl, il⁻²0]) / l
      Z[j,  il0] = F_l_0_f * Q_l0 * cj

      if WITHGRAD
         Q_l0_x = x[jl] * Q[jl, il⁻¹1]
         Q_l0_y = y[jl] * Q[jl, il⁻¹1]
         Q_l0_z = l * Q[jl, il⁻¹0]

         dZ[j, il0] = F_l_0_f * cj * SA[Q_l0_x, Q_l0_y, Q_l0_z ]
      end

   end

   # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   # fix the spherical harmonics gradients
   if SH && WITHGRAD
      for α = 1:size(dZ, 2) 
         dzj = dZ[j, α] / r[jl]
         𝐫̂j = SA[x[jl], y[jl], z[jl]]
         dZ[j, α] = dzj - dot(𝐫̂j, dzj) * 𝐫̂j
      end
   end
   # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   nothing 
end
