using KernelAbstractions, GPUArraysCore
using KernelAbstractions.Extras.LoopInfo: @unroll 

function solid_harmonics!(
               Z::AbstractGPUArray, 
               ::Val{L}, 
               Rs::AbstractGPUArray, 
               Flm::AbstractGPUArray) where {L}
   ka_solid_harmonics!(Z, Val{L}(), Rs, Flm)
end

function ka_solid_harmonics!(Z, ::Val{L}, Rs, Flm) where {L}

   # check sizes 
   @assert size(Rs, 1) == 3 
   nRs = size(Rs, 2)
   @assert size(Z, 1) >= nRs 
   len = sizeY(L)
   @assert size(Z, 2) >= len 

   # compile the kernels 
   backend = KernelAbstractions.get_backend(Z)
   solidh_main! = _ka_solidh_main!(backend, (32,))

   # call the kernels 
   nRs = size(Rs, 2)
   solidh_main!(Z, Val{L}(), Rs, Flm; ndrange = (nRs,))
   synchronize(backend)

   nothing               
end



@kernel function _ka_solidh_main!(
               Z, ::Val{L}, 
               @Const(Rs), 
               @Const(Flm) ) where {L}

   j = @index(Global)
   jl = @index(Local, Linear)
   @uniform len_grp = prod(@groupsize())
   
   @uniform T = eltype(Z)
   @uniform len = sizeY(L)
   @uniform rt2 = sqrt(T(2))

   # ------------------------------------------------------------------
   # STAGE 1a: load the coordinates into more convenient local variables 
   x = @localmem T (len_grp,)
   y = @localmem T (len_grp,)
   z = @localmem T (len_grp,)
   r² = @localmem T (len_grp,)

   x[jl] = Rs[1, j] 
   y[jl] = Rs[2, j]
   z[jl] = Rs[3, j]
   r²[jl] = x[jl]*x[jl] + y[jl]*y[jl] + z[jl]*z[jl] 

   # ------------------------------------------------------------------
   # STAGE 1b: evaluate sines and cosines 
   s = @localmem T (len_grp, L+1)
   c = @localmem T (len_grp, L+1)

   # initialise sin(0*θ), cos(0*θ)
   s[jl, 1] = zero(T)    # 0 -> 1 (1-based indexing)
   c[jl, 1] = one(T)
   # construct sin(mθ), cos(mθ) recursively {m -> m+1 and  m-1 -> m}
   for m = 1:L
      s[jl, m+1] = s[jl, m] * x[jl] + c[jl, m] * y[jl]
      c[jl, m+1] = c[jl, m] * x[jl] - s[jl, m] * y[jl]
   end
   # change c[0] to 1/rt2 to avoid a special case l-1=m=0 later 
   c[jl, 1] = one(T)/rt2

   # ------------------------------------------------------------------
   # STAGE 2: the actual computation ... 
   Q = @localmem T (len_grp, len)
   
   # fill Q_0^0 and Z_0^0 
   i00 = lm2idx(0, 0)
   Q[jl, i00] = one(T)
   Z[j, i00] = (Flm[1,1]/rt2) * one(T) # Q[jl, i00]

   for l = 1:L 
      ill = lm2idx(l, l)
      il⁻l = lm2idx(l, -l)
      ill⁻¹ = lm2idx(l, l-1)
      il⁻¹l⁻¹ = lm2idx(l-1, l-1)
      il⁻l⁺¹ = lm2idx(l, -l+1)
      F_l_l = Flm[l+1,l+1]
      F_l_l⁻¹ = Flm[l+1,l]

      # ----- inner j-loop ----- 
      Q[jl, ill]  = - (2*l-1) * Q[jl, il⁻¹l⁻¹]
      Q[jl, ill⁻¹]  = (2*l-1) * z[jl] * Q[jl, il⁻¹l⁻¹]
      Z[j, ill]  = F_l_l * Q[jl, ill] * c[jl, l+1]  # l -> l+1
      Z[j, il⁻l] = F_l_l * Q[jl, ill] * s[jl, l+1]  # l -> l+1
      Z[j, il⁻l⁺¹] = F_l_l⁻¹ * Q[jl, ill⁻¹] * s[jl, l]  # l-1 -> l
      Z[j, ill⁻¹]  = F_l_l⁻¹ * Q[jl, ill⁻¹] * c[jl, l]  # l-1 -> l
      # ------- 

      # now we can go to the second recursion 
      for m = l-2:-1:0 
         ilm = lm2idx(l, m)
         il⁻m = lm2idx(l, -m)
         il⁻¹m = lm2idx(l-1, m)
         il⁻²m = lm2idx(l-2, m)
         F_l_m = Flm[l+1,m+1]
         # ----- inner j-loop ----- 
         Q[jl, ilm] = ((2*l-1) * z[jl] * Q[jl, il⁻¹m] - (l+m-1) * r²[jl] * Q[jl, il⁻²m]) / (l-m)
         Z[j, il⁻m] = F_l_m * Q[jl, ilm] * s[jl, m+1]   # m -> m+1
         Z[j, ilm] = F_l_m * Q[jl, ilm] * c[jl, m+1]    # m -> m+1
      end
   end

   nothing; 
end
