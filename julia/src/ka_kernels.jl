using KernelAbstractions, GPUArraysCore

function solid_harmonics!(
               Z::AbstractGPUArray, 
               ::Val{L}, 
               Rs::AbstractGPUArray, 
               Flm::AbstractGPUArray) where {L}
   ka_solid_harmonics!(Z, Val{L}(), Rs, Flm)
   nothing               
end


function ka_solid_harmonics!(Z, ::Val{L}, Rs, Flm) where {L}
   # check sizes 
   @assert size(Rs, 1) == 3 
   nRs = size(Rs, 2)
   @assert size(Z, 1) >= nRs 
   len = sizeY(L)
   @assert size(Z, 2) >= len 

   # allocate temporary arrays
   x = similar(Rs, (nRs,))
   y = similar(Rs, (nRs,))
   z = similar(Rs, (nRs,))
   r² = similar(Rs, (nRs,))
   s = similar(Rs, (nRs, L+1))
   c = similar(Rs, (nRs, L+1))
   Q = similar(Rs, (nRs, len))

   # compile the kernels 
   backend = KernelAbstractions.get_backend(Z)
   solidh_load! = _ka_solidh_load!(backend) 
   solidh_sincos! = _ka_solidh_sincos!(backend)
   solidh_main! = _ka_solidh_main!(backend)

   # call the kernels 
   solidh_load!(x, y, z, r², Rs; ndrange = (nRs,))
   solidh_sincos!(s, c, x, y, Val{L}(); ndrange = (nRs,))
   solidh_main!(Z, Val{L}(), Q, Rs, Flm, x, y, z, r², s, c; ndrange = (nRs,))
   nothing; 
end


@kernel function _ka_solidh_load!(x, y, z, r², @Const(Rs))
   j = @index(Global) 
   x[j] = Rs[1, j] 
   y[j] = Rs[2, j]
   z[j] = Rs[3, j]
   r²[j] = x[j]^2 + y[j]^2 + z[j]^2
   nothing; 
end

@kernel function _ka_solidh_sincos!(s, c, @Const(x), @Const(y), ::Val{L})  where {L}
   j = @index(Global) 
   T = eltype(s) 
   # initialise sin(0*θ), cos(0*θ)
   s[j, 1] = zero(T)    # 0 -> 1 (1-based indexing)
   c[j, 1] = one(T)
   # construct sin(mθ), cos(mθ) recursively {m -> m+1 and  m-1 -> m}
   for m = 1:L
      s[j, m+1] = s[j, m] * x[j] + c[j, m] * y[j]
      c[j, m+1] = c[j, m] * x[j] - s[j, m] * y[j]
   end
   # change c[0] to 1/rt2 to avoid a special case l-1=m=0 later 
   T = eltype(s)
   c[j, 1] = one(T)/sqrt(T(2))

   nothing 
end 

# @kernel function _ka_solidh_sincos!(s, c, @Const(x), @Const(y))
#    j = @index(Global) 
#    T = eltype(s) 
#    L = size(s, 2) - 1
#    # initialise sin(m*θ), cos(m*θ)
#    for m = 0:L 
#       s[j, m+1], c[j, m+1] = sincos(m * atan(y[j], x[j]))
#    end
#    # change c[0] to 1/rt2 to avoid a special case l-1=m=0 later 
#    c[j, 1] = one(T)/sqrt(T(2))
#    nothing 
# end 

@kernel function _ka_solidh_main!(
               Z, ::Val{L}, Q, 
               @Const(Rs), 
               @Const(Flm), 
               @Const(x), @Const(y), @Const(z), 
               @Const(r²), @Const(s), @Const(c) ) where {L}
   T = eltype(Z)
   j = @index(Global)
   len = sizeY(L)
   rt2 = sqrt(T(2)) 
   
   # fill Q_0^0 and Z_0^0 
   i00 = lm2idx(0, 0)
   Q[j, i00] = one(T)
   Z[j, i00] = (Flm[1,1]/rt2) * Q[j, i00]

   for l = 1:L 
      ill = lm2idx(l, l)
      il⁻l = lm2idx(l, -l)
      ill⁻¹ = lm2idx(l, l-1)
      il⁻¹l⁻¹ = lm2idx(l-1, l-1)
      il⁻l⁺¹ = lm2idx(l, -l+1)
      F_l_l = Flm[l+1,l+1]
      F_l_l⁻¹ = Flm[l+1,l]

      # ----- inner j-loop ----- 
      # Q_l^l and Y_l^l
      # m = l 
      Q[j, ill]  = - (2*l-1) * Q[j, il⁻¹l⁻¹]
      Z[j, ill]  = F_l_l * Q[j, ill] * c[j, l+1]  # l -> l+1
      Z[j, il⁻l] = F_l_l * Q[j, ill] * s[j, l+1]  # l -> l+1
      # Q_l^l-1 and Y_l^l-1
      # m = l-1 
      Q[j, ill⁻¹]  = (2*l-1) * z[j] * Q[j, il⁻¹l⁻¹]
      Z[j, il⁻l⁺¹] = F_l_l⁻¹ * Q[j, ill⁻¹] * s[j, l]  # l-1 -> l
      Z[j, ill⁻¹]  = F_l_l⁻¹ * Q[j, ill⁻¹] * c[j, l]  # l-1 -> l
      # overwrite if m = 0 -> ok 

      # now we can go to the second recursion 
      for m = l-2:-1:0 
         ilm = lm2idx(l, m)
         il⁻m = lm2idx(l, -m)
         il⁻¹m = lm2idx(l-1, m)
         il⁻²m = lm2idx(l-2, m)
         F_l_m = Flm[l+1,m+1]
         # ----- inner j-loop ----- 
         Q[j, ilm] = ((2*l-1) * z[j] * Q[j, il⁻¹m] - (l+m-1) * r²[j] * Q[j, il⁻²m]) / (l-m)
         Z[j, il⁻m] = F_l_m * Q[j, ilm] * s[j, m+1]   # m -> m+1
         Z[j, ilm] = F_l_m * Q[j, ilm] * c[j, m+1]    # m -> m+1
      end
   end

   nothing; 
end
