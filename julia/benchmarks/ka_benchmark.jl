##

# move this over to benchmarks asap...
# @info("Hand-coded (single-threaded)")
# @btime compute!(Z1, basis, Rs)
# @info("KA-Metal-32")
# # @btime SpheriCart.solid_harmonics!(Z2, Val(L), Rs_gpu, Flm_gpu, 32)
# @info("KA-Metal-16")
# @btime SpheriCart.solid_harmonics!(Z2, Val(L), Rs_gpu, Flm_gpu, 16)
# @info("KA-Metal-8")
# @btime SpheriCart.solid_harmonics!(Z2, Val(L), Rs_gpu, Flm_gpu, 8)
# @info("KA-CPU")
# @btime SpheriCart.ka_solid_harmonics!(Z3, nothing, Val{L}(), Rs, Flm_cpu)

##

# using ForwardDiff
# using ForwardDiff: Dual

# _part3(TV, i) = ForwardDiff.Partials(ntuple(j -> one(TV) * (i==j), 3))
# _dual(𝐫::SVector{N, T}) where {N, T} = 
#             SVector{N}( ntuple(j -> Dual(𝐫[j], _part3(T, j)), N)... )

# Rsd = MtlArray(_dual.(Rs))
# TD = eltype(Rsd)
# Zd = MtlArray(zeros(TD, size(Z1)))

# Z4 = MtlArray(zeros(Float32, size(Z1)))
# ∇Z4 = MtlArray(zeros(SVector{3, Float32}, size(Z1)))

# # evaluate with duals
# SpheriCart.solid_harmonics!(Zd, Val(L), Rsd, Flm_gpu, 16)


##

# @btime SpheriCart.solid_harmonics!(Z2, Val(L), Rs_gpu, Flm_gpu, 16)
# @btime SpheriCart.solid_harmonics!(Zd_gpu, Val(L), Rsd_gpu, Flm_gpu, 16)
