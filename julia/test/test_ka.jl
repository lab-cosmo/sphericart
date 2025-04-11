
using SpheriCart, GPUArraysCore, Metal, StaticArrays,  
      LinearAlgebra, KernelAbstractions, BenchmarkTools

##

L = 6
@info("L = $L")
basis = SolidHarmonics(L; T = Float32)
Flm_cpu = basis.Flm.parent
Flm_gpu = MtlArray(Flm_cpu)

nbatch = 128_000

Rs = [ @SVector randn(Float32, 3) for _=1:nbatch ]
Rs = [ 𝐫 / norm(𝐫) for 𝐫 in Rs ]
Rs_gpu = MtlArray(Rs)

# reference calculation 
Z1 = compute(basis, Rs)

##

@info("test GPU-KA execution")
Z2 = MtlArray(zeros(Float32, size(Z1)))
SpheriCart.solid_harmonics!(Z2, Val(L), Rs_gpu, Flm_gpu)
@show norm(Z1 - Array(Z2), Inf)

##

@info("test CPU-KA execution")
Z3 = zeros(Float32, size(Z1))
SpheriCart.ka_solid_harmonics!(Z3, Val{L}(), Rs, Flm_cpu)
@show norm(Z1 - Z3, Inf)

##

# @info("Hand-coded (single-threaded)")
# @btime compute!(Z1, basis, Rs)
# @info("KA-Metal-32")
# @btime SpheriCart.solid_harmonics!(Z2, Val(L), Rs_gpu, Flm_gpu, 32)
# @info("KA-Metal-16")
# @btime SpheriCart.solid_harmonics!(Z2, Val(L), Rs_gpu, Flm_gpu, 16)
# @info("KA-Metal-8")
# @btime SpheriCart.solid_harmonics!(Z2, Val(L), Rs_gpu, Flm_gpu, 8)
# @info("KA-CPU")
# @btime SpheriCart.ka_solid_harmonics!(Z3, Val{L}(), Rs, Flm_cpu)

##

# using ForwardDiff
# using ForwardDiff: Dual

# _part3(TV, i) = ForwardDiff.Partials(ntuple(j -> one(TV) * (i==j), 3))
# _dual(𝐫::SVector{N, T}) where {N, T} = 
#             SVector{N}( ntuple(j -> Dual(𝐫[j], _part3(T, j)), N)... )

# Rsd = _dual.(Rs)
# Rsd_gpu = MtlArray(Rsd)
# TD = eltype(Rsd[1])
# Zd = zeros(TD, size(Z1))
# Zd_gpu = MtlArray(Zd)

##

# @btime SpheriCart.solid_harmonics!(Z2, Val(L), Rs_gpu, Flm_gpu, 16)
# @btime SpheriCart.solid_harmonics!(Zd_gpu, Val(L), Rsd_gpu, Flm_gpu, 16)

##


Z2 = MtlArray(zeros(Float32, size(Z1)))
Z2a = MtlArray(zeros(Float32, size(Z1)))

SpheriCart.solid_harmonics!(Z2, Val(L), Rs_gpu, Flm_gpu)
@show norm(Z1 - Array(Z2), Inf)

SpheriCart.ka_solid_harmonics!(Z2a, nothing, Val(L), Rs_gpu, Flm_gpu)
@show norm(Z1 - Array(Z2a), Inf)


##

@btime SpheriCart.solid_harmonics!(Z2, Val(L), Rs_gpu, Flm_gpu)
@btime SpheriCart.ka_solid_harmonics!(Z2a, nothing, Val(L), Rs_gpu, Flm_gpu)
