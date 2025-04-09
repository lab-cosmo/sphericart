
using SpheriCart, GPUArraysCore, Metal, StaticArrays,  
      LinearAlgebra, KernelAbstractions, BenchmarkTools

##

L = 6
@info("L = $L")
basis = SolidHarmonics(L; T = Float32)
Flm_cpu = basis.Flm.parent
Flm_gpu = MtlArray(Flm_cpu)

nbatch = 64_000

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

@info("Hand-coded (single-threaded)")
@btime compute!(Z1, basis, Rs)
@info("KA-Metal-32")
@btime SpheriCart.solid_harmonics!(Z2, Val(L), Rs_gpu, Flm_gpu, 32)
@info("KA-Metal-16")
@btime SpheriCart.solid_harmonics!(Z2, Val(L), Rs_gpu, Flm_gpu, 16)
@info("KA-Metal-8")
@btime SpheriCart.solid_harmonics!(Z2, Val(L), Rs_gpu, Flm_gpu, 8)
@info("KA-CPU")
@btime SpheriCart.ka_solid_harmonics!(Z3, Val{L}(), Rs, Flm_cpu)
