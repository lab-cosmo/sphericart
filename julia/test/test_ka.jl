
using SpheriCart, GPUArraysCore, Metal, StaticArrays,  
      LinearAlgebra, KernelAbstractions, BenchmarkTools

##

L = 6
@info(L)
basis = SolidHarmonics(L; T = Float32)
Flm_cpu = basis.Flm.parent
Flm_gpu = MtlArray(Flm_cpu)

nbatch = 64_000

Rs = [ @SVector randn(Float32, 3) for _=1:nbatch ]
Rs = [ 𝐫 / norm(𝐫) for 𝐫 in Rs ]
Rs_mat = collect(reinterpret(reshape, Float32, Rs))
Rs_gpu = MtlArray(Rs)
Rs_mat_gpu = MtlArray(Rs_mat)
Rs_gpu_tup = MtlArray([ 𝐫.data for 𝐫 in Rs ])

# reference calculation 
Z1 = compute(basis, Rs)

##

Z2a = MtlArray(zeros(Float32, size(Z1)))
Z2b = MtlArray(zeros(Float32, size(Z1)))
Z2c = MtlArray(zeros(Float32, size(Z1)))
SpheriCart.solid_harmonics!(Z2a, Val(L), Rs_gpu, Flm_gpu)
SpheriCart.solid_harmonics!(Z2b, Val(L), Rs_mat_gpu, Flm_gpu)
SpheriCart.solid_harmonics!(Z2c, Val(L), Rs_gpu_tup, Flm_gpu)

@show norm(Z1 - Array(Z2a), Inf)
@show norm(Z1 - Array(Z2b), Inf)
@show norm(Z1 - Array(Z2c), Inf)

##

Z3a = zeros(Float32, size(Z1))
Z3b = zeros(Float32, size(Z1))
SpheriCart.ka_solid_harmonics!(Z3a, Val{L}(), Rs, Flm_cpu)
SpheriCart.ka_solid_harmonics!(Z3b, Val{L}(), Rs_mat, Flm_cpu)

@show norm(Z1 - Z3a, Inf)
@show norm(Z1 - Z3b, Inf)

##

@info("Hand-coded (single-threaded)")
@btime compute!(Z1, basis, Rs)
@info("KA-Metal-Matrix")
@btime SpheriCart.solid_harmonics!(Z2a, Val(L), Rs_gpu, Flm_gpu)
@info("KA-Metal-Vector{SVector}")
@btime SpheriCart.solid_harmonics!(Z2b, Val(L), Rs_mat_gpu, Flm_gpu)
@info("KA-Metal-Vector{Tuple}")
@btime SpheriCart.solid_harmonics!(Z2c, Val(L), Rs_gpu_tup, Flm_gpu)
@info("KA-CPU")
@btime SpheriCart.ka_solid_harmonics!(Z3b, Val{L}(), Rs_mat, Flm_cpu)
