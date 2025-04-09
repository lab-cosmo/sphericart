
using SpheriCart, GPUArraysCore, Metal, StaticArrays,  
      LinearAlgebra, KernelAbstractions, BenchmarkTools

##

L = 6
basis = SolidHarmonics(L)
nbatch = 128_000

Rs_64 = [ @SVector randn(3) for _=1:nbatch ]
Rs_64 = [ 𝐫 / norm(𝐫) for 𝐫 in Rs_64 ]
Rs = [ Float32.(𝐫) for 𝐫 in Rs_64 ]
Z1 = compute(basis, Rs_64)

T = Float32
temps = (x = zeros(T, nbatch), 
         y = zeros(T, nbatch),
         z = zeros(T, nbatch), 
        r² = zeros(T, nbatch),
         s = zeros(T, nbatch, L+1), 
         c = zeros(T, nbatch, L+1),
         Q = zeros(T, nbatch, SpheriCart.sizeY(L)),
       Flm = T.(basis.Flm) )

Z1_32 = zeros(Float32, size(Z1))
SpheriCart.solid_harmonics!(Z1_32, Val(L), Rs, temps)

@info(L)
@show norm(Z1 - Z1_32, Inf)

##

Rs_mat = collect(reinterpret(reshape, Float32, Rs))
Rs_gpu = MtlArray(Rs_mat)
Z2 = MtlArray(zeros(Float32, size(Z1)))
Flm_gpu = MtlArray(Float32.(basis.Flm.parent))

SpheriCart.solid_harmonics!(Z2, Val(L), Rs_gpu, Flm_gpu)
Z2_cpu = Matrix(Z2)

@show norm(Z1_32 - Z2_cpu, Inf)

##

Z3 = deepcopy(Z1)
Flm_cpu = Float32.(basis.Flm.parent)
SpheriCart.ka_solid_harmonics!(Z3, Val{L}(), Rs_mat, Flm_cpu)

@show norm(Z1_32 - Z3, Inf)

##

@info("Hand-coded (single-threaded)")
@btime compute!(Z1, basis, Rs)
@info("KA-Metal")
@btime SpheriCart.solid_harmonics!(Z2, Val(L), Rs_gpu, Flm_gpu)
@info("KA-CPU")
@btime SpheriCart.ka_solid_harmonics!(Z3, Val{L}(), Rs_mat, Flm_cpu)
