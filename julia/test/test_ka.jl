
using SpheriCart, GPUArraysCore, Metal, StaticArrays,  
      LinearAlgebra

##

L = 10
basis = SolidHarmonics(L)
nbatch = 100_000

Rs = [ @SVector randn(Float32, 3) for _=1:nbatch ]
Z1 = compute(basis, Rs)

##

Rs_mat = collect(reinterpret(reshape, Float32, Rs))
Rs_gpu = MtlArray(Rs_mat)
Z2 = MtlArray(zeros(Float32, size(Z1)))
Flm_gpu = MtlArray(Float32.(basis.Flm.parent))

SpheriCart.solid_harmonics!(Z2, Val(L), Rs_gpu, Flm_gpu)
Z2_cpu = Matrix(Z2)

@show norm(Z1 - Z2_cpu, Inf)

##

Z3 = deepcopy(Z1)
Flm_cpu = Float32.(basis.Flm.parent)
SpheriCart.ka_solid_harmonics!(Z3, Val{L}(), Rs_mat, Flm_cpu)

@show norm(Z1 - Z3, Inf)
@show norm(Z2_cpu - Z3, Inf)

##

nRs = size(Rs_gpu, 2)
len = SpheriCart.sizeY(L)
x = similar(Rs_gpu, (nRs,))
y = similar(Rs_gpu, (nRs,))
z = similar(Rs_gpu, (nRs,))
r² = similar(Rs_gpu, (nRs,))
s = similar(Rs_gpu, (nRs, L+1))
c = similar(Rs_gpu, (nRs, L+1))
Q = similar(Rs_gpu, (nRs, len))
Z4 = MtlArray(zeros(Float32, size(Z1)))
SpheriCart.ka_solid_harmonics!!(Z4, Val{L}(), Rs_gpu, Flm_gpu, x, y, z, r², s, c, Q)

@show norm(Z1 - Array(Z4), Inf)
@show norm(Z2_cpu - Array(Z4), Inf)                        

##

@info("Hand-coded (multi-threaded)")
@time compute!(Z1, basis, Rs)
@info("KA-Metal")
@time SpheriCart.solid_harmonics!(Z2, Val(L), Rs_gpu, Flm_gpu)
@info("KA-Metal (pre-allocated)")
@time SpheriCart.ka_solid_harmonics!!(Z4, Val{L}(), Rs_gpu, Flm_gpu, x, y, z, r², s, c, Q)
@info("KA-CPU")
@time SpheriCart.ka_solid_harmonics!(Z3, Val{L}(), Rs_mat, Flm_cpu)
