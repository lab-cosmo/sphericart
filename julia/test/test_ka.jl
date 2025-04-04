
using SpheriCart, GPUArraysCore, Metal, StaticArrays, OffsetArrays, 
      LinearAlgebra

##

L = 4
basis = SolidHarmonics(L)
nbatch = 512

Rs = [ @SVector randn(Float32, 3) for _=1:nbatch ]
Z1 = compute(basis, Rs)

##

Rs_mat = collect(reinterpret(reshape, Float32, Rs))
Rs_gpu = MtlArray(Rs_mat)
Z2 = MtlArray(zeros(Float32, size(Z1)))
Flm_gpu = MtlArray(Float32.(basis.Flm.parent))

SpheriCart.solid_harmonics!(Z2, Val(L), Rs_gpu, Flm_gpu)
Z2_cpu = Matrix(Z2)

norm(Z1 - Z2_cpu, Inf)

##

@time compute!(Z1, basis, Rs)
@time SpheriCart.solid_harmonics!(Z2, Val(L), Rs_gpu, Flm_gpu)
