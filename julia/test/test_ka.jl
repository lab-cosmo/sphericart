
using SpheriCart, GPUArraysCore, StaticArrays,
      LinearAlgebra, KernelAbstractions, Test

# auto-detects a GPU backend; resolves to CPU (dev = identity) on a plain
# runner and installs no GPU package. (Mechanism adapted from EquivariantTensors.)
include(joinpath(@__DIR__, "utils_gpu.jl"))

@info("============= Testset KernelAbstractions =============")

##

L = 8
nbatch = 1024
GRPSIZE = 8  # to be on the safe side; testing just for correctness

@info("L = $L, nbatch = $nbatch")

basis = SolidHarmonics(L; T = Float32)

Rs = [ @SVector randn(Float32, 3) for _=1:nbatch ]
Rs = [ 𝐫 / (norm(𝐫)+2*rand(Float32)) for 𝐫 in Rs ]

# reference calculation (CPU)
Z1 = compute(basis, Rs)

##

@info("test CPU-KA execution")
Z3 = zeros(Float32, size(Z1))
SpheriCart.ka_solid_harmonics!(Z3, nothing, Val{L}(), Rs, basis.Flm)
@show norm(Z1 - Z3, Inf) / L^2
@test norm(Z1 - Z3, Inf) / L^2 < 1e-7

##

@info("test CPU-KA gradients")
Z1, ∇Z1 = compute_with_gradients(basis, Rs)
Z5 = zeros(Float32, size(Z1))
∇Z5 = zeros(SVector{3, Float32}, size(Z1))
SpheriCart.ka_solid_harmonics!(Z5, ∇Z5, Val{L}(), Rs, basis.Flm)

@show norm(Z1 - Array(Z5), Inf) / L^2
@show norm(∇Z1 - Array(∇Z5), Inf) / L^3
@test norm(Z1 - Array(Z5), Inf) / L^2 < 1e-7
@test norm(∇Z1 - Array(∇Z5), Inf) / L^3 < 1e-7

##

# the remaining tests need an actual device; they are skipped on CPU runners
# (the KA kernels themselves are already exercised on the CPU backend above).
if gpu_backend == "CPU"
   @info("No GPU backend detected — skipping device KA tests.")
else

   Rs_gpu = gpu(Rs)

   @info("test GPU-KA execution")
   Z2 = gpu(zeros(Float32, size(Z1)))
   SpheriCart.ka_solid_harmonics!(Z2, Val(L), Rs_gpu, basis.Flm, GRPSIZE)
   @show norm(Z1 - Array(Z2), Inf) / L^2
   @test norm(Z1 - Array(Z2), Inf) / L^2 < 1e-7

   @info("test GPU-KA execution via api")
   Z2a = compute(basis, Rs_gpu)
   @test Z2a isa AbstractGPUArray
   @test norm(Z1 - Array(Z2a), Inf) / L^2 < 1e-7

   ##

   @info("test GPU-KA gradients")
   Z4 = gpu(zeros(Float32, size(Z1)))
   ∇Z4 = gpu(zeros(SVector{3, Float32}, size(Z1)))
   SpheriCart.ka_solid_harmonics_with_grad!(Z4, ∇Z4, Val{L}(), Rs_gpu, basis.Flm, GRPSIZE)

   Z4a, ∇Z4a = compute_with_gradients(basis, Rs_gpu)
   @test Z4a isa AbstractGPUArray
   @test ∇Z4a isa AbstractGPUArray

   @show norm(Z1 - Array(Z4), Inf) / L^2
   @show norm(∇Z1 - Array(∇Z4), Inf) / L^3
   @test norm(Z1 - Array(Z4), Inf) / L^2 < 1e-7
   @test norm(∇Z1 - Array(∇Z4), Inf) / L^3 < 1e-7
   @test norm(Z1 - Array(Z4a), Inf) / L^2 < 1e-7
   @test norm(∇Z1 - Array(∇Z4a), Inf) / L^3 < 1e-7

   ##

   @info("Test KernelAbstractions for spherical harmonics")

   sh = SphericalHarmonics(L; T = Float32)
   R̂s = [ 𝐫 / norm(𝐫) for 𝐫 in Rs ]
   R̂s_gpu = gpu(R̂s)

   Y1a = compute(basis, R̂s_gpu)
   Y1b = compute(sh, Rs)
   Y2 = compute(sh, Rs_gpu)
   @test Y2 isa AbstractGPUArray
   @test norm(Array(Y1a) - Array(Y2), Inf) / L^2 < 1e-7
   @test norm(Y1b - Array(Y2), Inf) / L^2 < 1e-7

   ##

   Y3, ∇Y3 = compute_with_gradients(sh, Rs)
   Y4, ∇Y4 = compute_with_gradients(sh, Rs_gpu)
   @test Y4 isa AbstractGPUArray
   @test ∇Y4 isa AbstractGPUArray
   @test norm(Y3 - Array(Y4), Inf) / L^2 < 1e-7
   @test norm(∇Y3 - Array(∇Y4), Inf) / L^3 < 1e-7

end
