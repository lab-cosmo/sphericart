
using SpheriCart, GPUArraysCore, StaticArrays,  
      LinearAlgebra, KernelAbstractions, Test 

@info("============= Testset KernelAbstractions =============")

##

using CUDA, Metal, AMDGPU 
if CUDA.functional()
	@info "Using CUDA"
	CUDA.versioninfo()
	gpu = CuArray
elseif AMDGPU.functional()
	@info "Using AMD"
	AMDGPU.versioninfo()
	gpu = ROCArray
elseif Metal.functional()
	@info "Using Metal"
	Metal.versioninfo()
	gpu = MtlArray
else
    @info "No GPU is available. Using CPU."
    gpu = Array
end


##

L = 8
nbatch = 32_000
GRPSIZE = 8  # to be on the safe side; testing just for correctness

@info("L = $L, nbatch = $nbatch")


basis = SolidHarmonics(L; T = Float32)

Rs = [ @SVector randn(Float32, 3) for _=1:nbatch ]
Rs = [ 𝐫 / norm(𝐫) for 𝐫 in Rs ]
Rs_gpu = gpu(Rs)

# reference calculation 
Z1 = compute(basis, Rs)

##

@info("test GPU-KA execution")
Z2 = gpu(zeros(Float32, size(Z1)))
SpheriCart.ka_solid_harmonics!(Z2, Val(L), Rs_gpu, basis.Flm, GRPSIZE)
@show norm(Z1 - Array(Z2), Inf) / L^2
@test norm(Z1 - Array(Z2), Inf) / L^2 < 1e-7

@info("test GPU-KA execution via api")
Z2a = compute(basis, Rs_gpu)
@test norm(Z1 - Array(Z2a), Inf) / L^2 < 1e-7

##

@info("test CPU-KA execution")
Z3 = zeros(Float32, size(Z1))
SpheriCart.ka_solid_harmonics!(Z3, nothing, Val{L}(), Rs, basis.Flm)
@show norm(Z1 - Z3, Inf) / L^2
@test norm(Z1 - Z3, Inf) / L^2 < 1e-7


## 

@info("test GPU-KA gradients")
Z1, ∇Z1 = compute_with_gradients(basis, Rs)
Z4 = gpu(zeros(Float32, size(Z1)))
∇Z4 = gpu(zeros(SVector{3, Float32}, size(Z1)))
SpheriCart.ka_solid_harmonics_with_grad!(Z4, ∇Z4, Val{L}(), Rs_gpu, basis.Flm, GRPSIZE)

Z4a, ∇Z4a = compute_with_gradients(basis, Rs_gpu)


@show norm(Z1 - Array(Z4), Inf) / L^2
@show norm(∇Z1 - Array(∇Z4), Inf) / L^3
@test norm(Z1 - Array(Z4), Inf) / L^2 < 1e-7
@test norm(∇Z1 - Array(∇Z4), Inf) / L^3 < 1e-7
@test norm(Z1 - Array(Z4a), Inf) / L^2 < 1e-7
@test norm(∇Z1 - Array(∇Z4a), Inf) / L^3 < 1e-7

##

@info("test CPU-KA gradients")
Z5 = zeros(Float32, size(Z1))
∇Z5 = zeros(SVector{3, Float32}, size(Z1))
SpheriCart.ka_solid_harmonics!(Z5, ∇Z5, Val{L}(), Rs, basis.Flm)

@show norm(Z1 - Array(Z5), Inf) / L^2
@show norm(∇Z1 - Array(∇Z5), Inf) / L^3
@test norm(Z1 - Array(Z5), Inf) / L^2 < 1e-7
@test norm(∇Z1 - Array(∇Z5), Inf) / L^3 < 1e-7

##

# --- unrolled (small-L @generated) KA kernel ---
# Force the unrolled kernel and check it matches the reference on both
# CPU and GPU, values and gradients, for L = 1 .. UNROLL_LMAX.

@info("test unrolled small-L KA kernel (CPU + GPU, val + grad)")
using SpheriCart: KA_KERNEL_MODE
for Lu = 1:SpheriCart.UNROLL_LMAX
   basis_u = SolidHarmonics(Lu; T = Float32)
   Rs_u = [ 𝐫 / norm(𝐫) for 𝐫 in [ @SVector randn(Float32, 3) for _=1:2000 ] ]
   Zr, ∇Zr = compute_with_gradients(basis_u, Rs_u)

   KA_KERNEL_MODE[] = :unrolled
   try
      # CPU
      Zc = zeros(Float32, size(Zr)); ∇Zc = zeros(SVector{3,Float32}, size(Zr))
      SpheriCart.ka_solid_harmonics!(Zc, Val{Lu}(), Rs_u, basis_u.Flm)
      SpheriCart.ka_solid_harmonics_with_grad!(Zc, ∇Zc, Val{Lu}(), Rs_u, basis_u.Flm)
      @test norm(Zr - Zc, Inf) / Lu^2 < 1e-5
      @test norm(∇Zr - ∇Zc, Inf) / Lu^3 < 1e-5

      # GPU
      Rsg = gpu(Rs_u)
      Zg = gpu(zeros(Float32, size(Zr))); ∇Zg = gpu(zeros(SVector{3,Float32}, size(Zr)))
      SpheriCart.ka_solid_harmonics!(Zg, Val{Lu}(), Rsg, basis_u.Flm, GRPSIZE)
      SpheriCart.ka_solid_harmonics_with_grad!(Zg, ∇Zg, Val{Lu}(), Rsg, basis_u.Flm, GRPSIZE)
      @test norm(Zr - Array(Zg), Inf) / Lu^2 < 1e-5
      @test norm(∇Zr - Array(∇Zg), Inf) / Lu^3 < 1e-5
   finally
      KA_KERNEL_MODE[] = :auto
   end
end
