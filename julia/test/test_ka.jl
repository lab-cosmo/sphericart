
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
Flm_cpu = basis.Flm
Flm_gpu = gpu(Flm_cpu)

Rs = [ @SVector randn(Float32, 3) for _=1:nbatch ]
Rs = [ 𝐫 / norm(𝐫) for 𝐫 in Rs ]
Rs_gpu = gpu(Rs)

# reference calculation 
Z1 = compute(basis, Rs)

##

@info("test GPU-KA execution")
Z2 = gpu(zeros(Float32, size(Z1)))
SpheriCart.solid_harmonics!(Z2, Val(L), Rs_gpu, Flm_gpu, GRPSIZE)
@show norm(Z1 - Array(Z2), Inf) / L^2
@test norm(Z1 - Array(Z2), Inf) / L^2 < 1e-7

##

@info("test CPU-KA execution")
Z3 = zeros(Float32, size(Z1))
SpheriCart.ka_solid_harmonics!(Z3, nothing, Val{L}(), Rs, Flm_cpu)
@show norm(Z1 - Z3, Inf) / L^2
@test norm(Z1 - Z3, Inf) / L^2 < 1e-7


## 

@info("test GPU-KA gradients")
Z1, ∇Z1 = compute_with_gradients(basis, Rs)
Z4 = gpu(zeros(Float32, size(Z1)))
∇Z4 = gpu(zeros(SVector{3, Float32}, size(Z1)))
SpheriCart.solid_harmonics_with_grad!(Z4, ∇Z4, Val{L}(), Rs_gpu, Flm_gpu, GRPSIZE)

@show norm(Z1 - Array(Z4), Inf) / L^2
@show norm(∇Z1 - Array(∇Z4), Inf) / L^3
@test norm(Z1 - Array(Z4), Inf) / L^2 < 1e-7
@test norm(∇Z1 - Array(∇Z4), Inf) / L^3 < 1e-7

##

@info("test CPU-KA gradients")
Z5 = zeros(Float32, size(Z1))
∇Z5 = zeros(SVector{3, Float32}, size(Z1))
SpheriCart.ka_solid_harmonics!(Z5, ∇Z5, Val{L}(), Rs, Flm_cpu)

@show norm(Z1 - Array(Z5), Inf) / L^2
@show norm(∇Z1 - Array(∇Z5), Inf) / L^3
@test norm(Z1 - Array(Z5), Inf) / L^2 < 1e-7
@test norm(∇Z1 - Array(∇Z5), Inf) / L^3 < 1e-7
