using SpheriCart, StaticArrays, LinearAlgebra, Quadmath, Printf, Test


##

@info("============= Testset F32 =============")

n_samples = 1_000 
Rs = [ @SVector randn(3) for _ = 1:n_samples ]
Rs_64 = [ 𝐫/norm(𝐫) for 𝐫 in Rs ]
Rs_32 = [ Float32.(𝐫) for 𝐫 in Rs_64 ]
Rs_128 = [ Float128.(𝐫) for 𝐫 in Rs_64 ]

tol_32 = eps(Float32) * 1e1 
tol_64 = eps(Float64) * 1e1

@printf("  L  |  |Z32-Z64|  |Z64-Z128|  |   |∇Z32-∇Z64|  |∇Z64-∇Z128|  \n")
@printf("     |     / L²        / L²    |      / L³          / L³   \n")
@printf("-----|-------------------------|----------------------------- \n")

for L = 4:4:20
   basis_64 = SolidHarmonics(L; static=false)
   Z_64, ∇Z_64 = compute_with_gradients(basis_64, Rs_64)

   basis_32 = SolidHarmonics(L; static=false, T = Float32)
   Z_32, ∇Z_32 = compute_with_gradients(basis_32, Rs_32)

   basis_128 = SolidHarmonics(L; static=false, T = Float128)
   Z_128, ∇Z_128 = compute_with_gradients(basis_128, Rs_128)

   err_32 = norm(Z_64 - Z_32, Inf) / L^2
   err_64 = norm(Z_64 - Z_128, Inf) / L^2
   ∇err_32 = norm(∇Z_64 - ∇Z_32, Inf) / L^3
   ∇err_64 = norm(∇Z_64 - ∇Z_128, Inf) / L^3

   @printf("  %2d |  %.2e    %.2e   |   %.2e     %.2e  \n", 
            L, err_32, err_64, ∇err_32, ∇err_64)

   @test err_32 < tol_32
   @test err_64 < tol_64
   @test ∇err_32 < tol_32
   @test ∇err_64 < tol_64
end
println()

##

@info("test Float32 type stability (no silent Float64 promotion)")

using SpheriCart: static_solid_harmonics, static_solid_harmonics_with_grads,
                  generate_Flms

# the generated single-point kernels read `Flm` and must produce Float32 with no
# Float64 in the typed code (a Float64 leak would be fatal on Float64-less GPUs
# e.g. Metal). We test the `Flm`-reading method directly (the convenience method
# that builds `Flm` pulls in `generate_Flms`, whose setup arithmetic is Float64).
let L = 6
   𝐫 = SVector{3, Float32}(randn(3)...)
   Flm = SMatrix{L+1, L+1}(generate_Flms(L; T = Float32))
   Z = @inferred static_solid_harmonics(Val(L), 𝐫, Flm)
   @test eltype(Z) == Float32
   Zg, dZg = @inferred static_solid_harmonics_with_grads(Val(L), 𝐫, Flm)
   @test eltype(Zg) == Float32 && eltype(eltype(dZg)) == Float32

   FT = typeof(Flm)
   ct  = code_typed(static_solid_harmonics, (Val{L}, Float32, Float32, Float32, FT); optimize=false)[1][1]
   ctg = code_typed(static_solid_harmonics_with_grads, (Val{L}, Float32, Float32, Float32, FT); optimize=false)[1][1]
   @test count("Float64", string(ct))  == 0
   @test count("Float64", string(ctg)) == 0
end

# the batched (KernelAbstractions) path must also be Float32-clean and type stable
let L = 8
   basis = SolidHarmonics(L; T = Float32)
   Rs = [ SVector{3, Float32}(randn(3)...) for _ = 1:16 ]
   Z = compute(basis, Rs)
   @test eltype(Z) == Float32
   Zg, dZg = compute_with_gradients(basis, Rs)
   @test eltype(Zg) == Float32 && eltype(eltype(dZg)) == Float32
end