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

function compute_Q(L, Rs::Vector{SVector{3, T}}) where {T} 
   basis = SolidHarmonics(L; static=false, T = T)
   nX = length(Rs)
   len = SpheriCart.sizeY(L)
   Z = zeros(T, nX, len)
   
   # allocate temporary arrays from an array cache 
   temps = (x = zeros(T, nX), 
            y = zeros(T, nX),
            z = zeros(T, nX), 
           r² = zeros(T, nX),
            s = zeros(T, nX, L+1), 
            c = zeros(T, nX, L+1),
            Q = zeros(T, nX, SpheriCart.sizeY(L)),
          Flm = basis.Flm )

   SpheriCart.solid_harmonics!(Z, Val{L}(), Rs, temps)          
             
   return temps.Q 
end


##

# Relative errors for Q 

# @printf("  L   |Q32-Q64|  |Q64-Q128| \n")
# @printf("----------------------------\n")
# for L = 4:4:20
#    Q_32  = compute_Q(L, Rs_32)
#    Q_64  = compute_Q(L, Rs_64)
#    Q_128 = compute_Q(L, Rs_128)

#    err_32 = norm(Q_32 - Q_64, Inf) / (1+norm(Q_64, Inf))
#    err_64 = norm(Q_64 - Q_128, Inf) / (1+norm(Q_128, Inf))

#    @printf(" %2d   %.2e   %.2e \n",  L, err_32, err_64)

#    # @test err_32 / L^2 < tol_32
#    # @test err_64 / L^2 < tol_64
# end