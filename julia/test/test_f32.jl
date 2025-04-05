using SpheriCart, StaticArrays, LinearAlgebra, Quadmath, Printf 


##

n_samples = 100_000 
Rs = [ @SVector randn(3) for _ = 1:n_samples ]

@printf("  L  |  |Z32-Z64|  |Z64-Z128|  |   |∇Z32-∇Z64|  |∇Z64-∇Z128|  \n")
@printf("-----|-------------------------|----------------------------- \n")

for L = 4:4:20
   basis = SolidHarmonics(L; static=false)
   
   Z, ∇Z = compute_with_gradients(basis, Rs)

   basis_f32 = SolidHarmonics(L; static=false, T = Float32)
   Rs_f32 = [ Float32.(𝐫) for 𝐫 in Rs]
   Z_f32, ∇Z_f32 = compute_with_gradients(basis_f32, Rs_f32)

   basis_quad = SolidHarmonics(L; static=false, T = Float128)
   Rs_quad = [ Float128.(𝐫) for 𝐫 in Rs]
   Z_quad, ∇Z_quad = compute_with_gradients(basis_quad, Rs_quad)

   @printf("  %2d |  %.2e    %.2e   |   %.2e     %.2e  \n", 
            L, 
            norm(Z - Z_f32, Inf), 
            norm(Z_f32 - Z_quad, Inf), 
            norm(∇Z - ∇Z_f32, Inf), 
            norm(∇Z_f32 - ∇Z_quad, Inf) )
end

## 

function compute_Q(L, Rs64, T)
   Rs = [ T.(𝐫) for 𝐫 in Rs64 ]
   basis = SolidHarmonics(L; static=false, T = T)
   nX = length(Rs)
   len = SpheriCart.sizeY(L)
   Z = zeros(T, nX, len)
   # ∇Z = zeros(SVector{3, T}, nX, len)
   
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

# n_samples = 100_000 
# Rs = [ @SVector randn(3) for _ = 1:n_samples ]

@printf("  L   |Q32-Q64|  |Q64-Q128| \n")
@printf("----------------------------\n")
for L = 4:4:20
   Q_32 = compute_Q(L, Rs, Float32)
   Q_64 = compute_Q(L, Rs, Float64)
   Q_128 = compute_Q(L, Rs, Float128)

   @printf(" %2d   %.2e   %.2e \n", 
            L, 
            norm(Q_32 - Q_64, Inf), 
            norm(Q_64 - Q_128, Inf) )
end