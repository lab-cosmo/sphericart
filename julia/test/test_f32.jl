using SpheriCart, StaticArrays, LinearAlgebra

n_samples = 100_000 

for L = 4:2:12
   basis = SolidHarmonics(L; static=false)
   Rs = [ @SVector randn(3) for _ = 1:n_samples ]
   Z = compute(basis, Rs)

   basis_f32 = SolidHarmonics(L; static=false, T = Float32)
   Rs_f32 = [ Float32.(𝐫) for 𝐫 in Rs]
   Z_f32 = compute(basis_f32, Rs_f32)

   @info(L)
   @show norm(Z - Z_f32, Inf)
   @show norm((Z - Z_f32) ./ (1 .+ abs.(Z)), Inf)
end
