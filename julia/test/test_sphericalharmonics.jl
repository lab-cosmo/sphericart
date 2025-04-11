

using SpheriCart, StaticArrays, LinearAlgebra, Test, ForwardDiff
using SpheriCart: idx2lm

@info("============= Testset Spherical Harmonics =============")

##

@info("test consistency of spherical and solid harmonics")

L = 12
spheri = SphericalHarmonics(L)
solids = SolidHarmonics(L)

for ntest = 1:30 
   local 𝐫, r, 𝐫̂, Z1, Z2, ll, rˡ
   𝐫 = @SVector randn(3)
   r = norm(𝐫)
   𝐫̂ = 𝐫 / r 

   Z1 = spheri(𝐫)
   Z2 = solids(𝐫̂)
   @test Z1 ≈ Z2 

   ll = [ idx2lm(i)[1] for i = 1:length(Z1) ]
   rˡ = [ r^l for l = ll ]
   @test Z1 .* rˡ ≈ solids(𝐫)
end 


##

@info("test consistency of batched spherical harmonics")

for ntest = 1:30 
   local Rs, Z1, Z2
   Rs = [ @SVector randn(3) for _ = 1:rand(2:37) ]
   Z1 = spheri(Rs)
   Z2 = 0 * copy(Z1)
   compute!(Z2, spheri, Rs)
   Z3 = vcat([ reshape(spheri(𝐫), 1, :) for 𝐫 in Rs ]...)
   @test Z1 ≈ Z2 ≈ Z3
end

##

@info("test gradients")

function fwd_grad_1(basis, 𝐫)
   _part3(TV, i) = ForwardDiff.Partials(ntuple(j -> one(TV) * (i==j), 3))
   _dual(𝐫::SVector{N, T}) where {N, T} = 
               SVector{N}( ntuple(j -> ForwardDiff.Dual(𝐫[j], _part3(T, j)), N)... )

   𝐫d = _dual(𝐫)               
   Zd = basis(𝐫d)
   Z = ForwardDiff.value.(Zd)
   ∇Z = SVector{3}.(ForwardDiff.partials.(Zd))
   return Z, ∇Z
end

for ntest = 1:30 
   local 𝐫, Y1, Y2, ∇Y1, ∇Y2
   𝐫 = @SVector randn(3)
   Y1, ∇Y1 = compute_with_gradients(spheri, 𝐫)
   Y2, ∇Y2 = fwd_grad_1(spheri, 𝐫)
   @test Y1 ≈ Y2
   @test ∇Y1 ≈ ∇Y2
end 

##

@info("test batched gradients")

for ntest = 1:30 
   local nX, Rs, Y1, Y2, ∇Y1, ∇Y2
   nX = rand(2:37)
   Rs = [ @SVector randn(3) for _ = 1:nX ]
   Y1, ∇Y1 = compute_with_gradients(spheri, Rs)
   Y2 = copy(Y1); ∇Y2 = copy(∇Y1)
   compute_with_gradients!(Y2, ∇Y2, spheri, Rs)
   ∇Y3 = vcat([ reshape(compute_with_gradients(spheri, 𝐫)[2], 1, :) for 𝐫 in Rs ]...)
   @test Y1 ≈ Y2
   @test ∇Y1 ≈ ∇Y2 ≈ ∇Y3
end
