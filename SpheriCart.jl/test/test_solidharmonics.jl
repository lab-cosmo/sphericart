
using Test, StaticArrays, LinearAlgebra, Random, SpheriCart
using ForwardDiff
using SpheriCart: compute, compute!, SolidHarmonics, sizeY, 
                  static_solid_harmonics, 
                  compute_with_gradients

##

# This is an implementation that ignores any normalisation factors. 
# Correctness of the implementation will be tested UP TO normalisation. 
# The normalisation will then separately be tested by computing the gramian 
# and confirming that the basis if L2-orthonormal on the sphere. 

# TODO: can we replace this against a generated code? (sympy or similar?)

function symbolic_zlm_4(𝐫)
   x, y, z = tuple(𝐫...)
   r = norm(𝐫)   
   return [ 
      1.0,  # l = 0
      y,    # l = 1
      z, 
      x, 
      x * y,  # l = 2 
      y * z, 
      3 * z^2 - r^2,
      x * z, 
      x^2 - y^2, 
      (3 * x^2 - y^2) * y,   # l = 3
      x * y * z, 
      (5 * z^2 - r^2) * y, 
      (5 * z^2 - 3 * r^2) * z,
      (5 * z^2 - r^2) * x,
      (x^2 - y^2) * z, 
      (x^2 - 3 * y^2) * x, 
      x * y * (x^2 - y^2),    # l = 4 
      y * z * (3 * x^2 - y^2), 
      x * y * (7 * z^2 - r^2), 
      y * z * (7 * z^2 - 3 * r^2), 
      (35 * z^4 - 30 * r^2 * z^2 + 3 * r^4),
      x * z * (7 * z^2 - 3 * r^2),
      (x^2 - y^2) * (7 * z^2 - r^2),
      x * z * (x^2 - 3 * y^2),
      x^2 * (x^2 - 3 * y^2) - y^2 * (3 * x^2 - y^2),
   ]
end

# the code to be tested against the symbolic code above 
# all other implementations will be tested against this. 
zlm_4 = SolidHarmonics(4; static=true)

𝐫0 = @SVector randn(3)
Z1 = zlm_4(𝐫0)
Z2 = symbolic_zlm_4(𝐫0)
F = Z1 ./ Z2

# check that zlm_4 evaluates the right thing. 
@test static_solid_harmonics(Val(4), 𝐫0) == zlm_4(𝐫0)

for ntest = 1:30 
   local 𝐫, Z1, Z2 
   𝐫 = @SVector randn(3)
   Z1 = zlm_4(𝐫)
   Z2 = symbolic_zlm_4(𝐫)
   @test Z1 ≈ Z2 .* F
end

##

@info("confirm that the two implementations are consistent with one another")
for L = 2:10, ntest = 1:10
   local 𝐫, Z1, Z2, basis 
   basis = SolidHarmonics(L)
   basis2 = SolidHarmonics(L; static=false)
   𝐫 = @SVector randn(3)
   Z1 = basis(𝐫)
   Z2 = basis([𝐫,])[:]
   Z3 = basis2(𝐫)
   @test Z1 ≈ Z2 ≈ Z3 
end

##

@info("test the orthogonality on the sphere: G ≈ I")

Random.seed!(0)
L = 3
basis = SolidHarmonics(L)
rand_sphere() = ( (𝐫 = @SVector randn(3)); 𝐫/norm(𝐫) )

for ntest = 1:10
   local Z
   Rs = [ rand_sphere() for _ = 1:10_000 ] 
   Z = compute(basis, Rs)
   G = (Z' * Z) / length(Rs) * 4 * π
   @test norm(G - I) < 0.33
   @test cond(G) < 1.5
end


##

@info("confirm batched evaluation is consistent with single")
for L = 2:10, ntest = 1:10
   local basis, Z1, Z2
   basis = SolidHarmonics(L)
   nbatch = rand(8:20)
   Rs = [ @SVector randn(3) for _=1:nbatch ]
   Z1 = reinterpret(reshape, Float64, 
                     static_solid_harmonics.(Val(L), Rs), )'
   Z2 = compute(basis, Rs)

   @test Z1 ≈ Z2
end

##

@info("test gradients")

zlm_4 = SolidHarmonics(12; static=false)

function fwd_grad(basis, 𝐫)
   Z = basis(𝐫)
   dZ = ForwardDiff.jacobian(basis, 𝐫)'
   return Z, [ SVector{3, eltype(𝐫)}(dZ[:, i]...) for i = 1:length(Z) ]
end

for ntest = 1:30
   local 𝐫0, Z0, Z1, Z2, dZ1, dZ2
   𝐫0 = @SVector randn(3)
   Z0 = zlm_4(𝐫0)
   Z1, dZ1 = compute_with_gradients(zlm_4, 𝐫0)
   Z2, dZ2 = fwd_grad(zlm_4, 𝐫0)

   @test Z0 ≈ Z1 ≈ Z2
   @test dZ1 ≈ dZ2
end

##

@info("test batched gradients")

basis = SolidHarmonics(12; static=false)

for ntest = 1:30
   local nX, Rs, Z0, Z1, dZ1, dZ2
   nX = rand(2:37)
   Rs = [ @SVector randn(3) for _ = 1:nX ]
   Z0 = compute(basis, Rs)
   Z1, dZ1 = compute_with_gradients(basis, Rs)
   dZ2 = vcat([ reshape(compute_with_gradients(basis, 𝐫)[2], 1, :) for 𝐫 in Rs ]...)
   
   @test Z0 ≈ Z1
   @test dZ1 ≈ dZ2
end


##

@info("test generated gradients")

basis_st = SolidHarmonics(12; static=true)
basis_dy = SolidHarmonics(12; static=false)
for ntest = 1:30 
   local 𝐫, Z0, Z1, Z2, dZ1, dZ2
   𝐫 = @SVector randn(3)
   Z0 = basis_st(𝐫)
   Z1, dZ1 = compute_with_gradients(basis_st, 𝐫)
   Z2, dZ2 = compute_with_gradients(basis_dy, 𝐫)

   @test Z1 ≈ Z0 ≈ Z2
   @test dZ1 ≈ dZ2
end
