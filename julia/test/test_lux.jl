using SpheriCart, StaticArrays, LinearAlgebra, Test, Random
using LuxCore
using SpheriCart: SolidHarmonics, SphericalHarmonics,
                  ComplexSolidHarmonics, ComplexSphericalHarmonics, _basis_Flm

@info("============= Testset Lux layers =============")

rng = Random.default_rng()
L  = 5
nX = 16
Rs = [ @SVector randn(3) for _ = 1:nX ]
𝐫  = Rs[1]

##

@info("the harmonics bases implement the LuxCore layer interface")

for basis in (SolidHarmonics(L), SphericalHarmonics(L),
              ComplexSolidHarmonics(L), ComplexSphericalHarmonics(L))
   local ps, st, y1, st1, Y1, stY, y2
   @info("   $(nameof(typeof(basis)))")

   # it is an AbstractLuxLayer with no parameters and `Flm` in the state
   @test basis isa LuxCore.AbstractLuxLayer
   ps, st = LuxCore.setup(rng, basis)
   @test ps == NamedTuple()
   @test keys(st) == (:Flm,)
   @test st.Flm == _basis_Flm(basis)
   @test LuxCore.parameterlength(basis) == 0

   # forward pass reproduces `compute`, single and batched, state unchanged
   y1, st1 = basis(𝐫, ps, st)
   @test y1 ≈ compute(basis, 𝐫)
   @test st1 === st
   Y1, stY = basis(Rs, ps, st)
   @test Y1 ≈ compute(basis, Rs)
   @test stY === st

   # same through LuxCore.apply
   y2, _ = LuxCore.apply(basis, 𝐫, ps, st)
   @test y2 ≈ y1
end

##

@info("the element type can be changed through the state")

let basis = SolidHarmonics(L)
   ps, st = LuxCore.setup(rng, basis)        # Float64 Flm
   st32 = (Flm = Float32.(st.Flm), )         # convert the state to Float32

   𝐫32 = SVector{3, Float32}(𝐫...)
   y32, _ = basis(𝐫32, ps, st32)
   @test eltype(y32) == Float32
   @test y32 ≈ compute(basis, 𝐫)  rtol = 1e-4

   Y32, _ = basis([ SVector{3,Float32}(r...) for r in Rs ], ps, st32)
   @test eltype(Y32) == Float32
   @test Y32 ≈ compute(basis, Rs)  rtol = 1e-4
end
