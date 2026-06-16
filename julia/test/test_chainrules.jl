using SpheriCart, StaticArrays, LinearAlgebra, Test, Random
using ChainRulesCore: rrule, unthunk, NoTangent
using ChainRulesTestUtils
using ForwardDiff
using SpheriCart: SolidHarmonics, SphericalHarmonics,
                  ComplexSolidHarmonics, ComplexSphericalHarmonics

@info("============= Testset ChainRules =============")

# the extension must be active (ChainRulesCore is loaded)
@test Base.get_extension(SpheriCart, :ChainRulesCoreExt) !== nothing

randsphere() = ( 𝐫 = @SVector randn(3); 𝐫 / norm(𝐫) )

##

@info("test_rrule against finite differences (single + batched, all four bases)")

for basis in (SolidHarmonics(5), SphericalHarmonics(5),
              ComplexSolidHarmonics(5), ComplexSphericalHarmonics(5))
   @info("   $(nameof(typeof(basis)))")
   𝐫  = randsphere()
   Rs = [ randsphere() for _ = 1:7 ]
   st = (; Flm = basis.Flm)
   # the rule is on the 3-arg `compute(basis, x, st)`; a 2-arg call differentiates
   # through the optional-argument stub into this same rule.
   test_rrule(compute, basis ⊢ NoTangent(), 𝐫,  st ⊢ NoTangent(); check_inferred = false)
   test_rrule(compute, basis ⊢ NoTangent(), Rs, st ⊢ NoTangent(); check_inferred = false)
end

##

@info("independent ForwardDiff cross-check (real bases, single point)")

for basis in (SolidHarmonics(5), SphericalHarmonics(5))
   len = length(basis)
   st  = (; Flm = basis.Flm)
   for _ = 1:10
      local 𝐫, W, Y, pb, r̄, g
      𝐫 = @SVector randn(3)
      W = randn(len)                          # an arbitrary (real) output cotangent
      Y, pb = rrule(compute, basis, 𝐫, st)
      r̄ = unthunk(pb(W)[3])                   # input cotangent from the rule
      @test Y ≈ compute(basis, 𝐫)
      g = ForwardDiff.gradient(𝐫 -> dot(W, compute(basis, 𝐫, st)), 𝐫)
      @test r̄ ≈ g
   end
end
