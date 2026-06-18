using SpheriCart, ACEbase, StaticArrays, LinearAlgebra, Test
using SpheriCart: SolidHarmonics, SphericalHarmonics,
                  ComplexSolidHarmonics, ComplexSphericalHarmonics,
                  compute, compute_with_gradients, lm2idx, sizeY
using ACEbase: evaluate, evaluate_ed, evaluate!, evaluate_ed!, natural_indices

##

# reference: original P4ML / ACE complex spherical harmonics up to L = 3,
# L2-normalised on the sphere.
function explicit_shs(θ, φ)
   Y00 = 0.5 * sqrt(1/π)
   Y1m1 = 0.5 * sqrt(3/(2*π))*sin(θ)*exp(-im*φ)
   Y10 = 0.5 * sqrt(3/π)*cos(θ)
   Y11 = -0.5 * sqrt(3/(2*π))*sin(θ)*exp(im*φ)
   Y2m2 = 0.25 * sqrt(15/(2*π))*sin(θ)^2*exp(-2*im*φ)
   Y2m1 = 0.5 * sqrt(15/(2*π))*sin(θ)*cos(θ)*exp(-im*φ)
   Y20 = 0.25 * sqrt(5/π)*(3*cos(θ)^2 - 1)
   Y21 = -0.5 * sqrt(15/(2*π))*sin(θ)*cos(θ)*exp(im*φ)
   Y22 = 0.25 * sqrt(15/(2*π))*sin(θ)^2*exp(2*im*φ)
   Y3m3 = 1/8 * exp(-3 * im * φ) * sqrt(35/π) * sin(θ)^3
   Y3m2 = 1/4 * exp(-2 * im * φ) * sqrt(105/(2*π)) * cos(θ) * sin(θ)^2
   Y3m1 = 1/8 * exp(-im * φ) * sqrt(21/π) * (-1 + 5 * cos(θ)^2) * sin(θ)
   Y30 = 1/4 * sqrt(7/π) * (-3 * cos(θ) + 5 * cos(θ)^3)
   Y31 = -(1/8) * exp(im * φ) * sqrt(21/π) * (-1 + 5 * cos(θ)^2) * sin(θ)
   Y32 = 1/4 * exp(2 * im * φ) * sqrt(105/(2*π)) * cos(θ) * sin(θ)^2
   Y33 = -(1/8) * exp(3 * im * φ) * sqrt(35/π) * sin(θ)^3
   return [Y00, Y1m1, Y10, Y11, Y2m2, Y2m1, Y20, Y21, Y22,
           Y3m3, Y3m2, Y3m1, Y30, Y31, Y32, Y33]
end

# complex harmonics built from the real ones (the R2C convention)
function cY_from_rY(Yr, LMAX)
   Yc = zeros(Complex{eltype(Yr)}, length(Yr))
   for l = 0:LMAX
      Yc[lm2idx(l, 0)] = Yr[lm2idx(l, 0)]
      for m = 1:l
         i⁺ = lm2idx(l, m); i⁻ = lm2idx(l, -m)
         Yc[i⁺] = (-1)^m * (Yr[i⁺] + im * Yr[i⁻]) / sqrt(2)
         Yc[i⁻] =          (Yr[i⁺] - im * Yr[i⁻]) / sqrt(2)
      end
   end
   return Yc
end

_rand_sphere() = ( u = (@SVector randn(3)); u ./ norm(u) )
function _rand_angles()
   θ = rand() * π; φ = (rand()-0.5) * 2*π
   return SVector(sin(θ)*cos(φ), sin(θ)*sin(φ), cos(θ)), θ, φ
end

##

@testset "complex harmonics vs reference" begin
   r_spher = SphericalHarmonics(3)
   c_spher = ComplexSphericalHarmonics(3)
   c_solid = ComplexSolidHarmonics(3)
   for ntest = 1:30
      𝐫, θ, φ = _rand_angles()
      Yref = explicit_shs(θ, φ)
      Yc1 = cY_from_rY(r_spher(𝐫), 3)
      @test Yc1 ≈ Yref
      @test c_spher(𝐫) ≈ Yref
      @test c_solid(𝐫) ≈ Yref     # on the unit sphere solid == spherical
   end
end

@testset "complex == R2C(real), batched" begin
   for (RB, CB) in [ (SolidHarmonics(6), ComplexSolidHarmonics(6)),
                     (SphericalHarmonics(6), ComplexSphericalHarmonics(6)) ]
      R = [ (@SVector randn(3)) for _ = 1:20 ]
      Yr = RB(R); Yc = CB(R)
      for j = 1:length(R)
         @test Yc[j, :] ≈ cY_from_rY(Yr[j, :], 6)
      end
   end
end

##

@testset "ACEbase interface == compute" begin
   bases = [ SolidHarmonics(5), SphericalHarmonics(5),
             ComplexSolidHarmonics(5), ComplexSphericalHarmonics(5) ]
   for basis in bases
      𝐫 = _rand_sphere()
      R = [ _rand_sphere() for _ = 1:13 ]
      @test evaluate(basis, 𝐫) == compute(basis, 𝐫)
      @test evaluate(basis, R) == compute(basis, R)
      @test evaluate(basis, R, nothing, nothing) == compute(basis, R)
      Yc, dYc = compute_with_gradients(basis, R)
      Ye, dYe = evaluate_ed(basis, R)
      @test Yc == Ye && dYc == dYe
      # in-place
      Y = similar(compute(basis, R))
      @test evaluate!(Y, basis, R) == compute(basis, R)
   end
end

@testset "natural_indices" begin
   for L in (0, 3, 6)
      for basis in (SolidHarmonics(L), ComplexSphericalHarmonics(L))
         spec = natural_indices(basis)
         @test length(spec) == sizeY(L) == length(basis)
         @test spec[1] == (l = 0, m = 0)
         @test all( lm2idx(spec[i].l, spec[i].m) == i for i = 1:length(spec) )
      end
   end
end

##

@testset "finite-difference gradients" begin
   bases = [ SolidHarmonics(8), SphericalHarmonics(8),
             ComplexSolidHarmonics(7), ComplexSphericalHarmonics(7) ]
   for basis in bases
      𝐫 = _rand_sphere() * (0.5 + rand())
      Y, ∇Y = evaluate_ed(basis, 𝐫)
      @test Y ≈ evaluate(basis, 𝐫)
      u = _rand_sphere()
      errs = Float64[]
      for p = 2:10
         h = 0.1^p
         dh = (evaluate(basis, 𝐫 + h*u) .- evaluate(basis, 𝐫 - h*u)) ./ (2h)
         # note: plain (non-conjugating) contraction, since Y may be complex
         da = [ sum(∇Y[i] .* u) for i = 1:length(Y) ]
         push!(errs, norm(dh .- da, Inf))
      end
      @test minimum(errs) < 1e-6
   end
end
