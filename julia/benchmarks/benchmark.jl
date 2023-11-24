# to run this benchmark add `BenchmarkTools` to the standard julia 
# environment (e.g. `julia` and `] add BenchmarkTools`)
# then run this scrict via from the current folder via 
# `julia --project=.. -O3 benchmark.jl`

# This is just a preliminary benchmark for initial testing. 
# We need to write a proper benchmark suite to check performance 
# regressions and compare against the C++ implementation on the same 
# system. 

using StaticArrays, BenchmarkTools, SpheriCart
using SpheriCart: SolidHarmonics, compute, compute!, 
                  static_solid_harmonics, 
                  compute_with_gradients, compute_with_gradients!

##

@info("static_solid_harmonics")
𝐫 = @SVector randn(3)
for L = 1:12
   @show L
   basis = SolidHarmonics(L; static=true)
   basis(𝐫) # warmup 
   @btime ($basis)($𝐫)
   # these two are equivalent - just keeping them here for testing 
   # since there is an odd effect that in some environments there there 
   # is an unexplained overhead in the `compute` interface. 
   # @btime static_solid_harmonics($(Val(L)), $𝐫)
   # @btime compute($basis, $𝐫)
end


##

@info("batched evaluation vs broadcast")
@info("nX = 32 (try a nice number)")
@info("broadcast! cost is almost exactly single * nX")

Rs = [ (@SVector randn(3)) for _ = 1:32 ]

for L = 3:3:15
   @show L
   basis = SolidHarmonics(L)
   Zs = static_solid_harmonics.(Val(L), Rs)
   print("    single: "); @btime static_solid_harmonics($(Val(L)), $(Rs[1]))
   print("broadcast!: "); @btime broadcast!(𝐫 -> static_solid_harmonics($(Val(L)), 𝐫), $Zs, $Rs)
   Zb = compute(basis, Rs)
   print("   batched: "); (@btime compute!($Zb, $basis, $Rs));
end

##

@info("static vs generic basis for single input (nX = 1)")
@info("  this shouws that the generic single-input implementation needs work")

for L = 3:3:30 
   local 𝐫
   @show L 
   𝐫 = @SVector randn(3)
   basis_st = SolidHarmonics(L; static=true)
   basis_dy = SolidHarmonics(L; static=false)
   print("  static: "); @btime compute($basis_st, $𝐫)
   print(" generic: "); @btime compute($basis_dy, $𝐫)
end 

##


@info("compute! vs compute_with_gradients! (using 32 inputs)")

for L = 1:2:15
   local Rs
   @show L 
   nX = 32
   basis = SolidHarmonics(L)
   Rs = [ (@SVector randn(3)) for _ = 1:nX ]
   Z, dZ = compute_with_gradients(basis, Rs)
   print("        compute!: "); @btime compute!($Z, $basis, $Rs)
   print(" _with_gradient!: "); @btime compute_with_gradients!($Z, $dZ, $basis, $Rs)
end 


##


@info("compute vs compute_with_gradients for code-generated basis")
for L = 1:10
   local 𝐫
   @show L 
   basis = SolidHarmonics(L; static=true)
   𝐫 = @SVector randn(3)
   Z1 = compute(basis, 𝐫)
   Z, dZ = compute_with_gradients(basis, 𝐫)
   print("        compute: "); @btime compute($basis, $𝐫)
   print(" _with_gradient: "); @btime compute_with_gradients($basis, $𝐫)
end 



## -------------- 

# @profview let Z = Z, uf_Zlm = uf_Zlm, XX = XX
#    for n = 1:3_000_000
#       evaluate!(Z, uf_Zlm, XX)
#    end
# end