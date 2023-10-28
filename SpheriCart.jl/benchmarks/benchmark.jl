
using StaticArrays, BenchmarkTools, SpheriCart
using SpheriCart: SolidHarmonics, compute, compute!
using StrideArrays: PtrArray

##

@info("static_solid_harmonics")
@info("NOTE: bizarrely the nice api has an overhead only for a few L values 5..10")
𝐫 = @SVector randn(3)
for L = 1:12
   @show L
   basis = SolidHarmonics(L; static=true)
   basis(𝐫)
   @btime static_solid_harmonics($(Val(L)), $𝐫)
   @btime ($basis)($𝐫)
   @btime compute($basis, $𝐫)
end


##

@info("batched evaluation vs broadcast")
@info("nX = 32 (try a nice number)")

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
   @show L 
   𝐫 = @SVector randn(3)
   basis_st = SolidHarmonics(L; static=true)
   basis_dy = SolidHarmonics(L; static=false)
   print("  static: "); @btime compute($basis_st, $𝐫)
   # static_solid_harmonics($(Val(L)), $𝐫)
   print(" dynamic: "); @btime compute($basis_dy, $𝐫)
end 


## -------------- 

# @profview let Z = Z, uf_Zlm = uf_Zlm, XX = XX
#    for n = 1:3_000_000
#       evaluate!(Z, uf_Zlm, XX)
#    end
# end