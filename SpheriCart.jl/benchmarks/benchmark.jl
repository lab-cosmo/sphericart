
using UltraFastACE, StaticArrays, BenchmarkTools, Test
using UltraFastACE.SpheriCart: ZlmBasis, static_solid_harmonics, evaluate!, evaluate 
using ACEbase.Testing: print_tf, println_slim 
using StrideArrays: PtrArray
using LoopVectorization
import Polynomials4ML

##

@info("static_solid_harmonics")
𝐫 = @SVector randn(3)
for L = 1:6 
   @show L
   static_solid_harmonics(Val(L), 𝐫)
   @btime static_solid_harmonics($(Val(L)), $𝐫)
end

##

@info("batched evaluation vs broadcast")
@info("nX = 32 (try a nice number)")

Rs = [ (@SVector randn(3)) for _ = 1:32 ]

for L = 3:3:15
   @show L
   basis = ZlmBasis(L)
   Zs = static_solid_harmonics.(Val(L), Rs)
   print("    single: "); @btime static_solid_harmonics($(Val(L)), $(Rs[1]))
   print("broadcast!: "); @btime broadcast!(𝐫 -> static_solid_harmonics($(Val(L)), 𝐫), $Zs, $Rs)
   Zb = evaluate(basis, Rs)
   print("   batched: "); (@btime evaluate!($Zb, $basis, $Rs));
end

## -------------- 

# @profview let Z = Z, uf_Zlm = uf_Zlm, XX = XX
#    for n = 1:3_000_000
#       evaluate!(Z, uf_Zlm, XX)
#    end
# end