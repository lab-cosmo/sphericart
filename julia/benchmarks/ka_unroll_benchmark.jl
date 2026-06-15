#
# Benchmark: looped vs unrolled batched KA kernel, CPU + GPU, val + grad.
# Compares the two kernels back-to-back in the same Julia session.
#
# FINDINGS (A100, Float32, nX=32000, 8 CPU threads; min-time over 3 runs):
#   - @generated-inside-@kernel COMPILES AND RUNS CORRECTLY on CUDA (A100).
#   - the unrolled kernel emits ZERO `.shared` PTX (no @localmem Q/s/c buffer);
#     the looped kernel emits 36 (val) / 56 (grad). Confirmed via PTX.
#   - CPU: unrolled is a clear win for L=6 values (~4.2 vs ~8.6 ns/pt, ~2x)
#     and a modest grad win (~14.5 vs ~18). L=3 roughly neutral.
#   - GPU: unrolled is NEUTRAL for values and SLIGHTLY SLOWER for gradients,
#     despite removing all shared memory. The A100 is not shared-mem bound
#     here; the unrolled body trades shared mem for higher register pressure.
#   Recommendation: enable unrolling on CPU for L in 4..6; do NOT switch the
#   GPU path to unrolled (keep the looped kernel as the GPU default).
#
# Usage (CPU only):
#   JULIA_NUM_THREADS=8 julia -O3 --project=.. ka_unroll_benchmark.jl
# Usage (CPU + GPU, scratch env with CUDA):
#   JULIA_NUM_THREADS=8 julia -O3 --project=/tmp/sph_unroll_env ka_unroll_benchmark.jl
#

using SpheriCart, StaticArrays, LinearAlgebra, BenchmarkTools
using SpheriCart: ka_solid_harmonics!, ka_solid_harmonics_with_grad!, KA_KERNEL_MODE

const HAS_GPU = try
   using CUDA
   CUDA.functional()
catch
   false
end

if HAS_GPU
   using KernelAbstractions
   @info "CUDA functional: $(CUDA.name(CUDA.device()))"
   gpu = CuArray
else
   @info "No GPU; CPU benchmarks only."
end

nX = 32_000
GRP = 32
Ls = [3, 6, 9, 12]

println()
println("nX = $nX, GRPSZ = $GRP, Float32, JULIA_NUM_THREADS = ", Threads.nthreads())
println()

# minimum time is the most stable estimator (least affected by GC / OS noise)
bench_med(f) = (b = @benchmark $f() samples=400 seconds=3.0 evals=1; minimum(b).time)

function run_cpu(L)
   basis = SolidHarmonics(L; T = Float32)
   Rs = [ (r = @SVector randn(Float32,3); r/norm(r)) for _=1:nX ]
   Z = zeros(Float32, nX, SpheriCart.sizeY(L))
   dZ = zeros(SVector{3,Float32}, nX, SpheriCart.sizeY(L))
   res = Dict{Symbol,NTuple{2,Float64}}()
   for mode in (:looped, :unrolled)
      if mode == :unrolled && L > SpheriCart.UNROLL_LMAX
         res[mode] = (NaN, NaN); continue
      end
      KA_KERNEL_MODE[] = mode
      ka_solid_harmonics!(Z, Val{L}(), Rs, basis.Flm)  # warmup
      tval = bench_med(() -> ka_solid_harmonics!(Z, Val{L}(), Rs, basis.Flm))
      ka_solid_harmonics_with_grad!(Z, dZ, Val{L}(), Rs, basis.Flm)
      tgrad = bench_med(() -> ka_solid_harmonics_with_grad!(Z, dZ, Val{L}(), Rs, basis.Flm))
      res[mode] = (tval/nX, tgrad/nX)
   end
   KA_KERNEL_MODE[] = :auto
   return res
end

function run_gpu(L)
   basis = SolidHarmonics(L; T = Float32)
   Rs = [ (r = @SVector randn(Float32,3); r/norm(r)) for _=1:nX ]
   Rsg = gpu(Rs)
   Z = gpu(zeros(Float32, nX, SpheriCart.sizeY(L)))
   dZ = gpu(zeros(SVector{3,Float32}, nX, SpheriCart.sizeY(L)))
   res = Dict{Symbol,NTuple{2,Float64}}()
   for mode in (:looped, :unrolled)
      if mode == :unrolled && L > SpheriCart.UNROLL_LMAX
         res[mode] = (NaN, NaN); continue
      end
      KA_KERNEL_MODE[] = mode
      # warmup
      ka_solid_harmonics!(Z, Val{L}(), Rsg, basis.Flm, GRP); CUDA.synchronize()
      tval = bench_med(() -> (ka_solid_harmonics!(Z, Val{L}(), Rsg, basis.Flm, GRP); CUDA.synchronize()))
      ka_solid_harmonics_with_grad!(Z, dZ, Val{L}(), Rsg, basis.Flm, GRP); CUDA.synchronize()
      tgrad = bench_med(() -> (ka_solid_harmonics_with_grad!(Z, dZ, Val{L}(), Rsg, basis.Flm, GRP); CUDA.synchronize()))
      res[mode] = (tval/nX, tgrad/nX)
   end
   KA_KERNEL_MODE[] = :auto
   return res
end

using Printf

function report(title, runner)
   println("===== $title (ns/point) =====")
   @printf("  %-4s | %-22s | %-22s\n", "L", "looped (val/grad)", "unrolled (val/grad)")
   println("  -----|------------------------|------------------------")
   for L in Ls
      r = runner(L)
      lv, lg = r[:looped]
      uv, ug = r[:unrolled]
      us = isnan(uv) ? "    -   /    -" : @sprintf("%6.2f / %6.2f", uv, ug)
      @printf("  %-4d | %6.2f / %6.2f         | %s\n", L, lv, lg, us)
   end
   println()
end

report("CPU", run_cpu)
if HAS_GPU
   report("GPU (A100)", run_gpu)
end
