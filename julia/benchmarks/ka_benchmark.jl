# Benchmark the KernelAbstractions batched code path (CPU + GPU).
#
# Run, e.g., from this folder in an environment that has `BenchmarkTools`
# (and, for the GPU section, `CUDA` loadable + a functional device):
#
#   julia --project=.. -O3 ka_benchmark.jl
#
# The same kernel serves CPU and GPU; the CPU backend auto-multithreads, so set
# `JULIA_NUM_THREADS` accordingly. Timings are reported per input point (ns/pt).

using SpheriCart, StaticArrays, BenchmarkTools, LinearAlgebra, Printf

# optional GPU backend (CUDA); falls back to CPU-only if unavailable
const HAVE_GPU = try
   @eval using CUDA
   CUDA.functional()
catch
   false
end
to_gpu(x) = HAVE_GPU ? CUDA.cu(x) : x

@info "KA batched benchmark" threads=Threads.nthreads() gpu=(HAVE_GPU ? "CUDA" : "none")

const T  = Float32          # Float32 is the realistic GPU element type
const NX = 32_000

function bench_case(L, nX)
   basis = SolidHarmonics(L; T = T)
   Rs = [ (𝐫 = @SVector randn(T, 3); 𝐫 / norm(𝐫)) for _ = 1:nX ]
   Z  = compute(basis, Rs)
   dZ = similar(Z, SVector{3, T})

   tc  = @belapsed compute!($Z, $basis, $Rs)
   tcg = @belapsed compute_with_gradients!($Z, $dZ, $basis, $Rs)
   @printf("  L=%2d  CPU  val %7.2f ns/pt   grad %7.2f ns/pt\n",
           L, tc/nX*1e9, tcg/nX*1e9)

   if HAVE_GPU
      Rg  = to_gpu(Rs)
      Zg  = compute(basis, Rg)
      dZg = similar(Zg, SVector{3, T})
      tg  = @belapsed CUDA.@sync compute!($Zg, $basis, $Rg)
      tgg = @belapsed CUDA.@sync compute_with_gradients!($Zg, $dZg, $basis, $Rg)
      @printf("        GPU  val %7.3f ns/pt   grad %7.3f ns/pt\n",
              tg/nX*1e9, tgg/nX*1e9)
   end
end

@info "Solid harmonics, T=$T, nX=$NX"
for L in (3, 6, 9, 12)
   bench_case(L, NX)
end
