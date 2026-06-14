import Pkg

if !isdefined(Main, :___SPHERICART_UTILS_GPU___)

   """
       detect_gpu_backend() -> String

   Pick a GPU backend by probing the *system* — no GPU package is loaded here,
   so a plain CI runner resolves to `"CPU"` and installs no GPU package. Set the
   `TEST_BACKEND` env var to force a choice (`"CPU"`, `"CUDA"`, `"AMDGPU"`,
   `"Metal"`, `"oneAPI"`).
   """
   function detect_gpu_backend()
      haskey(ENV, "TEST_BACKEND") && return ENV["TEST_BACKEND"]   # manual override
      if Sys.isapple() && Sys.ARCH == :aarch64
         return "Metal"
      elseif !isnothing(Sys.which("nvidia-smi")) && success(`nvidia-smi`)
         return "CUDA"
      elseif !isnothing(Sys.which("rocm-smi")) || isdir("/dev/kfd")
         return "AMDGPU"
      elseif !isnothing(Sys.which("sycl-ls"))   # crude oneAPI probe
         return "oneAPI"
      else
         return "CPU"
      end
   end

   # When a GPU is detected, install the matching backend *into the (sandboxed)
   # test env* and use it; a plain CI runner resolves to "CPU" and installs
   # nothing. A detected-but-unusable backend (e.g. Metal on a GPU-less macOS
   # VM) degrades to CPU with a warning so the suite still runs. `gpu`/`dev`
   # transfer arrays host -> device (identity on CPU).
   global gpu_backend = detect_gpu_backend()
   global gpu = global dev = identity

   if gpu_backend != "CPU"
      try
         Pkg.add(gpu_backend)                  # into the sandboxed test env only
         @eval using $(Symbol(gpu_backend))
         if gpu_backend == "CUDA"
            @assert CUDA.functional();   global gpu = global dev = CUDA.CuArray
         elseif gpu_backend == "Metal"
            @assert Metal.functional();  global gpu = global dev = Metal.MtlArray
         elseif gpu_backend == "AMDGPU"
            @assert AMDGPU.functional(); global gpu = global dev = AMDGPU.ROCArray
         elseif gpu_backend == "oneAPI"
            @assert oneAPI.functional(); global gpu = global dev = oneAPI.oneArray
         else
            error("unknown TEST_BACKEND = $(gpu_backend)")
         end
         @info "GPU test backend: $(gpu_backend)"
      catch e
         @warn "GPU backend '$(gpu_backend)' detected but not usable; using CPU." exception=(e, catch_backtrace())
         global gpu_backend = "CPU"
         global gpu = global dev = identity
      end
   end

   gpu_backend == "CPU" && @info "GPU test backend: CPU (dev = identity)."

   global ___SPHERICART_UTILS_GPU___ = true

end
