# SpheriCart 

A Julia implementation of real solid and spherical harmonics, following
```quote
Fast evaluation of spherical harmonics with sphericart, 
Filippo Bigi, Guillaume Fraux, Nicholas J. Browning and Michele Ceriotti; 
J. Chem. Phys. 159, 064802 (2023); arXiv:2302.08381
```

`SpheriCart.jl` is released under MIT license and under Apache 2.0 license.

## Installation 

Install the package by opening a REPL, switch to the package manager by typing `]` and then `add SpheriCart`.

## Basic Usage

Real solid and spherical harmonics are evaluated through a single interface
(`compute`, `compute!`, `compute_with_gradients`, `compute_with_gradients!`),
backed by two implementations:
- a fully unrolled, **generated** implementation for a single `𝐫::SVector{3, T}`
  input, returning the harmonics as an `SVector`. This is the fastest option for
  single inputs and is used by default for small `L` (see the `static` keyword).
- a [KernelAbstractions](https://github.com/JuliaGPU/KernelAbstractions.jl)
  **batched** implementation for collections of inputs. The same kernel runs on
  the CPU (multi-threaded across the batch when Julia is started with several
  threads) and on CUDA, AMD, Apple and Intel GPU devices — just pass a GPU array
  of inputs to `compute` / `compute!`. It works for any real element type
  (`Float32`, `Float64`, `Float128`, ...).

For single inputs the generated implementation is far superior in performance;
for batches use the batched path, which is what `compute(basis, Rs)` and the
in-place `compute!` call.


```julia
using SpheriCart, StaticArrays 

# generate the basis object 
L = 5
basis = SolidHarmonics(L)
# Replace this with 
#  basis = SphericalHarmonics(L) 
# to evaluate the spherical instead of solid harmonics 

# evaluate for a single input 
𝐫 = @SVector randn(3) 
# Z : SVector of length (L+1)²
Z = basis(𝐫)  
Z = compute(basis, 𝐫)
# ∇Z : SVector of length (L+1)², each ∇Z[i] is an SVector{3, T}
Z, ∇Z = compute_with_gradients(basis, 𝐫)

# evaluate for many inputs 
nX = 32
Rs = [ @SVector randn(3)  for _ = 1:nX ]
# Z : Matrix of size nX × (L+1)² of scalar 
# dZ : Matrix of size nX × (L+1)² of SVector{3, T}
Z = basis(Rs)  
Z = compute(basis, Rs)
Z, ∇Z = compute_with_gradients(basis, Rs)

# in-place evaluation to avoid the allocation 
compute!(Z, basis, Rs)
compute_with_gradients!(Z, ∇Z, basis, Rs)
```

Note that Julia uses column-major indexing, which means that for batched output the loop over inputs is contiguous in memory. 

### GPU evaluation

Batched evaluation runs on the GPU through the same functions — move the inputs
to the device and call `compute` / `compute!` as usual:

```julia
using CUDA   # or AMDGPU / Metal / oneAPI

Rs_gpu = CuArray(Rs)
Z_gpu = compute(basis, Rs_gpu)              # returns a GPU array
Z_gpu, ∇Z_gpu = compute_with_gradients(basis, Rs_gpu)
```

`SphericalHarmonics` is supported on the GPU as well.

### The `static` keyword

`SolidHarmonics(L; static = (L <= 6))` (likewise `SphericalHarmonics`) selects
whether **single-input** evaluation uses the unrolled generated code (an
`SVector` output, fastest for small `L`, with a compile/stack footprint that
grows with `L`) or falls back to the batched kernel. Batched evaluation always
uses the KernelAbstractions kernel regardless of this flag. Raise the threshold
if you evaluate single inputs at larger `L` in a hot loop.

## Alternative interface (ACEbase / ACEsuit)

Loading `ACEbase` alongside `SpheriCart` activates a package extension that
exposes the harmonics through the standard ACEsuit evaluation interface — the
same `evaluate` / `evaluate_ed` / `evaluate!` / `evaluate_ed!` / `natural_indices`
API used across the ACE ecosystem (including Polynomials4ML), but with no
Polynomials4ML dependency. The methods simply forward to the native `compute`
API, so they take the same single and batched inputs.

```julia
using SpheriCart, ACEbase, StaticArrays

basis = SolidHarmonics(5)     # or SphericalHarmonics / the Complex* variants
𝐫 = @SVector randn(3)

Y      = evaluate(basis, 𝐫)        # ≡ compute(basis, 𝐫)
Y, ∇Y  = evaluate_ed(basis, 𝐫)     # ≡ compute_with_gradients(basis, 𝐫)
spec   = natural_indices(basis)    # the (l, m) label of each basis function
```

<!-- ## Advanced Usage

TODO:  
- different normalizations
- enforce static versus dynamic 
- wrapping outputs into zvec for easier indexing  -->