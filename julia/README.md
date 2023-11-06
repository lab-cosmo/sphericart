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

There are two implementations of real solid harmonics and real spherical harmonics
- a generated  implementation for a single `𝐫::SVector{3, T}` input, returning the spherical harmonics as an `SVector{T}`. 
- a generic implementation that is optimized for evaluating over batches of inputs, exploiting SIMD vectorization. 

For large enough batches (system dependent) the second implementation is comparable to or faster than broadcasting over the generated implementation. For single inputs, the generated implementation is far superior in performance. 


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

<!-- ## Advanced Usage

TODO:  
- different normalizations
- enforce static versus dynamic 
- wrapping outputs into zvec for easier indexing  -->