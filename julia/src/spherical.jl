
export SphericalHarmonics

using LinearAlgebra: norm, dot

"""
`struct SphericalHarmonics` : datatype representing a real spherical harmonics basis.

### Constructor to generate the basis object
```julia
basis = SphericalHarmonics(L::Integer; kwargs...)
```

### Keyword arguments:
* `normalisation = :L2` : choose the normalisation of the basis, default is to
   make it orthonoormal on the unit sphere.
* `static = (L <= 6)` : decide whether single-point evaluation uses fully
unrolled generated code that outputs an `SVector` (faster for a single input,
larger compile/stack footprint), or reuses the batched kernel. See
`SolidHarmonics` for details.
* `T = Float64` : datatype in which basis parameters are stored. The output type
is inferred at runtime, but the rule of thumb is to use `T = FloatX` for
`FloatX` output.

### Usage example:
```julia
using StaticArrays, SpheriCart
basis = SphericalHarmonics(4)
# evaluate basis with single input
𝐫 = @SVector randn(3)
Z = basis(𝐫)
Z = compute(basis, 𝐫)
# evaluate basis with multiple inputs (batching)
R = [ @SVector randn(3) for _ = 1:32 ]
Z = basis(Rs)
Z = compute(basis, Rs)

# to be implented:
# Z, ∇Z = compute_and_gradients(basis, 𝐫)
# Z, ∇Z, ∇²Z = compute_and_hessian(basis, 𝐫)
```
See documentation for more details.
"""
struct SphericalHarmonics{L, NORM, STATIC, TF} <: AbstractLuxLayer
   solids::SolidHarmonics{L, NORM, STATIC, TF}
end

SphericalHarmonics(L::Integer; kwargs...) =
      SphericalHarmonics(SolidHarmonics(L; kwargs...))

Base.getproperty(basis::SphericalHarmonics, prop::Symbol) = (
      prop == :solids ? getfield(basis, :solids)
                      : getfield(getfield(basis, :solids), prop) )

@inline (basis::SphericalHarmonics)(args...) = compute(basis, args...)

@inline function compute(basis::SphericalHarmonics, 𝐫::SVector{3},
                         st = (; Flm = basis.Flm))
   𝐫̂ = 𝐫 / norm(𝐫)
   return compute(basis.solids, 𝐫̂, st)
end

@inline function compute_with_gradients(basis::SphericalHarmonics, 𝐫::SVector{3},
                                        st = (; Flm = basis.Flm))
   r = norm(𝐫)
   𝐫̂ = 𝐫 / r
   Y, ∇Z = compute_with_gradients(basis.solids, 𝐫̂, st)

   ∇Y = map(dZ -> (dz = dZ / r; dz - dot(𝐫̂, dz) * 𝐫̂), ∇Z)

   return Y, ∇Y
end

# ---------------------
#  batched api

function compute(basis::SphericalHarmonics{L},
                  Rs::AbstractVector{<: SVector{3, T1}},
                  st = (; Flm = basis.Flm)) where {L, T1}
   Y = similar(Rs, T1, (length(Rs), sizeY(L)))
   compute!(Y, basis, Rs, st)
   return Y
end

function compute_with_gradients(basis::SphericalHarmonics{L},
                  Rs::AbstractVector{<: SVector{3, T1}},
                  st = (; Flm = basis.Flm)) where {L, T1}
   Y = similar(Rs, T1, (length(Rs), sizeY(L)))
   ∇Y = similar(Rs, SVector{3, T1}, (length(Rs), sizeY(L)))
   compute_with_gradients!(Y, ∇Y, basis, Rs, st)
   return Y, ∇Y
end


# A single KernelAbstractions kernel serves all backends (CPU + GPU) via
# `get_backend`. The `Val{true}()` selects the spherical path: the inputs are
# rescaled to the unit sphere and the gradient is projected onto it, in-kernel.
# (The single-input methods above keep the fast unrolled solid path and do the
# normalise/project on the host.)

function compute!(Y::AbstractMatrix, basis::SphericalHarmonics{L},
                  Rs::AbstractVector{<: SVector{3}},
                  st = (; Flm = basis.Flm)) where {L}
   ka_solid_harmonics!(Y, nothing, Val{L}(), Val{true}(), Rs, st.Flm)
   return Y
end

function compute_with_gradients!(Y, ∇Y, basis::SphericalHarmonics{L},
                  Rs::AbstractVector{<: SVector{3}},
                  st = (; Flm = basis.Flm)) where {L}
   ka_solid_harmonics!(Y, ∇Y, Val{L}(), Val{true}(), Rs, st.Flm)
   return Y, ∇Y
end
