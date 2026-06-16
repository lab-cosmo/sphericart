
# ---------------------------------------------------------------------------
#  LuxCore layer interface
#
#  All harmonics bases are parameter-free `AbstractLuxLayer`s. The normalisation
#  prefactors `Flm` are exposed as the layer STATE, so that the standard Lux
#  device / element-type transforms (`gpu`, `cpu`, `f32`, `f64`, ...) move and
#  convert them. `Flm` is also retained in the layer (so the plain `compute` API
#  keeps working without setting up states); the Lux forward pass evaluates with
#  the `Flm` taken from the state.
#
#  Usage:
#  ```julia
#  using SpheriCart, LuxCore, Random
#  basis = SolidHarmonics(4)
#  ps, st = LuxCore.setup(Random.default_rng(), basis)   # ps == (;)  st == (; Flm)
#  Y, st  = basis(𝐫, ps, st)                              # == compute(basis, 𝐫)
#  ```
# ---------------------------------------------------------------------------

const _Harmonics = Union{SolidHarmonics, SphericalHarmonics,
                         ComplexSolidHarmonics, ComplexSphericalHarmonics}

# extract / replace the prefactor matrix, preserving the wrapper structure
_basis_Flm(b::SolidHarmonics) = b.Flm
_basis_Flm(b::SphericalHarmonics) = _basis_Flm(b.solids)
_basis_Flm(b::ComplexSolidHarmonics) = _basis_Flm(b.realbasis)
_basis_Flm(b::ComplexSphericalHarmonics) = _basis_Flm(b.realbasis)

_with_Flm(::SolidHarmonics{L, NORM, STATIC}, Flm) where {L, NORM, STATIC} =
      SolidHarmonics{L, NORM, STATIC, typeof(Flm)}(Flm)
_with_Flm(b::SphericalHarmonics, Flm) = SphericalHarmonics(_with_Flm(b.solids, Flm))
_with_Flm(b::ComplexSolidHarmonics, Flm) = ComplexSolidHarmonics(_with_Flm(b.realbasis, Flm))
_with_Flm(b::ComplexSphericalHarmonics, Flm) = ComplexSphericalHarmonics(_with_Flm(b.realbasis, Flm))

# the bases have no trainable parameters ...
LuxCore.initialparameters(::AbstractRNG, ::_Harmonics) = NamedTuple()

# ... and carry the (non-trainable) prefactor matrix as their state
LuxCore.initialstates(::AbstractRNG, basis::_Harmonics) = (Flm = _basis_Flm(basis),)

# forward pass: evaluate using the `Flm` from the state, return `(Y, st)`
@inline _lux_apply(basis, x, st) = (compute(_with_Flm(basis, st.Flm), x), st)

@inline (basis::SolidHarmonics)(x, ps, st) = _lux_apply(basis, x, st)
@inline (basis::SphericalHarmonics)(x, ps, st) = _lux_apply(basis, x, st)
@inline (basis::ComplexSolidHarmonics)(x, ps, st) = _lux_apply(basis, x, st)
@inline (basis::ComplexSphericalHarmonics)(x, ps, st) = _lux_apply(basis, x, st)
