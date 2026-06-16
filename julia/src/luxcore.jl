
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

# the bases have no trainable parameters ...
LuxCore.initialparameters(::AbstractRNG, ::_Harmonics) = NamedTuple()

# ... and carry the (non-trainable) prefactor matrix as their state
LuxCore.initialstates(::AbstractRNG, basis::_Harmonics) = (Flm = basis.Flm,)

# forward pass: `compute` takes the state directly (it reads `st.Flm`), so the
# Lux call is simply a thin wrapper returning `(Y, st)`.
@inline (basis::SolidHarmonics)(x, ps, st) = (compute(basis, x, st), st)
@inline (basis::SphericalHarmonics)(x, ps, st) = (compute(basis, x, st), st)
@inline (basis::ComplexSolidHarmonics)(x, ps, st) = (compute(basis, x, st), st)
@inline (basis::ComplexSphericalHarmonics)(x, ps, st) = (compute(basis, x, st), st)
