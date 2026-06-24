module ACEbaseExt

# Provides the standard ACEbase evaluation interface (evaluate / evaluate_ed /
# evaluate! / evaluate_ed! / natural_indices) for the SpheriCart harmonics
# bases, by forwarding to the native `compute` API. Loading this extension
# (i.e. `using SpheriCart, ACEbase`) makes the harmonics usable through the
# shared ACEsuit interface with no Polynomials4ML dependency.

using SpheriCart
import SpheriCart: SolidHarmonics, SphericalHarmonics,
                   ComplexSolidHarmonics, ComplexSphericalHarmonics,
                   compute, compute!,
                   compute_with_gradients, compute_with_gradients!,
                   idx2lm
import ACEbase: evaluate, evaluate_ed, evaluate!, evaluate_ed!, natural_indices

const Harmonics = Union{SolidHarmonics, SphericalHarmonics,
                        ComplexSolidHarmonics, ComplexSphericalHarmonics}

# Two arities are supported throughout:
#   * `evaluate(basis, x)`          — uses the basis' built-in prefactors;
#   * `evaluate(basis, x, ps, st)`  — the ACEsuit `(parameters, state)` convention.
# The bases are parameter-free, so `ps` is ignored, but the *state* carries the
# `Flm` prefactor matrix and must be forwarded to `compute` (e.g. so that a
# state with `Float32` prefactors evaluates in `Float32`). The previous
# `args...`-swallowing methods dropped `st` and always used `basis.Flm`, so the
# `(ps, st)` calls silently ignored the supplied state.

# allocating interface
evaluate(basis::Harmonics, x) = compute(basis, x)
evaluate(basis::Harmonics, x, ps, st) = compute(basis, x, st)

evaluate_ed(basis::Harmonics, x) = compute_with_gradients(basis, x)
evaluate_ed(basis::Harmonics, x, ps, st) = compute_with_gradients(basis, x, st)

# in-place interface
evaluate!(Y, basis::Harmonics, x) = (compute!(Y, basis, x); Y)
evaluate!(Y, basis::Harmonics, x, ps, st) = (compute!(Y, basis, x, st); Y)

evaluate_ed!(Y, dY, basis::Harmonics, x) =
      (compute_with_gradients!(Y, dY, basis, x); (Y, dY))
evaluate_ed!(Y, dY, basis::Harmonics, x, ps, st) =
      (compute_with_gradients!(Y, dY, basis, x, st); (Y, dY))

# the (l, m) spec, in the order the basis values are stored
natural_indices(basis::Harmonics) =
      [ (l = lm[1], m = lm[2]) for lm in idx2lm.(1:length(basis)) ]

end
