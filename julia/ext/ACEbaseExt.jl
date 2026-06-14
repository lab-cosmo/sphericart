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

# allocating interface; the trailing args... swallow optional (ps, st) since
# the harmonics bases are parameter-free.
evaluate(basis::Harmonics, x, args...) = compute(basis, x)

evaluate_ed(basis::Harmonics, x, args...) = compute_with_gradients(basis, x)

# in-place interface
function evaluate!(Y, basis::Harmonics, x, args...)
   compute!(Y, basis, x)
   return Y
end

function evaluate_ed!(Y, dY, basis::Harmonics, x, args...)
   compute_with_gradients!(Y, dY, basis, x)
   return Y, dY
end

# the (l, m) spec, in the order the basis values are stored
natural_indices(basis::Harmonics) =
      [ (l = lm[1], m = lm[2]) for lm in idx2lm.(1:length(basis)) ]

end
