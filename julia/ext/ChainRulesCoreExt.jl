module ChainRulesCoreExt

# Reverse-mode AD (Zygote, ...) through `compute`, using the package's analytic
# gradients (`compute_with_gradients`) for the pullback rather than differentiating
# through the kernels. Differentiation is w.r.t. the input point(s); the bases are
# parameter-free (`Flm` is fixed), so the basis cotangent is `NoTangent()`.

using SpheriCart
import SpheriCart: SolidHarmonics, SphericalHarmonics,
                   ComplexSolidHarmonics, ComplexSphericalHarmonics,
                   compute, compute_with_gradients
import ChainRulesCore: rrule, NoTangent, @thunk, unthunk

const _Harmonics = Union{SolidHarmonics, SphericalHarmonics,
                         ComplexSolidHarmonics, ComplexSphericalHarmonics}

# Contract the output cotangent `Ȳ` with the analytic gradient `∇Z` to give the
# cotangent of the input point(s). The conjugate-adjoint form `real(Σ Ȳᵢ conj(∇Zᵢ))`
# is the ChainRules real-gradient convention and yields a real tangent (matching the
# real input) for both the real and the complex bases -- for the real bases `conj`
# and `real` are identities. Broadcast + `sum` only, so it also runs on the GPU.
_pull(Ȳ, ∇Z::AbstractVector) = real.(sum(Ȳ .* conj.(∇Z)))                 # single point
_pull(Ȳ, ∇Z::AbstractMatrix) = vec(real.(sum(Ȳ .* conj.(∇Z); dims = 2)))  # batched

# `compute(basis, x)` is just `compute(basis, x, (; Flm = basis.Flm))` (the state
# argument is optional), so a rule on the 3-arg form is sufficient: AD of a 2-arg
# call recurses through the default-argument stub into this rule. `basis` and `st`
# are non-differentiable; differentiation is w.r.t. the input only.
function rrule(::typeof(compute), basis::_Harmonics, x, st)
   Y, ∇Z = compute_with_gradients(basis, x, st)
   compute_pb(Ȳ) = (NoTangent(), NoTangent(),
                    @thunk(_pull(unthunk(Ȳ), ∇Z)), NoTangent())
   return Y, compute_pb
end

end
