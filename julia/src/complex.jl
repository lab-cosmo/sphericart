
export ComplexSolidHarmonics, ComplexSphericalHarmonics

# Complex spherical / solid harmonics. These are synthesised from the real
# harmonics by the standard linear combination of the ±m components; the real
# kernels do all the work and the result is converted in place. (Moved here
# from Polynomials4ML `src/sphericart.jl::_convert_R2C!`.)

"""
`struct ComplexSolidHarmonics` : complex solid harmonics basis, wrapping a real
`SolidHarmonics` basis of the same degree. See `SolidHarmonics` for the
constructor keyword arguments.
"""
struct ComplexSolidHarmonics{L, NORM, STATIC, TF}
   realbasis::SolidHarmonics{L, NORM, STATIC, TF}
end

"""
`struct ComplexSphericalHarmonics` : complex spherical harmonics basis, wrapping
a real `SphericalHarmonics` basis of the same degree.
"""
struct ComplexSphericalHarmonics{L, NORM, STATIC, TF}
   realbasis::SphericalHarmonics{L, NORM, STATIC, TF}
end

ComplexSolidHarmonics(L::Integer; kwargs...) =
      ComplexSolidHarmonics(SolidHarmonics(L; kwargs...))

ComplexSphericalHarmonics(L::Integer; kwargs...) =
      ComplexSphericalHarmonics(SphericalHarmonics(L; kwargs...))

const ComplexHarmonics{L} =
      Union{ComplexSolidHarmonics{L}, ComplexSphericalHarmonics{L}}

@inline (basis::ComplexHarmonics)(args...) = compute(basis, args...)

# ---------------------- basis length

Base.length(::SolidHarmonics{L}) where {L} = sizeY(L)
Base.length(::SphericalHarmonics{L}) where {L} = sizeY(L)
Base.length(::ComplexSolidHarmonics{L}) where {L} = sizeY(L)
Base.length(::ComplexSphericalHarmonics{L}) where {L} = sizeY(L)

# ---------------------- real -> complex conversion (in place)

function _convert_R2C!(Y::AbstractMatrix, LMAX::Integer)
   Nx = size(Y, 1)
   for l = 0:LMAX
      # m = 0 => do nothing
      # m ≠ 0 => linear combinations of the ± m terms
      @inbounds for m = 1:l
         i_lm⁺ = lm2idx(l,  m)
         i_lm⁻ = lm2idx(l, -m)
         @simd ivdep for j = 1:Nx
            Ylm⁺ = Y[j, i_lm⁺]
            Ylm⁻ = Y[j, i_lm⁻]
            Y[j, i_lm⁺] = (-1)^m * (Ylm⁺ + im * Ylm⁻) / sqrt(2)
            Y[j, i_lm⁻] =          (Ylm⁺ - im * Ylm⁻) / sqrt(2)
         end
      end
   end
   return Y
end

# ---------------------- batched api

function compute(basis::ComplexHarmonics{L},
                 Rs::AbstractVector{<: SVector{3, T}}) where {L, T}
   Z = similar(Rs, Complex{T}, (length(Rs), sizeY(L)))
   compute!(Z, basis, Rs)
   return Z
end

function compute_with_gradients(basis::ComplexHarmonics{L},
                 Rs::AbstractVector{<: SVector{3, T}}) where {L, T}
   Z = similar(Rs, Complex{T}, (length(Rs), sizeY(L)))
   dZ = similar(Rs, SVector{3, Complex{T}}, (length(Rs), sizeY(L)))
   compute_with_gradients!(Z, dZ, basis, Rs)
   return Z, dZ
end

function compute!(Z::AbstractMatrix, basis::ComplexHarmonics{L},
                  Rs::AbstractVector{<: SVector{3}}) where {L}
   # the real kernel writes real values into the complex array Z
   compute!(Z, basis.realbasis, Rs)
   _convert_R2C!(Z, L)
   return Z
end

function compute_with_gradients!(Z::AbstractMatrix, dZ::AbstractMatrix,
                  basis::ComplexHarmonics{L},
                  Rs::AbstractVector{<: SVector{3}}) where {L}
   compute_with_gradients!(Z, dZ, basis.realbasis, Rs)
   _convert_R2C!(Z, L)
   _convert_R2C!(dZ, L)
   return Z, dZ
end

# ---------------------- single-input api
#  routed through the batched (matrix) kernel, as in the original P4ML wrapper

function compute(basis::ComplexHarmonics{L}, 𝐫::SVector{3, T}) where {L, T}
   Z = zeros(Complex{T}, sizeY(L))
   Zmat = reshape(Z, 1, :)   # a view, not a copy
   compute!(Zmat, basis, SA[𝐫,])
   return Z
end

function compute_with_gradients(basis::ComplexHarmonics{L},
                                𝐫::SVector{3, T}) where {L, T}
   Z = zeros(Complex{T}, sizeY(L))
   dZ = zeros(SVector{3, Complex{T}}, sizeY(L))
   Zmat = reshape(Z, 1, :)
   dZmat = reshape(dZ, 1, :)
   compute_with_gradients!(Zmat, dZmat, basis, SA[𝐫,])
   return Z, dZ
end
