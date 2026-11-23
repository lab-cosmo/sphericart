# Check that the Julia implementation gives the same results as the Python
# package. Python is called as a subprocess, and arrays are exchanged as raw
# Float64 binary files, to avoid depending on a Python <=> Julia bridge.
#
# Usage: julia --project=julia julia/test/python/consistency.jl
# (requires `python` with `sphericart` and `numpy` installed in the PATH)

using SpheriCart
using StaticArrays
using Test

const L_MAX = 25
const N_SAMPLES = 100

const PYTHON_SCRIPT = """
import sys
import numpy as np
import sphericart

directory, l_max, n_samples = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])

# Julia arrays are column-major
xyz = np.fromfile(f"{directory}/xyz.bin").reshape(3, n_samples).T

solid = sphericart.SolidHarmonics(l_max).compute(xyz)
solid.T.tofile(f"{directory}/solid.bin")

spherical = sphericart.SphericalHarmonics(l_max).compute(xyz)
spherical.T.tofile(f"{directory}/spherical.bin")
"""

mktempdir() do dir
    xyz = rand(N_SAMPLES, 3)
    write(joinpath(dir, "xyz.bin"), xyz)

    run(`python -c $PYTHON_SCRIPT $dir $L_MAX $N_SAMPLES`)

    xyz_svectors = [SVector{3}(x) for x in eachrow(xyz)]
    n_basis = (L_MAX + 1)^2

    @testset "Julia/Python consistency" begin
        @testset "Solid Harmonics" begin
            python = read!(joinpath(dir, "solid.bin"), Matrix{Float64}(undef, N_SAMPLES, n_basis))
            julia = SpheriCart.compute(SolidHarmonics(L_MAX), xyz_svectors)
            @test julia ≈ python
        end

        @testset "Spherical Harmonics" begin
            python = read!(joinpath(dir, "spherical.bin"), Matrix{Float64}(undef, N_SAMPLES, n_basis))
            julia = SpheriCart.compute(SphericalHarmonics(L_MAX), xyz_svectors)
            @test julia ≈ python
        end
    end
end
