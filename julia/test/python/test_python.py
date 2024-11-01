import numpy as np
from julia import Julia
import sphericart


L_MAX = 25
N_SAMPLES = 100


def test_consistency_solid_harmonics():

    jl = Julia(compiled_modules=False)
    jl_code = """
    using Pkg
    Pkg.activate("../../")
    Pkg.instantiate()
    using SpheriCart
    using StaticArrays

    function compute_solid_harmonics_from_array(xyz::Array{Float64,2}, L_MAX)
        xyz_svectors = [SVector{3}(x) for x in eachrow(xyz)]  # Convert to an array of StaticArrays.SVector{3}
        basis = SpheriCart.SolidHarmonics(L_MAX)
        results = SpheriCart.compute(basis, xyz_svectors)
        return results
    end
    """
    xyz = np.random.rand(N_SAMPLES, 3)
    jl.eval(jl_code)
    compute_solid_harmonics_from_array = jl.eval("compute_solid_harmonics_from_array")
    julia_results = compute_solid_harmonics_from_array(xyz, L_MAX)

    python_results = sphericart.SolidHarmonics(L_MAX).compute(xyz)
    assert np.allclose(julia_results, python_results)


def test_consistency_spherical_harmonics():

    jl = Julia(compiled_modules=False)
    jl_code = """
    using Pkg
    Pkg.activate("../../")
    Pkg.instantiate()
    using SpheriCart
    using StaticArrays

    function compute_solid_harmonics_from_array(xyz::Array{Float64,2}, L_MAX)
        xyz_svectors = [SVector{3}(x) for x in eachrow(xyz)]  # Convert to an array of StaticArrays.SVector{3}
        basis = SpheriCart.SphericalHarmonics(L_MAX)
        results = SpheriCart.compute(basis, xyz_svectors)
        return results
    end
    """
    xyz = np.random.rand(N_SAMPLES, 3)
    jl.eval(jl_code)
    compute_solid_harmonics_from_array = jl.eval("compute_solid_harmonics_from_array")
    julia_results = compute_solid_harmonics_from_array(xyz, L_MAX)

    python_results = sphericart.SphericalHarmonics(L_MAX).compute(xyz)
    assert np.allclose(julia_results, python_results)
