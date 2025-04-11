using SpheriCart
using Test

@testset "SpheriCart.jl" begin
   @testset "Solid Harmonics" begin include("test_solidharmonics.jl"); end 
   @testset "Spherical Harmonics" begin include("test_sphericalharmonics.jl"); end
   @testset "FloatX" begin include("test_f32.jl"); end
   @testset "KernelAbstractions" begin include("test_ka.jl"); end
end
