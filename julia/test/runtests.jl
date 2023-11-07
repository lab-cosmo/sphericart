using SpheriCart
using Test

@testset "SpheriCart.jl" begin
   @testset "Solid Harmonics" begin include("test_solidharmonics.jl"); end 
   @testset "Spherical Harmonics" begin include("test_sphericalharmonics.jl"); end
end
