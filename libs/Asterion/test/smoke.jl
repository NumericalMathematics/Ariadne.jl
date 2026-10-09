using Test
using Asterion

@testset "Asterion" begin
    @test Asterion isa Module
    @test pkgversion(Asterion) isa VersionNumber
end
