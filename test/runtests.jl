# Packages required for testing
using Test
using Random
using Logging

using CMSSPs

# Set logging level
global_logger(SimpleLogger(stderr, Logging.Debug))

# Fix randomness during tests
Random.seed!(0)

# Define package tests
@time @testset "CMSSPs Package Tests" begin
    testdir = joinpath(dirname(@__DIR__), "test")
    @time @testset "CMSSPs.CMSSPModel" begin
        include(joinpath(testdir, "test_cmssp_model.jl"))
    end
end