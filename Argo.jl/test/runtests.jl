using Test

@testset "Argo" begin
    include("jump.jl")
    include("autotune.jl")
end
