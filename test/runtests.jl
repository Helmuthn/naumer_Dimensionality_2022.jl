using naumer_ICML_2022
using SafeTestsets

# Break up the testing into a per-file setup

@safetestset "core.jl" begin
    include("core_test.jl")
end

@safetestset "evaluation.jl" begin
    include("evaluation_test.jl")
end

@safetestset "systems.jl" begin
    include("systems_test.jl")
end
