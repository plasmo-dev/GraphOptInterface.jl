module TestHyperFunctions

using Graphs
using GraphOptInterface
using SparseArrays
using Test

const GOI = GraphOptInterface

function test_hypermap()
end

function run_tests()
    for name in names(@__MODULE__; all=true)
        if !startswith("$(name)", "test_")
            continue
        end
        @testset "$(name)" begin
            getfield(@__MODULE__, name)()
        end
    end
end

end