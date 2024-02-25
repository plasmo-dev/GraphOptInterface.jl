module TestOptiGraph

using Graphs
using SparseArrays
using Test

using MathOptInterface
const MOI = MathOptInterface

using GraphOptInterface
const GOI = GraphOptInterface

function test_optigraph()
    graph = GOI.OptiGraph()

    node1 = GOI.add_node(graph)
    @test GOI.num_nodes(graph) == 1
    @test node1.nonlinear_model == nothing

    x1 = MOI.add_variables(node1, 3)
    @test MOI.get(node1, MOI.NumberOfVariables()) == 3

    for x in x1
        MOI.add_constraint(node1, x, MOI.GreaterThan(0.0))
        MOI.add_constraint(node1, x, MOI.LessThan(5.0))
    end

    @test MOI.get(
        node1, MOI.NumberOfConstraints{MOI.VariableIndex,MOI.GreaterThan{Float64}}()
    ) == 3

    @test MOI.get(
        node1, MOI.NumberOfConstraints{MOI.VariableIndex,MOI.LessThan{Float64}}()
    ) == 3

    c1 = [1.0, 2.0, 3.0]
    w1 = [0.3, 0.5, 1.0]
    C1 = 3.2

    MOI.set(
        node1,
        MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(c1, x1), 0.0),
    )
    MOI.set(node1, MOI.ObjectiveSense(), MOI.MAX_SENSE)
    @test MOI.get(node1, MOI.ObjectiveFunctionType()) == MOI.ScalarAffineFunction{Float64}
    @test MOI.get(node1, MOI.ObjectiveSense()) == MOI.MAX_SENSE

    ci = MOI.add_constraint(
        node1,
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(w1, x1), 0.0),
        MOI.LessThan(C1),
    )
    @test MOI.get(
        node1,
        MOI.NumberOfConstraints{MOI.ScalarAffineFunction{Float64},MOI.LessThan{Float64}}(),
    ) == 1

    MOI.Nonlinear.add_constraint(node1, :(1.0 + sqrt($(x1[1]))), MOI.LessThan(5.0))
    @test length(node1.nonlinear_model.constraints) == 1

    node2 = GOI.add_node(graph)
    x2 = MOI.add_variables(node2, 3)
    for x in x2
        MOI.add_constraint(node2, x, MOI.GreaterThan(0.0))
        MOI.add_constraint(node2, x, MOI.LessThan(5.0))
    end
    edge = GOI.add_edge(graph, (node1, node2))
    @test GOI.num_edges(graph) == 1
    @test GOI.get_nodes(edge) == [node1, node2]

    e_x1 = MOI.add_variable(edge, node1, MOI.VariableIndex(1))
    e_x2 = MOI.add_variable(edge, node2, MOI.VariableIndex(1))
    e_x3 = MOI.add_variable(edge, node2, MOI.VariableIndex(3))
    @test MOI.get(edge, MOI.NumberOfVariables()) == 3

    MOI.add_constraint(
        edge,
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0, -1.0], [e_x1, e_x2]), 0.0),
        MOI.EqualTo(0.0),
    )
    @test MOI.get(
        edge,
        MOI.NumberOfConstraints{MOI.ScalarAffineFunction{Float64},MOI.EqualTo{Float64}}(),
    ) == 1

    MOI.Nonlinear.add_constraint(
        edge, :(1.0 + sqrt($(e_x1)) + $(e_x3)^3), MOI.LessThan(5.0)
    )
    @test length(edge.nonlinear_model.constraints) == 1

    MOI.set(edge, MOI.ObjectiveSense(), MOI.MAX_SENSE)
    @test MOI.get(edge, MOI.ObjectiveSense()) == MOI.MAX_SENSE
end

function test_optigraph_subgraphs()
    graph = GOI.OptiGraph()

    node0 = GOI.add_node(graph)
    x0 = MOI.add_variables(node0, 3)
    for x_i in x0
        MOI.add_constraint(node0, x_i, MOI.GreaterThan(0.0))
    end

    subgraph1 = GOI.add_subgraph(graph)
    @assert GOI.num_subgraphs(graph) == 1

    node1 = GOI.add_node(subgraph1)
    x1 = MOI.add_variables(node1, 3)
    for x in x1
        MOI.add_constraint(node1, x, MOI.GreaterThan(0.0))
    end
    @assert GOI.num_nodes(graph) == 1
    @assert GOI.num_nodes(subgraph1) == 1
    @assert GOI.num_all_nodes(graph) == 2

    edge_1 = GOI.add_edge(graph, (node0, node1))
    e1_x0 = MOI.add_variable(edge_1, node0, MOI.VariableIndex(1))
    e1_x1 = MOI.add_variable(edge_1, node1, MOI.VariableIndex(1))
    MOI.add_constraint(
        edge_1,
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0, -1.0], [e1_x0, e1_x1]), 0.0),
        MOI.EqualTo(0.0),
    )
    @assert GOI.num_edges(graph) == 1

    subgraph2 = GOI.add_subgraph(graph)
    node2 = GOI.add_node(subgraph2)
    x2 = MOI.add_variables(node2, 3)
    for x_i in x2
        MOI.add_constraint(node2, x_i, MOI.GreaterThan(0.0))
    end
    @assert GOI.num_nodes(subgraph2) == 1
    @assert GOI.num_all_nodes(graph) == 3

    edge_2 = GOI.add_edge(graph, (node0, node2))
    e2_x0 = MOI.add_variable(edge_2, node0, MOI.VariableIndex(1))
    e2_x1 = MOI.add_variable(edge_2, node2, MOI.VariableIndex(1))
    MOI.add_constraint(
        edge_2,
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0, -1.0], [e2_x0, e2_x1]), 0.0),
        MOI.EqualTo(0.0),
    )
    @assert GOI.num_edges(graph) == 2

    edge_3 = GOI.add_edge(graph, (node1, node2))
    e3_x1 = MOI.add_variable(edge_3, node1, MOI.VariableIndex(2))
    e3_x2 = MOI.add_variable(edge_3, node2, MOI.VariableIndex(2))
    MOI.add_constraint(
        edge_3,
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0, -1.0], [e3_x1, e3_x2]), 0.0),
        MOI.EqualTo(0.0),
    )
    @assert GOI.num_edges(graph) == 3
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

end # module

TestOptiGraph.run_tests()
