using MathOptInterface
const MOI = MathOptInterface

using GraphOptInterface
const GOI = GraphOptInterface

##################################################
# graph: node0
##################################################
graph = GOI.OptiGraph()
node0 = GOI.add_node(graph)
x0 = MOI.add_variables(node0, 3)

# constraints on variables
for x_i in x0
    MOI.add_constraint(node0, x_i, MOI.GreaterThan(0.0))
end

c0 = [1.0, 2.0, 3.0]
w0 = [0.3, 0.5, 1.0]
C0 = 3.2

MOI.set(
    node0,
    MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
    MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(c0, x0), 0.0),
)
MOI.set(node0, MOI.ObjectiveSense(), MOI.MAX_SENSE)

ci = MOI.add_constraint(
    node0, MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(w0, x0), 0.0), MOI.LessThan(C0)
)

MOI.Nonlinear.add_constraint(node0, :(1 + sqrt($(x0[1]))), MOI.LessThan(2.0))

##################################################
# subgraph1: node1
##################################################
subgraph1 = GOI.add_subgraph(graph)
node1 = GOI.add_node(subgraph1)

x1 = MOI.add_variables(node1, 3)
for x_i in x1
    MOI.add_constraint(node1, x_i, MOI.GreaterThan(0.0))
end

c1 = [2.0, 3.0, 4.0]
w1 = [0.2, 0.1, 1.2]
C1 = 2.0

MOI.set(
    node1,
    MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
    MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(c1, x1), 0.0),
)
MOI.set(node1, MOI.ObjectiveSense(), MOI.MAX_SENSE)
MOI.add_constraint(
    node1, MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(w1, x1), 0.0), MOI.LessThan(C1)
)
MOI.Nonlinear.add_constraint(node1, :(1 + sqrt($(x1[2]))), MOI.LessThan(3.0))

##################################################
# subgraph2: node2
##################################################
subgraph2 = GOI.add_subgraph(graph)
node2 = GOI.add_node(subgraph2)
x2 = MOI.add_variables(node2, 3)
for x_i in x2
    MOI.add_constraint(node2, x_i, MOI.GreaterThan(0.0))
end

c2 = [2.0, 3.0, 4.0]
w2 = [0.2, 0.1, 1.2]
C2 = 2.0

MOI.set(
    node2,
    MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
    MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(c2, x2), 0.0),
)
MOI.set(node2, MOI.ObjectiveSense(), MOI.MAX_SENSE)
MOI.add_constraint(
    node2, MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(w2, x2), 0.0), MOI.LessThan(C2)
)
MOI.Nonlinear.add_constraint(node2, :(1 + sqrt($(x2[2]))), MOI.LessThan(3.0))

##################################################
# links between subgraphs
##################################################

### edge from graph to subgraph1

edge1 = GOI.add_edge(graph, (node0, node1))
e1_x0 = MOI.add_variable(edge1, node0, MOI.VariableIndex(1))
e1_x1 = MOI.add_variable(edge1, node1, MOI.VariableIndex(1))
MOI.add_constraint(
    edge1,
    MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0, -1.0], [e1_x0, e1_x1]), 0.0),
    MOI.EqualTo(0.0),
)

### edge from graph to subgraph2

edge2 = GOI.add_edge(graph, (node0, node2))
e2_x0 = MOI.add_variable(edge2, node0, MOI.VariableIndex(1))
e2_x1 = MOI.add_variable(edge2, node2, MOI.VariableIndex(1))
MOI.add_constraint(
    edge2,
    MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0, -1.0], [e2_x0, e2_x1]), 0.0),
    MOI.EqualTo(0.0),
)

### edge between subgraph1 and subgraph2

edge3 = GOI.add_edge(graph, (node1, node2))
e3_x1 = MOI.add_variable(edge3, node1, MOI.VariableIndex(2))
e3_x2 = MOI.add_variable(edge3, node2, MOI.VariableIndex(2))
MOI.add_constraint(
    edge3,
    MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0, -1.0], [e3_x1, e3_x2]), 0.0),
    MOI.EqualTo(0.0),
)
