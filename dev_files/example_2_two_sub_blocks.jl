using Revise
using Pkg
Pkg.activate(@__DIR__())

using GraphOptInterface
const GOI = GraphOptInterface

# schur_optimizer.jl will be migrated to MadNLP
# include(joinpath(@__DIR__(),"../schur_optimizer/schur_optimizer.jl"))
#optimizer = SchurOptimizer()

##################################################
# block 0: node 0 and edge 0
##################################################
graph = GOI.get(optimizer, GOI.Graph())
node0 = GOI.add_node(graph)
x0 = GOI.add_variables(node0, 3)

# constraints on variables
for x_i in x0
   MOI.add_constraint(node0, x_i, MOI.GreaterThan(0.0))
end

edge0 = add_edge(graph, node0)

c0 = [1.0, 2.0, 3.0]
w0 = [0.3, 0.5, 1.0]
C0 = 3.2

MOI.set(
	edge0,
	MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
	MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(c0, x0), 0.0)
)
MOI.set(edge0, MOI.ObjectiveSense(), MOI.MAX_SENSE)

ci = MOI.add_constraint(
    edge0,
    MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(w0, x0), 0.0),
    MOI.LessThan(C0)
)

# node nonlinear constraint
MOI.Nonlinear.add_constraint(edge0, :(1 + sqrt($(x0[1]))), MOI.LessThan(2.0))

# the solver will create the NLP Block, or we create the block evaluator and pass it in.
# evaluator0 = MOI.Nonlinear.Evaluator(nlp0, MOI.Nonlinear.SparseReverseMode(), x0)
# nlp_block0 = MOI.NLPBlockData(evaluator0)
# MOI.set(optimizer, node0, MOI.NLPBlock(), nlp_block0)

##################################################
# sub-block 1
##################################################
sub_block1 = GOI.add_sub_block(graph)
node1 = GOI.add_node(graph, sub_block1)

x1 = MOI.add_variables(node1, 3)
for x_i in x1
   MOI.add_constraint(node, x_i, MOI.GreaterThan(0.0))
end

c1 = [2.0, 3.0, 4.0]
w1 = [0.2, 0.1, 1.2]
C1 = 2.0

edge1 = add_edge(graph, node1)

MOI.set(
	edge1,
	MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
	MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(c1, x1), 0.0),
)
MOI.set(edge1, MOI.ObjectiveSense(), MOI.MAX_SENSE)
MOI.add_constraint(
    edge1,
    MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(w1, x1), 0.0),
    MOI.LessThan(C1)
)

MOI.Nonlinear.add_constraint(edge1, :(1 + sqrt($(x1[2]))), MOI.LessThan(3.0))

#nlp1 = MOI.Nonlinear.Model()

# evaluator1 = MOI.Nonlinear.Evaluator(nlp1, MOI.Nonlinear.SparseReverseMode(), x1)
# nlp_block1 = MOI.NLPBlockData(evaluator1)
# MOI.set(optimizer, node1, MOI.NLPBlock(), nlp_block1)

##################################################
# sub-block 2
##################################################
sub_block2 = GOI.add_sub_block(graph)
node2 = GOI.add_node(graph, sub_block2)

x2 = MOI.add_variables(graph, node2, 3)
for x_i in x2
   MOI.add_constraint(node2, x_i, MOI.GreaterThan(0.0))
end

c2 = [2.0, 3.0, 4.0]
w2 = [0.2, 0.1, 1.2]
C2 = 2.0

edge2 = add_edge(graph, node2)

MOI.set(
	edge2,
	MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
	MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(c2, x2), 0.0),
)
MOI.set(edge2, MOI.ObjectiveSense(), MOI.MAX_SENSE)
MOI.add_constraint(
    edge2,
    MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(w2, x2), 0.0),
    MOI.LessThan(C2)
)
MOI.Nonlinear.add_constraint(edge2, :(1 + sqrt($(x2[2]))), MOI.LessThan(3.0))

#nlp2 = MOI.Nonlinear.Model()

# evaluator2 = MOI.Nonlinear.Evaluator(nlp2, MOI.Nonlinear.SparseReverseMode(), x2)
# nlp_block2 = MOI.NLPBlockData(evaluator2)
# MOI.set(optimizer, node2, MOI.NLPBlock(), nlp_block2)

##################################################
# links between blocks
##################################################

### edge from node0 to sub-block1
edge_0_1 = GOI.add_edge(graph, node0, node1)
x0 = MOI.add_variable(edge_0_1, node0, MOI.VariableIndex(1))
x1 = MOI.add_variable(edge_0_1, node1, MOI.VariableIndex(1))

MOI.add_constraint(
    edge_0_1,
    MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0,-1.0], [x0,x1]), 0.0),
    MOI.EqualTo(0.0)
)

### edge from node0 to node2

edge_0_2 = GOI.add_edge(graph, node0, node2)
x0 = MOI.add_variable(edge, node0, MOI.VariableIndex(1))
x1 = MOI.add_variable(edge, node2, MOI.VariableIndex(1))
MOI.add_constraint( 
    edge_0_2,
    MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0,-1.0], [x0,xn_1]), 0.0),
    MOI.EqualTo(0.0)
)

### edge between sub-block1 and sub-block2

edge_1_2 = GOI.add_edge(graph, node1, node2)
x1 = MOI.add_variable(edge_1_2, node1, MOI.VariableIndex(2))
x2 = MOI.add_variable(edge_1_2, node2, MOI.VariableIndex(2))

MOI.add_constraint(
    edge_1_2,
    MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0,-1.0], [x1,x2]), 0.0),
    MOI.EqualTo(0.0)
)

# GOI.all_neighbors(graph, sub_block1)

# GOI.parent_neighbors(optimizer, sub_block1)

# GOI.neighbors(optimizer, sub_block1)

# GOI.all_incident_edges(optimizer, sub_block1)

# GOI.parent_incident_edges(optimizer, sub_block1)

# GOI.incident_edges(optimizer, sub_block1)