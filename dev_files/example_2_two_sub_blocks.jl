using Revise
using Pkg
Pkg.activate(@__DIR__())

# schur_optimizer.jl will be migrated to MadNLP
include(joinpath(@__DIR__(),"../schur_optimizer/schur_optimizer.jl"))

optimizer = SchurOptimizer()

##################################################
# block 0: node 0 and edge 0
##################################################
node0 = BOI.add_node!(optimizer)
x0 = MOI.add_variables(optimizer, node0, 3)

# constraints/bounds on variables
for x_i in x0
   MOI.add_constraint(optimizer, node0, x_i, MOI.GreaterThan(0.0))
end

edge0 = BOI.add_edge!(optimizer, node0)

c0 = [1.0, 2.0, 3.0]
w0 = [0.3, 0.5, 1.0]
C0 = 3.2

# set edge0 objective
MOI.set(
	edge0,
	MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
	MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(c0, x0), 0.0)
)
MOI.set(edge0, MOI.ObjectiveSense(), MOI.MAX_SENSE)

# add edge0 constraint
ci = MOI.add_constraint(
    optimizer,
    edge0,
    MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(w0, x0), 0.0),
    MOI.LessThan(C0)
)

# add edge0 nonlinear constraint 
nlp0 = MOI.Nonlinear.Model()
MOI.Nonlinear.add_constraint(nlp0, :(1 + sqrt($(x0[1]))), MOI.LessThan(2.0))
evaluator0 = MOI.Nonlinear.Evaluator(nlp0, MOI.Nonlinear.SparseReverseMode(), x0)
nlp_block0 = MOI.NLPBlockData(evaluator0)
MOI.set(edge0, MOI.NLPBlock(), nlp_block0)


##################################################
# sub-block 1
##################################################
sub_block1 = BOI.add_sub_block!(optimizer)
node1 = BOI.add_node!(optimizer, sub_block1)

# NOTE: returns optimizer index, not node index
x1 = MOI.add_variables(optimizer, node1, 3)
for x_i in x1
   MOI.add_constraint(optimizer, node1, x_i, MOI.GreaterThan(0.0))
end

edge1 = BOI.add_edge!(optimizer, node1)

c1 = [2.0, 3.0, 4.0]
w1 = [0.2, 0.1, 1.2]
C1 = 2.0

MOI.set(
	edge1,
	MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
	MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(c1, x1), 0.0),
)
MOI.set(edge1, MOI.ObjectiveSense(), MOI.MAX_SENSE)
MOI.add_constraint(
    optimizer, 
    edge1,
    MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(w1, x1), 0.0),
    MOI.LessThan(C1)
)

nlp1 = MOI.Nonlinear.Model()
MOI.Nonlinear.add_constraint(nlp1, :(1 + sqrt($(x1[2]))), MOI.LessThan(3.0))
evaluator1 = MOI.Nonlinear.Evaluator(nlp1, MOI.Nonlinear.SparseReverseMode(), x1)
nlp_block1 = MOI.NLPBlockData(evaluator1)
MOI.set(edge1, MOI.NLPBlock(), nlp_block1)

##################################################
# sub-block 2
##################################################
sub_block2 = BOI.add_sub_block!(optimizer)
node2 = BOI.add_node!(optimizer, sub_block2)

# NOTE: returns optimizer index, not node index
x2 = MOI.add_variables(optimizer, node2, 3)
for x_i in x2
   MOI.add_constraint(optimizer, node2, x_i, MOI.GreaterThan(0.0))
end

edge2 = BOI.add_edge!(optimizer, node2)

c2 = [2.0, 3.0, 4.0]
w2 = [0.2, 0.1, 1.2]
C2 = 2.0

MOI.set(
	edge2,
	MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
	MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(c2, x2), 0.0),
)
MOI.set(edge2, MOI.ObjectiveSense(), MOI.MAX_SENSE)
MOI.add_constraint(
    optimizer, 
    edge2,
    MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(w2, x2), 0.0),
    MOI.LessThan(C2)
)

nlp2 = MOI.Nonlinear.Model()
MOI.Nonlinear.add_constraint(nlp2, :(1 + sqrt($(x2[2]))), MOI.LessThan(3.0))
evaluator2 = MOI.Nonlinear.Evaluator(nlp2, MOI.Nonlinear.SparseReverseMode(), x2)
nlp_block2 = MOI.NLPBlockData(evaluator2)
MOI.set(edge2, MOI.NLPBlock(), nlp_block2)

##################################################
# links between blocks
##################################################
block = optimizer.block

### edge from node0 to sub-block1

edge_0_1 = BOI.add_edge!(optimizer, BOI.BlockIndex(0), node0, sub_block1)
xe_0 = BOI.add_edge_variable!(block, edge_0_1, node0, node0[1])
xe_1 = BOI.add_edge_variable!(block, edge_0_1, node1, node1[1])

MOI.add_constraint(
    optimizer,   
    edge_0_1,
    MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0,-1.0], [xe_0,xe_1]), 0.0),
    MOI.EqualTo(0.0)
)

### edge from node0 to sub-block2

edge_0_2 = BOI.add_edge!(optimizer, BOI.BlockIndex(0), node0, sub_block2)
xe_0 = BOI.add_edge_variable!(block, edge_0_2, node0, node0[1])
xe_1 = BOI.add_edge_variable!(block, edge_0_2, node2, node2[1])

MOI.add_constraint(
    optimizer,   
    edge_0_2,
    MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0,-1.0], [xe_0,xe_1]), 0.0),
    MOI.EqualTo(0.0)
)

### edge between sub-block1 and sub-block2

edge_1_2 = BOI.add_edge!(optimizer, BOI.BlockIndex(0), (sub_block1, sub_block2))
xe_1 = BOI.add_edge_variable!(block, edge_1_2, node1, node1[2])
xe_2 = BOI.add_edge_variable!(block, edge_1_2, node2, node2[2])

MOI.add_constraint(
    optimizer,   
    edge_1_2,
    MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0,-1.0], [xe_1,xe_2]), 0.0),
    MOI.EqualTo(0.0)
)