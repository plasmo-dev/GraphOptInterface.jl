using Revise
using Pkg
Pkg.activate(@__DIR__())

# schur_optimizer.jl will be migrated to MadNLP
include(joinpath(@__DIR__(),"../schur_optimizer/schur_optimizer.jl"))

optimizer = SchurOptimizer()
block = MOI.get(optimizer, BOI.BlockStructure())

##################################################
# node 1 and edge 1
##################################################
node1 = BOI.add_node!(optimizer)
x1 = MOI.add_variables(optimizer, node1, 3)

# constraints/bounds on variables
for x_i in x1
   MOI.add_constraint(optimizer, node1, x_i, MOI.GreaterThan(0.0))
   MOI.add_constraint(optimizer, node1, x_i, MOI.LessThan(5.0))
end

edge1 = BOI.add_edge!(optimizer, node1)

c1 = [1.0, 2.0, 3.0]
w1 = [0.3, 0.5, 1.0]
C1 = 3.2

# set edge1 objective
MOI.set(
	edge1,
	MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
	MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(c1, x1), 0.0)
)
MOI.set(edge1, MOI.ObjectiveSense(), MOI.MAX_SENSE)

# add edge1 constraint
ci = MOI.add_constraint(
    optimizer,
    edge1,
    MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(w1, x1), 0.0),
    MOI.LessThan(C1)
)

# add edge1 nonlinear constraint
nlp1 = MOI.Nonlinear.Model()
# MOI.Nonlinear.set_objective(nlp1, :($(c1[1])*$(x1[1]) + $(c1[2])*$(x1[2]) + $(c1[3])*$(x1[3])))
MOI.Nonlinear.add_constraint(nlp1, :(1.0 + sqrt($(x1[1]))), MOI.LessThan(5.0))
evaluator1 = MOI.Nonlinear.Evaluator(nlp1, MOI.Nonlinear.SparseReverseMode(), x1)
block1 = MOI.NLPBlockData(evaluator1)
MOI.set(edge1, MOI.NLPBlock(), block1)

##################################################
# node 2 and edge 2
##################################################
node2 = BOI.add_node!(optimizer)
x2 = MOI.add_variables(optimizer, node2, 3)
for x_i in x2
   MOI.add_constraint(optimizer, node2, x_i, MOI.GreaterThan(0.0))
   MOI.add_constraint(optimizer, node2, x_i, MOI.LessThan(5.0))
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
MOI.Nonlinear.add_constraint(nlp2, :(1.0 + sqrt($(x2[2]))), MOI.LessThan(3.0))
evaluator2 = MOI.Nonlinear.Evaluator(nlp2, MOI.Nonlinear.SparseReverseMode(), x2)
block2 = MOI.NLPBlockData(evaluator2)
MOI.set(edge2, MOI.NLPBlock(), block2)

##################################################
# edge 3 - couple node1 and node2
##################################################
edge3 = BOI.add_edge!(optimizer, BOI.BlockIndex(0), (node1, node2))
# add variables to edge
BOI.add_edge_variable!(optimizer.block, edge3, node1, node1[1])
BOI.add_edge_variable!(optimizer.block, edge3, node2, node2[1])
BOI.add_edge_variable!(optimizer.block, edge3, node2, node2[3])
x3 = BOI.variable_indices(optimizer.block, edge3)
MOI.add_constraint(
    optimizer,   
    edge3,
    MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0,-1.0], [x3[1],x3[2]]), 0.0),
    MOI.EqualTo(0.0)
)
nlp3 = MOI.Nonlinear.Model()
MOI.Nonlinear.add_constraint(nlp3, :(1.0 + sqrt($(x3[1])) + $(x3[3])^3), MOI.LessThan(5.0))
evaluator3 = MOI.Nonlinear.Evaluator(nlp3, MOI.Nonlinear.SparseReverseMode(), x3)
block3 = MOI.NLPBlockData(evaluator3)
MOI.set(edge3, MOI.NLPBlock(), block3)
MOI.set(edge3, MOI.ObjectiveSense(), MOI.MAX_SENSE)