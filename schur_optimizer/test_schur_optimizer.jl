include("schur_optimizer.jl")

optimizer = SchurOptimizer()

# add node 1 with variables
node1 = BOI.add_node!(optimizer, BOI.BlockIndex(0))
x1 = MOI.add_variables(optimizer, node1, 3)

# add edge1 with constraints for node
edge1 = BOI.add_edge!(optimizer, BOI.BlockIndex(0), node1)

c1 = [1.0, 2.0, 3.0]
w1 = [0.3, 0.5, 1.0]
C1 = 3.2

# set edge1 objective
MOI.set(
	edge1,
	MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
	MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(c1, x1), 0.0),
)
MOI.set(edge1, MOI.ObjectiveSense(), MOI.MAX_SENSE)

# add edge1 constraint
ci = MOI.add_constraint(
    optimizer,
    edge1,
    MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(w1, x1), 0.0),
    MOI.LessThan(C1),
)

# add edge1 nonlinear constraint
nlp1 = MOI.Nonlinear.Model()
MOI.Nonlinear.add_constraint(nlp1, :(1 + sqrt($(x1[1]))), MOI.LessThan(2.0))
evaluator1 = MOI.Nonlinear.Evaluator(nlp1, MOI.Nonlinear.SparseReverseMode(), x1)
block1 = MOI.NLPBlockData(evaluator1)
MOI.set(edge1, MOI.NLPBlock(), block1)


node2 = BOI.add_node!(optimizer, BOI.BlockIndex(0))
x2 = MOI.add_variables(optimizer, node2, 3)
edge2 = BOI.add_edge!(optimizer, BOI.BlockIndex(0), node2)

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
evaluator2 = MOI.Nonlinear.Evaluator(nlp1, MOI.Nonlinear.SparseReverseMode(), x2)
block2 = MOI.NLPBlockData(evaluator2)
MOI.set(edge2, MOI.NLPBlock(), block2)


# NOTE: x1 and x2 need to have a different VariableIndex.
# edge needs all node variables?

edge3 = BOI.add_edge!(optimizer, BOI.BlockIndex(0), (node1, node2))
MOI.add_constraint(
    optimizer,   
    edge3,
    MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0,-1.0], [x1[1],x2[1]]), 0.0),
    MOI.EqualTo(0.0)
)

g = [0.0]

# NOTE: the edge can reference all of the node variables
x = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
MOI.eval_constraint(edge3.model, g, x)



# MOI.initialize(evaluator, [:Grad, :Jac, :JacVec, :ExprGraph])
