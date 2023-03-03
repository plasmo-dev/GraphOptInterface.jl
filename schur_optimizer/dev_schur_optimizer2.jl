include("schur_optimizer.jl")

using SparseArrays

optimizer = SchurOptimizer()

##################################################
# node 1 and edge 1
##################################################
node1 = BOI.add_node!(optimizer, BOI.BlockIndex(0))
MOI.add_variables(optimizer, node1, 3)

# constraints/bounds on variables
for x_i in x
   MOI.add_constraint(optimizer, x_i, MOI.ZeroOne())
end

edge1 = BOI.add_edge!(optimizer, BOI.BlockIndex(0), node1)
x1 = BOI.edge_variables(optimizer.block, edge1)

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
MOI.initialize(evaluator1, [:Grad, :Jac, :Hess])
block1 = MOI.NLPBlockData(evaluator1)
MOI.set(edge1, MOI.NLPBlock(), block1)

# evaluate edge 1 NLP functions
# objective
x1_eval = [1.0, 1.0, 1.0]
edge1_obj = MOI.eval_objective(edge1, x1_eval)

# constraints
g1_eval = zeros(2)
MOI.eval_constraint(edge1, g1_eval, x1_eval)

# gradient
grad1_eval = zeros(3)
MOI.eval_objective_gradient(edge1, grad1_eval, x1_eval)

# jacobian
jac_structure1 = MOI.jacobian_structure(edge1)
jac_values1 = zeros(length(jac_structure1))
MOI.eval_constraint_jacobian(edge1, jac_values1, x1_eval)

# hessian
hess_lag_structure1 = MOI.hessian_lagrangian_structure(edge1)
hess_values1 = zeros(length(hess_lag_structure1))
MOI.eval_hessian_lagrangian(edge1, hess_values1, x1_eval, 1.0, ones(2))


# ##################################################
# # node 2 and edge 2
# ##################################################
node2 = BOI.add_node!(optimizer, BOI.BlockIndex(0))

# NOTE: returns optimizer index, not node index
node2_vars = MOI.add_variables(optimizer, node2, 3)
edge2 = BOI.add_edge!(optimizer, BOI.BlockIndex(0), node2)
x2 = BOI.edge_variables(optimizer.block, edge2, node2_vars)

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
MOI.initialize(evaluator2, [:Grad, :Jac, :Hess])
block2 = MOI.NLPBlockData(evaluator2)
MOI.set(edge2, MOI.NLPBlock(), block2)

# evaluate edge 2 NLP functions
# objective
x2_eval = [1.0, 1.0, 1.0]
edge2_obj = MOI.eval_objective(edge2, x2_eval)

# constriants
g2_eval = zeros(2)
MOI.eval_constraint(edge2, g2_eval, x2_eval)

# gradient
grad2_eval = zeros(3)
MOI.eval_objective_gradient(edge2, grad2_eval, x2_eval)

# jacobian
jac_structure2 = MOI.jacobian_structure(edge2)
jac_values2 = zeros(length(jac_structure2))
MOI.eval_constraint_jacobian(edge2, jac_values2, x2_eval)

# hessian
hess_lag_structure2 = MOI.hessian_lagrangian_structure(edge2)
hess_values2 = zeros(length(hess_lag_structure2))
MOI.eval_hessian_lagrangian(edge2, hess_values2, x2_eval, 1.0, ones(2))

##################################################
# edge 3 - couple node1 and node2
##################################################
edge3 = BOI.add_edge!(optimizer, BOI.BlockIndex(0), (node1, node2))

# get local edge variables
x3 = BOI.edge_variables(optimizer.block, edge3, [node1[1], node2[1], node2[3]])

MOI.add_constraint(
    optimizer,   
    edge3,
    MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0,-1.0], [x3[1],x3[2]]), 0.0),
    MOI.EqualTo(0.0)
)
nlp3 = MOI.Nonlinear.Model()
MOI.Nonlinear.add_constraint(nlp3, :(1 + sqrt($(x3[1])) + $(x3[3])^3), MOI.LessThan(5.0))
evaluator3 = MOI.Nonlinear.Evaluator(nlp3, MOI.Nonlinear.SparseReverseMode(), x3)
MOI.initialize(evaluator3, [:Grad, :Jac, :Hess])
block3 = MOI.NLPBlockData(evaluator3)
MOI.set(edge3, MOI.NLPBlock(), block3)

# evaluate edge 3 NLP functions
# specify a sparse vector of variable indices and values
x3_eval = [1.0, 1.0, 1.0]

# objective
edge3_obj = MOI.eval_objective(edge3, x3_eval)

# constraints
g3_eval = zeros(2)
MOI.eval_constraint(edge3, g3_eval, x3_eval)

# gradient
grad3_eval = zeros(3)
MOI.eval_objective_gradient(edge3, grad3_eval, x3_eval)

# jacobian
jac_structure3 = MOI.jacobian_structure(edge3)
jac_values3 = zeros(length(jac_structure3))
MOI.eval_constraint_jacobian(edge3, jac_values3, x3_eval)

# hessian
hess_lag_structure3 = MOI.hessian_lagrangian_structure(edge3)
hess_values3 = zeros(length(hess_lag_structure3))
MOI.eval_hessian_lagrangian(edge3, hess_values3, x3_eval, 1.0, ones(2))

# block_evaluator = BOI.BlockEvaluator(optimizer.block)

# x_block = ones(6)

# # objective
# block_obj = MOI.eval_objective(block_evaluator, x_block)

# # constraints
# g_block = zeros(6)
# MOI.eval_constraint(block_evaluator, g_block, x_block)

# # gradient
# grad_block = zeros(6)
# MOI.eval_objective_gradient(block_evaluator, grad_block, x_block)

# jac_structure_block = MOI.jacobian_structure(block_evaluator)
# jac_values_block = zeros(length(jac_structure_block))
# MOI.eval_constraint_jacobian(block_evaluator, jac_values_block, x_block)

# hess_lag_structure3 = MOI.hessian_lagrangian_structure(edge3)
# hess_values3 = zeros(length(hess_lag_structure3))
# MOI.eval_hessian_lagrangian(edge3, hess_values3, x3_eval, 1.0, ones(2))