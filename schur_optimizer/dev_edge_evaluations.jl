include(joinpath(@__DIR__,"example_1_two_nodes.jl"))

##################################################
# node 1 and edge 1
##################################################
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


##################################################
# node 2 and edge 2
##################################################
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