include(joinpath(@__DIR__,"example_2_two_sub_blocks.jl"))

include(joinpath(@__DIR__,"../schur_optimizer/edge_model.jl"))

include(joinpath(@__DIR__,"../schur_optimizer/block_nlp_evaluator.jl"))

include(joinpath(@__DIR__(),"../schur_optimizer/schur_optimizer.jl"))

# edge_model = build_edge_model(edge0)

##################################################
# test out block evaluator
##################################################
evaluator = BlockNLPEvaluator(graph)

MOI.initialize(evaluator, [:Grad, :Jac, :Hess])

x_block = ones(9)

# objective
block_obj = MOI.eval_objective(evaluator, x_block)
@assert block_obj == 24.0

# # constraints
c_block = zeros(9)
MOI.eval_constraint(evaluator, c_block, x_block)

# # gradient
grad_block = zeros(9)
MOI.eval_objective_gradient(evaluator, grad_block, x_block)

jac_structure_block = MOI.jacobian_structure(evaluator)
jac_values_block = zeros(length(jac_structure_block))
MOI.eval_constraint_jacobian(evaluator, jac_values_block, x_block)

hess_lag_structure_block = MOI.hessian_lagrangian_structure(evaluator)
hess_values_block = zeros(length(hess_lag_structure_block))
MOI.eval_hessian_lagrangian(evaluator, hess_values_block, x_block, 1.0, ones(9))

