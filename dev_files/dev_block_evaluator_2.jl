include(joinpath(@__DIR__,"example_2_two_sub_blocks.jl"))


##################################################
# test out block evaluator
##################################################
block_evaluator = BOI.BlockEvaluator(optimizer.block)
MOI.initialize(block_evaluator, [:Grad, :Jac, :Hess])

bd = block_evaluator.block_data

x_block = ones(9)

# objective
block_obj = MOI.eval_objective(block_evaluator, x_block)
@assert block_obj == 24.0

# constraints
c_block = zeros(9)
MOI.eval_constraint(block_evaluator, c_block, x_block)

# gradient
grad_block = zeros(9)
MOI.eval_objective_gradient(block_evaluator, grad_block, x_block)

jac_structure_block = MOI.jacobian_structure(block_evaluator)
jac_values_block = zeros(length(jac_structure_block))
MOI.eval_constraint_jacobian(block_evaluator, jac_values_block, x_block)

hess_lag_structure_block = MOI.hessian_lagrangian_structure(block_evaluator)
hess_values_block = zeros(length(hess_lag_structure_block))
MOI.eval_hessian_lagrangian(block_evaluator, hess_values_block, x_block, 1.0, ones(9))