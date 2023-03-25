include(joinpath(@__DIR__,"example_2_two_sub_blocks.jl"))

using MadNLP
##################################################
# test nlp solves
##################################################
nlp = BlockNLPModel(optimizer)

### check derivative evaluations
x_boi = ones(nlp.meta.nvar)
c_boi = zeros(nlp.meta.ncon)
MOI.eval_constraint(nlp.evaluator, c_boi, x_boi)

boi_obj = MOI.eval_objective(nlp.evaluator, x_boi)

# # gradient
grad_boi = zeros(length(x_boi))
MOI.eval_objective_gradient(nlp.evaluator, grad_boi, x_boi)

jac_structure_boi = MOI.jacobian_structure(nlp.evaluator)
jac_values_boi = zeros(length(jac_structure_boi))
MOI.eval_constraint_jacobian(nlp.evaluator, jac_values_boi, x_boi)

hess_structure_boi = MOI.hessian_lagrangian_structure(nlp.evaluator)
hess_values_boi = zeros(length(hess_structure_boi))
MOI.eval_hessian_lagrangian(nlp.evaluator, hess_values_boi, x_boi, 1.0, ones(length(c_boi)))


# solve with vanilla MadNLP, but using the block evaluator
solver = MadNLP.MadNLPSolver(nlp)
# solver.opt.max_iter=300
MadNLP.solve!(solver)