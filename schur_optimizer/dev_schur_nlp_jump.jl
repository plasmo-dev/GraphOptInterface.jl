using JuMP
using MadNLP

c1 = [1.0, 2.0, 3.0]
w1 = [0.3, 0.5, 1.0]
C1 = 3.2


m = Model()

@variable(m, 0<= x1[1:3] <= 5)
@NLobjective(m, Max, sum(c1[i]*x1[i] for i = 1:3))
@NLconstraint(m, 1 + sqrt(x1[1]) <= 5.0)


set_optimizer(m, MadNLP.Optimizer)
optimize!(m)

joptimizer = m.moi_backend.optimizer.model
jnlp = joptimizer.nlp
jsolver = MadNLP.MadNLPSolver(jnlp)
MadNLP.solve!(jsolver)


x_moi = ones(jnlp.meta.nvar)
c_moi = zeros(jnlp.meta.ncon)
MOI.eval_constraint(joptimizer, c_moi, x_moi)
c_upper = jnlp.meta.ucon

moi_obj = MOI.eval_objective(joptimizer, x_moi)

# # gradient
grad_moi = zeros(length(x_moi))
MOI.eval_objective_gradient(joptimizer, grad_moi, x_moi)

jac_structure_moi = MOI.jacobian_structure(joptimizer)
jac_values_moi = zeros(length(jac_structure_moi))
MOI.eval_constraint_jacobian(joptimizer, jac_values_moi, x_moi)

hess_structure_moi = MOI.hessian_lagrangian_structure(joptimizer)
hess_values_moi = zeros(length(hess_structure_moi))
MOI.eval_hessian_lagrangian(joptimizer, hess_values_moi, x_moi, 1.0, ones(length(c_moi)))