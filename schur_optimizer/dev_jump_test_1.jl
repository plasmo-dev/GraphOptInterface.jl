using JuMP
using MadNLP

c1 = [1.0, 2.0, 3.0]
w1 = [0.3, 0.5, 1.0]
C1 = 3.2

m = Model()

@variable(m, x1[1:3] >= 0)
obj1 = @expression(m, sum(c1[i]*x1[i] for i = 1:3))
@constraint(m, sum(w1[i]*x1[i] for i=1:3) <= C1)
@NLconstraint(m, 1 + sqrt(x1[1]) <= 2.0)

c2 = [2.0, 3.0, 4.0]
w2 = [0.2, 0.1, 1.2]
C2 = 2.0

@variable(m, x2[1:3] >= 0)
obj2 = @expression(m, sum(c2[i]*x2[i] for i = 1:3))
@constraint(m, sum(w2[i]*x2[i] for i=1:3) <= C2)
@NLconstraint(m, 1 + sqrt(x2[2]) <= 3)


# linking between nodes
#x1[1], x2[1], x2[3]
@constraint(m, x1[1] == x2[1])
@NLconstraint(m, 1 + sqrt(x1[1]) + x2[3]^3 <= 5)

@objective(m, Max, obj1 + obj2)

set_optimizer(m, MadNLP.Optimizer)
optimize!(m)


### pull out the madnlp optimizer to test derivatives

joptimizer = m.moi_backend.optimizer.model
jnlp = joptimizer.nlp
jsolver = MadNLP.MadNLPSolver(jnlp)
MadNLP.solve!(jsolver)

x_moi = ones(6)
c_moi = zeros(6)
MOI.eval_constraint(joptimizer, c_moi, x_moi)
c_upper = jnlp.meta.ucon

x_moi = ones(6)
moi_obj = MOI.eval_objective(joptimizer, x_moi)

# # gradient
grad_moi = zeros(6)
MOI.eval_objective_gradient(joptimizer, grad_moi, x_moi)

jac_structure_moi = MOI.jacobian_structure(joptimizer)
jac_values_moi = zeros(length(jac_structure_moi))
MOI.eval_constraint_jacobian(joptimizer, jac_values_moi, x_moi)

hess_lag_structure_moi = MOI.hessian_lagrangian_structure(moi_evaluator)
hess_values_moi = zeros(length(hess_lag_structure_moi))
MOI.eval_hessian_lagrangian(joptimizer, hess_values_moi, x_moi, 1.0, ones(6))