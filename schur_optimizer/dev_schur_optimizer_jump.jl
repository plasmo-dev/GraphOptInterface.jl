using JuMP
using MadNLP

c1 = [1.0, 2.0, 3.0]
w1 = [0.3, 0.5, 1.0]
C1 = 3.2


m = Model()

@variable(m, x1[1:3] >= 0)
obj1 = @expression(m, sum(c1[i]*x1[i] for i = 1:3))
@NLconstraint(m, sum(w1[i]*x1[i] for i=1:3) <= C1)
@NLconstraint(m, 1 + sqrt(x1[1]) <= 2.0)

c2 = [2.0, 3.0, 4.0]
w2 = [0.2, 0.1, 1.2]
C2 = 2.0

@variable(m, x2[1:3] >= 0)
obj2 = @expression(m, sum(c2[i]*x2[i] for i = 1:3))
@NLconstraint(m, sum(w2[i]*x2[i] for i=1:3) <= C2)
@NLconstraint(m, 1 + sqrt(x2[2]) <= 3)


# linking between nodes
#x1[1], x2[1], x2[3]
@NLconstraint(m, x1[1] == x2[1])
@NLconstraint(m, 1 + sqrt(x1[1]) + x2[3]^3 <= 5)

@NLobjective(m, Max, obj1 + obj2)

set_optimizer(m, MadNLP.Optimizer)
optimize!(m)

# TODO: test NLP evaluations
moi_evaluator = JuMP.NLPEvaluator(m)

MOI.initialize(moi_evaluator, [:Grad, :Hess, :Jac])

x_moi = ones(6)
moi_obj = MOI.eval_objective(moi_evaluator, x_moi)

# constraints
c_moi = zeros(6)
MOI.eval_constraint(moi_evaluator, c_moi, x_moi)

# gradient
grad_moi = zeros(6)
MOI.eval_objective_gradient(moi_evaluator, grad_moi, x_moi)

jac_structure_moi = MOI.jacobian_structure(moi_evaluator)
jac_values_moi = zeros(length(jac_structure_moi))
MOI.eval_constraint_jacobian(moi_evaluator, jac_values_moi, x_moi)

hess_lag_structure_moi = MOI.hessian_lagrangian_structure(moi_evaluator)
hess_values_moi = zeros(length(hess_lag_structure_moi))
MOI.eval_hessian_lagrangian(moi_evaluator, hess_values_moi, x_moi, 1.0, ones(6))