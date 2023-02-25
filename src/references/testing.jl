using MathOptInterface
const MOI = MathOptInterface
const MOIU = MOI.Utilities

# use a dummy model to test with
model = MOIU.CachingOptimizer(
        MOIU.UniversalFallback(MOIU.Model{Float64}()),
        MOIU.AUTOMATIC,
    )

x = MOI.add_variables(model, length(c));


using JuMP

raw_index(v::MOI.VariableIndex) = v.value
model = Model()
@variable(model, x)
@variable(model, y)
@NLobjective(model, Min, sin(x) + sin(y))
values = zeros(2)
x_index = raw_index(JuMP.index(x))
y_index = raw_index(JuMP.index(y))
values[x_index] = 2.0
values[y_index] = 3.0
d = NLPEvaluator(model)

MOI.initialize(d, [:Grad])
MOI.eval_objective(d, values) 
