[![CI](https://github.com/plasmo-dev/GraphOptInterface.jl/workflows/CI/badge.svg)](https://github.com/plasmo-dev/GraphOptInterface.jl/actions)
[![codecov](https://codecov.io/gh/plasmo-dev/GraphOptInterface.jl/branch/main/graph/badge.svg)](https://app.codecov.io/github/plasmo-dev/GraphOptInterface.jl)
[![](https://img.shields.io/badge/docs-latest-blue.svg)](https://plasmo-dev.github.io/GraphOptInterface.jl/dev/)

# GraphOptInterface.jl

A graph data structure for communicating to optimization solvers.

## Simple Example

```julia
using MathOptInterface
const MOI = MathOptInterface

using GraphOptInterface
const GOI = GraphOptInterface

##################################################
# node 1
##################################################
graph = GOI.OptiGraph()

node1 = GOI.add_node(graph)
x1 = MOI.add_variables(node1, 3)

# constraints/bounds on variables
for x_i in x1
   MOI.add_constraint(node1, x_i, MOI.GreaterThan(0.0))
   MOI.add_constraint(node1, x_i, MOI.LessThan(5.0))
end

c1 = [1.0, 2.0, 3.0]
w1 = [0.3, 0.5, 1.0]
C1 = 3.2

# set node1 objective
MOI.set(
	node1,
	MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
	MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(c1, x1), 0.0)
)
MOI.set(node1, MOI.ObjectiveSense(), MOI.MAX_SENSE)

# add node1 constraint
ci = MOI.add_constraint(
    node1,
    MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(w1, x1), 0.0),
    MOI.LessThan(C1)
)

# add node1 nonlinear constraint
MOI.Nonlinear.add_constraint(node1, :(1.0 + sqrt($(x1[1]))), MOI.LessThan(5.0))


##################################################
# node 2
##################################################
node2 = GOI.add_node(graph)
x2 = MOI.add_variables(node2, 3)
for x_i in x2
   MOI.add_constraint(node2, x_i, MOI.GreaterThan(0.0))
   MOI.add_constraint(node2, x_i, MOI.LessThan(5.0))
end

c2 = [2.0, 3.0, 4.0]
w2 = [0.2, 0.1, 1.2]
C2 = 2.0
MOI.set(
	node2,
	MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
	MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(c2, x2), 0.0),
)
MOI.set(node2, MOI.ObjectiveSense(), MOI.MAX_SENSE)
MOI.add_constraint(
    node2,
    MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(w2, x2), 0.0),
    MOI.LessThan(C2)
)
MOI.Nonlinear.add_constraint(node2, :(1.0 + sqrt($(x2[2]))), MOI.LessThan(3.0))

##################################################
# edge 3 - couple node1 and node2
##################################################
edge3 = GOI.add_edge(graph, (node1, node2))

# add coupling variables to the edge and map to associated node variables
x1 = MOI.add_variable(edge3, node1, MOI.VariableIndex(1))
x2 = MOI.add_variable(edge3, node2, MOI.VariableIndex(1))
x3 = MOI.add_variable(edge3, node2, MOI.VariableIndex(3))

MOI.add_constraint(
    edge3,
    MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0,-1.0], [x1,x2]), 0.0),
    MOI.EqualTo(0.0)
)
MOI.Nonlinear.add_constraint(edge3, :(1.0 + sqrt($(x1)) + $(x3)^3), MOI.LessThan(5.0))
MOI.set(edge3, MOI.ObjectiveSense(), MOI.MAX_SENSE)
```