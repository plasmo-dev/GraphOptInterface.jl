# build BOI model with two nodes
include(joinpath(@__DIR__,"example_1_two_nodes.jl"))

# load linear solver code (this will be migrated to MadNLP)
include(joinpath(@__DIR__(),"../schur_optimizer/schur_linear.jl"))

MOI.optimize!(optimizer)