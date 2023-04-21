# build BOI model with two sub-blocks
include(joinpath(@__DIR__,"example_2_two_sub_blocks.jl"))

include(joinpath(@__DIR__,"../schur_optimizer/edge_model.jl"))

include(joinpath(@__DIR__,"../schur_optimizer/utils.jl"))

include(joinpath(@__DIR__,"../schur_optimizer/block_nlp_evaluator.jl"))

# schur_optimizer.jl will be migrated to MadNLP
include(joinpath(@__DIR__(),"../schur_optimizer/schur_optimizer.jl"))

# load linear solver code (this will be migrated to MadNLP)
include(joinpath(@__DIR__(),"../schur_optimizer/schur_linear.jl"))

optimizer = SchurOptimizer(graph)

nlp = BlockNLPModel(optimizer)

# MOI.optimize!(optimizer)