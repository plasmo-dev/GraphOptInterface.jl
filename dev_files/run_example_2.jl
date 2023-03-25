# build BOI model with two sub-blocks
include(joinpath(@__DIR__,"example_2_two_sub_blocks.jl"))

# load linear solver code (this will be migrated to MadNLP)
include(joinpath(@__DIR__(),"../schur_optimizer/schur_linear.jl"))

MOI.optimize!(optimizer)