include(joinpath(@__DIR__,"example_2_two_sub_blocks.jl"))

include(joinpath(@__DIR__(),"../schur_optimizer/schur_linear.jl"))

using MadNLP

nlp = BlockNLPModel(optimizer)
partition = get_partition_vector(nlp)

# run with pure umf-pack as a check
solver1 = MadNLPSolver(nlp)
MadNLP.solve!(solver1)

# run with schur
solver2 = MadNLP.MadNLPSolver(nlp, linear_solver=SchurLinearSolver, partition=partition)
MadNLP.solve!(solver2)