include("dev_schur_optimizer.jl")
include("schur_linear.jl")

using MadNLP

nlp = BlockNLPModel(optimizer)

# partition = get_partition_vector(nlp)
# schur_opt = SchurOptions()
# schur_opt.partition = partition

madnlpsolver = MadNLP.MadNLPSolver(nlp)