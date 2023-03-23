include("dev_schur_optimizer.jl")

include("schur_linear.jl")

using MadNLP

nlp = BlockNLPModel(optimizer)

# TODO: test that the block evaluator actually works
madnlpsolver = MadNLP.MadNLPSolver(nlp)
MadNLP.solve!(madnlpsolver)


partition = get_partition_vector(nlp)
schur_opt = SchurOptions(partition=partition)
# kkt = MadNLP.get_kkt(madnlpsolver.kkt)
schur_linear = SchurLinearSolver(kkt; opt=schur_opt)

mad_opt = MadNLPOptions(linear_solver=SchurLinearSolver)
MadNLP.MadNLPSolver(nlp, mad_opt, schur_opt)

# # this works, but not sure how to test. the subproblem solvers are also throwing buffer errors for printing.
# MadNLP.factorize!(schur_linear)
solver = MadNLP.MadNLPSolver{T,MadNLP.KKTSystem}(nlp, mad_opt, schur_opt) where {T, MadNLP.KKTSystem<:MadNLP.AbstractKKTSystem{T}}


# TODO: figure out how to pass schur linear solver with options
#madnlpsolver = MadNLP.MadNLPSolver(nlp; linear_solver=SchurLinearSolver, partition=partition)