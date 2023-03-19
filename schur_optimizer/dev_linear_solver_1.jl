include("dev_schur_optimizer.jl")

include("schur_linear.jl")

using MadNLP

nlp = BlockNLPModel(optimizer)
partition = get_partition_vector(nlp)
schur_opt = SchurOptions(partition=partition)

# TODO: test that the linear solver works
madnlpsolver = MadNLP.MadNLPSolver(nlp)

kkt = MadNLP.get_kkt(madnlpsolver.kkt)

schur_linear = SchurLinearSolver(kkt; opt=schur_opt)

# this works, but not sure how to test. the subproblem solvers are also throwing buffer errors for printing.
MadNLP.factorize!(schur_linear)


# TODO: figure out how to pass schur linear solver with options
# madnlpsolver = MadNLP.MadNLPSolver(nlp; linear_solver=SchurLinearSolver, partition=partition)