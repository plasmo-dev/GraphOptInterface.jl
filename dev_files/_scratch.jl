# schur_opt = SchurOptions(
# 	partition=partition, 
# 	subproblem_solver=MadNLP.LapackCPUSolver,
# 	subproblem_solver_options=MadNLP.LapackOptions()
# )

# Test schur linear solver using the solver KKT system
# kkt = MadNLP.get_kkt(solver1.kkt)
# schur_linear = SchurLinearSolver(kkt; opt=schur_opt)
# MadNLP.factorize!(schur_linear)
# x_test = MadNLP.solve!(schur_linear, ones(length(partition)))

# solver = MadNLP.MadNLPSolver{T,KKTSystem}(nlp, mad_opt, schur_opt) where {T,KKTSystem<:MadNLP.AbstractKKTSystem{T}}

# madnlpsolver = MadNLP.MadNLPSolver(nlp; linear_solver=SchurLinearSolver, partition=partition)

# try instantiating a MadNLP solver
# schur_opt = SchurOptions(partition=partition)
# mad_opt = MadNLPOptions(linear_solver=SchurLinearSolver)