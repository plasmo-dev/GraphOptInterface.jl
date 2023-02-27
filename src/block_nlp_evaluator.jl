struct EdgeData
    ninds::UnitRange{Int}
    minds::UnitRange{Int}
    nnzs_jac_inds::UnitRange{Int}
    nnzs_hess_inds::UnitRange{Int}
end

"""
    BlockEvaluator(
        model::Model,
        backend::AbstractAutomaticDifferentiation,
        ordered_variables::Vector{MOI.VariableIndex},
    )
Create `Evaluator`, a subtype of `MOI.AbstractNLPEvaluator`, from `Model`.
"""
mutable struct BlockEvaluator{B} <: MOI.AbstractNLPEvaluator
    # The internal datastructure.
    block::Block
    
    # The abstract-differentiation backend
    backend::B

    # edge data for evaluations
    edge_data::OrderedDict{Edge,EdgeData}

    eval_objective_timer::Float64
    eval_constraint_timer::Float64
    eval_objective_gradient_timer::Float64
    eval_constraint_jacobian_timer::Float64
    eval_hessian_lagrangian_timer::Float64

    function BlockEvaluator(
        block::Block,
        backend::B=MOI.Nonlinear.SparseReverseMode()
    ) where {B<:MOI.Nonlinear.AbstractAutomaticDifferentiation}

        edge_data = Dict{Edge,EdgeData}()

        # count_variables = 0
        count_constraints = 0
        count_nnzh = 0
        count_nnzj = 0
        for edge in all_edges(block)

            nlp_data = MOI.get(edge, MOI.NLPBlock())
            MOI.initialize(nlp_data.evaluator, [:Grad,:Hess,:Jac])
            

            # variable indices
            # n_var_edge = num_variables(edge)
            # ninds = count_variables+1 : count_variables+n_var_edge
            # count_variables += n_var_edge
            columns = column_inds(edge)
            ninds = range(columns[1],columns[end])

            n_con_edge = num_constraints(edge)
            minds = count_constraints+1 : count_constraints+n_con_edge
            count_constraints += n_con_edge

            # hessian indices
            hessian_sparsity = MOI.hessian_lagrangian_structure(edge)
            nnzs_hess_inds = count_nnzh+1 : count_nnzh + length(hessian_sparsity)
            count_nnzh += length(hessian_sparsity)

            # jacobian indices
            jacobian_sparsity = MOI.jacobian_structure(edge)
            nnzs_jac_inds = count_nnzj+1 : count_nnzj + length(jacobian_sparsity)
            count_nnzj += length(jacobian_sparsity)

            edge_data[edge] = EdgeData(ninds, minds, nnzs_hess_inds, nnzs_jac_inds)
        end

        return new{B}(
            block,
            backend,
            edge_data,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )
    end
end


function num_variables(node::Node)
    return MOI.get(node, MOI.NumberOfVariables())
end

function num_constraints(edge::Edge)
    n_con = 0
    for (F,S) in MOI.get(edge, MOI.ListOfConstraintTypesPresent())
        n_con += MOI.get(edge, MOI.NumberOfConstraints{F,S}())
    end
    nlp_block = MOI.get(edge, MOI.NLPBlock())
    n_con += length(nlp_block.constraint_bounds)
    return n_con
end

function MOI.initialize(evaluator::BlockEvaluator, requested_features::Vector{Symbol})
    for edge in all_edges(evaluator.block)
        MOI.initialize(edge, requested_features)
    end
end

### Eval_F_CB

function MOI.eval_objective(evaluator::BlockEvaluator, x)
	obj = Threads.Atomic{Float64}(0.)
	for edge in all_edges(evaluator.block)
        ninds = evaluator.edge_data[edge].ninds
        Threads.atomic_add!(obj, MOI.eval_objective(edge, view(x, ninds)))
    end
    return obj.value
end

### Eval_Grad_F_CB

function MOI.eval_objective_gradient(evaluator::BlockEvaluator, f, x)
    block = evaluator.block
    for edge in all_edges(block)
        ninds = evaluator.edge_data[edge].ninds
        MOI.eval_objective_gradient(edge, view(f,ninds), view(x,ninds))
    end
    return nothing
end

### Eval_G_CB

# function eval_constraint(evaluator::BlockEvaluator, c, x)
#     #@blas_safe_threads 
#     block = evaluator.block
# 	for edge in all_edges(block)
# 		ninds = column_inds(block, edge)
# 		minds = row_inds(block, edge)
# 		MOI.eval_constraint(edge, view(c, minds), view(x, ninds))
# 	end
#     return nothing
# end

# ### Eval_H_CB

# function MOI.hessian_lagrangian_structure(evaluator::BlockEvaluator)
    
# 	nnzs_hess_inds = d.nnzs_hess_inds
#     nnz_hess = d.nnz_hess
#     nodes = d.optinodes

#     I = Vector{Int64}(undef, d.nnz_hess)
#     J = Vector{Int64}(undef, d.nnz_hess)

#     #@blas_safe_threads for k=1:length(optinodes)
# 	for edge in all_edges(blk)
        
# 		ninds = column_inds(edge)
# 		nnzs_hess_inds = get_hess_inds(edge)

# 		isempty(nnzs_hess_inds) && continue

# 		offset = ninds[1]-1

#         II = view(I, nnzs_hess_inds[k])
#         JJ = view(J, nnzs_hess_inds[k])


#         madnlp_optimizer = inner_optimizer(optinodes[k])
#         cnt = 1
#         for (row, col) in  MOI.hessian_lagrangian_structure(madnlp_optimizer)
#             II[cnt], JJ[cnt] = row, col
#             cnt += 1
#         end
#         II.+= offset
#         JJ.+= offset
#     end
#     return nothing
# end