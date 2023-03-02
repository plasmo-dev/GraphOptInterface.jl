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
    # The block containing nodes and edges
    block::Block
    
    # The abstract-differentiation backend
    backend::B

    # edge data for evaluations
    edge_data::OrderedDict{Edge,EdgeData}
    num_variables::Int64
    num_constraints::Int64
    nnz_hess::Int64
    nnz_jac::Int64

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

        count_constraints = 0
        count_nnzh = 0
        count_nnzj = 0
        for edge in all_edges(block)

            nlp_data = MOI.get(edge, MOI.NLPBlock())
            MOI.initialize(nlp_data.evaluator, [:Grad,:Hess,:Jac])
            
            # edge columns
            columns = column_inds(edge)
            ninds = range(columns[1],columns[end])

            # edge rows
            n_con_edge = num_constraints(edge)
            minds = count_constraints+1 : count_constraints+n_con_edge
            count_constraints += n_con_edge

            # edge hessian indices
            hessian_sparsity = MOI.hessian_lagrangian_structure(edge)
            nnzs_hess_inds = count_nnzh+1 : count_nnzh + length(hessian_sparsity)
            count_nnzh += length(hessian_sparsity)

            # edge jacobian indices
            jacobian_sparsity = MOI.jacobian_structure(edge)
            nnzs_jac_inds = count_nnzj+1 : count_nnzj + length(jacobian_sparsity)
            count_nnzj += length(jacobian_sparsity)

            edge_data[edge] = EdgeData(ninds, minds, nnzs_jac_inds, nnzs_hess_inds)
        end

        num_variables = MOI.get(block, MOI.NumberOfVariables())

        return new{B}(
            block,
            backend,
            edge_data,
            num_variables,
            count_constraints,
            count_nnzh,
            count_nnzj,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )
    end
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
    # evaluate gradient for each edge and sum
    f_locals = [spzeros(length(f)) for _ = 1:length(evaluator.block.edges)]
    edges = all_edges(evaluator.block)
    Threads.@threads for i = 1:length(edges)
        edge = edges[i]
        ninds = evaluator.edge_data[edge].ninds
        MOI.eval_objective_gradient(edge, view(f_locals[i],ninds), view(x,ninds))
    end
    f[:] = sum(f_locals)
    return nothing
end

### Eval_G_CB

function MOI.eval_constraint(evaluator::BlockEvaluator, c, x)
	Threads.@threads for edge in all_edges(evaluator.block)
		ninds = evaluator.edge_data[edge].ninds
		minds = evaluator.edge_data[edge].minds
		MOI.eval_constraint(edge, view(c, minds), view(x, ninds))
	end
    return nothing
end

### Eval_Jac_G_CB




# function MOI.jacobian_structure(evaluator::BlockEvaluator)::Vector{Tuple{Int64,Int64}}
#     I = Vector{Int64}(undef, evaluator.nnz_jac) # row indices
#     J = Vector{Int64}(undef, evaluator.nnz_jac) # column indices

#     for edge in all_edges(evaluator.block)
#         edge_data = evaluator.edge_data[edge]
#         isempty(edge_data.nnzs_jac_inds) && continue

#         offset_i = edge_data.minds[1] - 1
#         offset_j = edge_data.ninds[1] - 1
        
#         II = view(I, edge_data.nnzs_jac_inds)
#         JJ = view(J, edge_data.nnzs_jac_inds)

#         edge_jacobian_sparsity = MOI.jacobian_structure(edge)

#         for (k,(row,col)) in enumerate(edge_jacobian_sparsity)
#             II[k+offset_i] = row
#             JJ[k+offset_j] = col
#         end
#     end
#     jacobian_sparsity = collect(zip(I, J))
#     return jacobian_sparsity
# end


# function MOI.jacobian_structure(model::Optimizer)
#     J = MOI.jacobian_structure(model.qp_data)
#     offset = length(model.qp_data)
#     if length(model.nlp_data.constraint_bounds) > 0
#         for (row, col) in MOI.jacobian_structure(model.nlp_data.evaluator)
#             push!(J, (row + offset, col))
#         end
#     end
#     return J
# end

# function MOI.eval_constraint_jacobian(model::Optimizer, values, x)
#     offset = MOI.eval_constraint_jacobian(model.qp_data, values, x)
#     nlp_values = view(values, (offset+1):length(values))
#     MOI.eval_constraint_jacobian(model.nlp_data.evaluator, nlp_values, x)
#     return
# end

# function MOI.jacobian_structure(d::OptiGraphNLPEvaluator)
#     nnzs_jac_inds = d.nnzs_jac_inds
#     I = Vector{Int64}(undef, d.nnz_jac)
#     J = Vector{Int64}(undef, d.nnz_jac)
#     #@blas_safe_threads for k=1:length(modelnodes)
#     for k in 1:length(d.nlps)
#         isempty(nnzs_jac_inds[k]) && continue
#         offset_i = d.minds[k][1] - 1
#         offset_j = d.ninds[k][1] - 1
#         II = view(I, nnzs_jac_inds[k])
#         JJ = view(J, nnzs_jac_inds[k])
#         _jacobian_structure(d.nlps[k], II, JJ)
#         II .+= offset_i
#         JJ .+= offset_j
#     end
#     jacobian_sparsity = collect(zip(I, J)) # return Tuple{Int64,Int64}[]
#     return jacobian_sparsity
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