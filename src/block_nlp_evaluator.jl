struct EdgeData
    column_indices::Union{UnitRange{Int64},Vector{Int64}}
    row_indices::Union{UnitRange{Int64},Vector{Int64}}
    nnzs_jac_inds::UnitRange{Int}
    nnzs_hess_inds::UnitRange{Int}
end

struct BlockData
    num_variables::Int64
    num_constraints::Int64
    nnz_jac::Int64
    nnz_hess::Int64
    edge_data::OrderedDict{Edge,EdgeData}
    sub_block_data::OrderedDict{BlockIndex,BlockData}
    function BlockData()
        block_data = new(
            0,
            0,
            0,
            0,
            OrderedDict{Edge,EdgeData}(),
            OrderedDict{BlockIndex,BlockData}()
        )
        return block_data
    end

    function BlockData(
        num_variables::Int64,
        num_constraints::Int64,
        nnz_jac::Int64,
        nnz_hess::Int64,
        edge_data::OrderedDict{Edge,EdgeData},
        sub_block_data::OrderedDict{BlockIndex,BlockData}
    )
        return new(
            num_variables,
            num_constraints,
            nnz_jac,
            nnz_hess,
            edge_data,
            sub_block_data
        )
    end
end

function _build_column_data(block::Block)
    for sub_block in block.columns

    end
end

function build_block_data(
    block::Block, 
    requested_features::Vector{Symbol};
    offset_columns=0,
    offset_rows=0,
    offset_nnzj=0,
    offset_nnzh=0
    )
    
    # nodes / variables
    count_columns = 0
    node_columns = Dict{Node,UnitRange}()
    for node in block.nodes
        num_vars = _num_variables(node)
        node_columns[node] = (count_columns+1:count_columns+num_vars).+offset_columns
        count_columns += num_vars
    end

    # loop through sub-blocks here



    # edges / everything else
    count_rows = 0
    count_nnzh = 0
    count_nnzj = 0
    edge_data = OrderedDict{Edge,EdgeData}()
    for edge in block.edges
        nlp_data = MOI.get(edge, MOI.NLPBlock())
        MOI.initialize(nlp_data.evaluator, requested_features)

        # map variables to evaluator columns
        # we treat an edge with one node as a special case and just unit ranges
        if edge isa Edge{NTuple{1,Node}}
            node = edge.elements[1]
            columns = node_columns[node.index]
        else
            nvs = node_variable_indices(block, edge)#::Vector{NodeVariableIndex}
            columns = Int64[]
            for nvi in nvs
                node = nvi.node
                # BUG: we don't have the sub-block nodes yet to index here
                push!(columns, node_columns[node][nvi.index.value])
            end
        end

        # edge rows
        n_con_edge = _num_constraints(edge)
        rows = (count_rows+1:count_rows+n_con_edge).+offset_rows
        count_rows += n_con_edge

        # edge jacobian indices
        if :Jac in requested_features
            jacobian_sparsity = MOI.jacobian_structure(edge)
            nnzs_jac_inds = (count_nnzj+1:count_nnzj+length(jacobian_sparsity)).+offset_nnzj
            count_nnzj += length(jacobian_sparsity)
        end

        # edge hessian indices
        if :Hess in requested_features
            hessian_sparsity = MOI.hessian_lagrangian_structure(edge)
            nnzs_hess_inds = (count_nnzh+1:count_nnzh+length(hessian_sparsity)).+offset_nnzh
            count_nnzh += length(hessian_sparsity)
        end

        edge_data[edge] = EdgeData(columns, rows, nnzs_jac_inds, nnzs_hess_inds)
    end

    # update offsets for sub-blocks
    offset_columns = offset_columns+count_columns
    offset_rows = offset_rows+count_rows
    offset_nnzj = offset_nnzj+count_nnzj
    offset_nnzh = offset_nnzh+count_nnzh

    sub_block_data_map = OrderedDict{BlockIndex,BlockData}()
    for sub_block in block.sub_blocks
        sub_block_data = build_block_data(
            sub_block,
            requested_features;
            offset_columns=offset_columns,
            offset_rows=offset_rows,
            offset_nnzj=offset_nnzj,
            offset_nnzh=offset_nnzh
        )
        offset_columns = offset_columns+sub_block_data.num_variables
        offset_rows = offset_rows+sub_block_data.num_constraints
        offset_nnzj = offset_nnzj+sub_block_data.nnz_jac
        offset_nnzh = offset_nnzh+sub_block_data.nnz_hess
        sub_block_data_map[sub_block.index] = sub_block_data
    end

    # update parent block rows, columns, etc...
    if length(sub_block_data_map) > 0
        count_columns += sum(sb.num_variables for sb in values(sub_block_data_map))
        count_rows += sum(sb.num_constraints for sb in values(sub_block_data_map))
        count_nnzj += sum(sb.nnz_hess for sb in values(sub_block_data_map))
        count_nnzh += sum(sb.nnz_jac for sb in values(sub_block_data_map))
    end

    return BlockData(
        count_columns,
        count_rows,
        count_nnzj,
        count_nnzh,
        edge_data,
        sub_block_data_map
    )
end

"""
    BlockEvaluator(
        block::Block
    )
Create `Evaluator`, a subtype of `MOI.AbstractNLPEvaluator`, from `Model`.
"""
mutable struct BlockEvaluator{B} <: MOI.AbstractNLPEvaluator
    # The block containing nodes and edges
    block::Block
    
    # The abstract-differentiation backend
    backend::B

    block_data::BlockData
    
    eval_objective_timer::Float64
    eval_constraint_timer::Float64
    eval_objective_gradient_timer::Float64
    eval_constraint_jacobian_timer::Float64
    eval_hessian_lagrangian_timer::Float64

    function BlockEvaluator(
        block::Block,
        backend::B=MOI.Nonlinear.SparseReverseMode()
    ) where {B<:MOI.Nonlinear.AbstractAutomaticDifferentiation}

        count_columns = 0
        count_rows = 0
        count_nnzh = 0
        count_nnzj = 0
        block_data = BlockData()

        return new{B}(
            block,
            backend,
            block_data,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )
    end
end

function Base.string(evaluator::BlockEvaluator)
    return """Block NLP Evaluator
    """
end
Base.print(io::IO, evaluator::BlockEvaluator) = print(io, string(evaluator))
Base.show(io::IO, evaluator::BlockEvaluator) = print(io, evaluator)

function MOI.initialize(evaluator::BlockEvaluator, requested_features::Vector{Symbol})
    evaluator.block_data = build_block_data(evaluator.block, requested_features)
    return
end

### Eval_F_CB

function MOI.eval_objective(evaluator::BlockEvaluator, x)
    return eval_objective(evaluator.block, evaluator.block_data, x)
end

function eval_objective(block::Block, block_data::BlockData, x)
    # initialize 
    obj = Threads.Atomic{Float64}(0.)

    # evaluate root edges
    Threads.@threads for i = 1:length(block.edges)
        edge = block.edges[i]
        edge_data = block_data.edge_data[edge]
        columns = edge_data.column_indices
        Threads.atomic_add!(obj, MOI.eval_objective(edge, view(x, columns)))
    end

    # evaluate sub blocks
    Threads.@threads for i = 1:length(block.sub_blocks)
        sub_block = block.sub_blocks[i]
        sub_block_data = block_data.sub_block_data[sub_block.index]
        # TODO: think about how to pass only the block columns instead of `x`
        Threads.atomic_add!(
            obj,
            eval_objective(sub_block, sub_block_data, x)
        )
    end
    return obj.value
end

### Eval_Grad_F_CB

function MOI.eval_objective_gradient(evaluator::BlockEvaluator, g, x)
    eval_objective_gradient(evaluator.block, evaluator.block_data, g, x)
end

function eval_objective_gradient(block::Block, block_data::BlockData, g, x)
    edges = block.edges
    Threads.@threads for i = 1:length(edges)
        edge = edges[i]
        edge_data = block_data.edge_data[edge]
        columns = edge_data.column_indices
        MOI.eval_objective_gradient(edge, view(g,columns), view(x,columns))
    end

    # evaluate sub blocks
    Threads.@threads for i = 1:length(block.sub_blocks)
        sub_block = block.sub_blocks[i]
        sub_block_data = block_data.sub_block_data[sub_block.index]
        eval_objective_gradient(sub_block, sub_block_data, g, x)
    end
    
    return
end


### Eval_G_CB

function MOI.eval_constraint(evaluator::BlockEvaluator, c, x)
    eval_constraint(evaluator.block, evaluator.block_data, c, x)
end

function eval_constraint(block::Block, block_data::BlockData, c, x)
    edges = block.edges
    Threads.@threads for i = 1:length(edges)
        edge = edges[i]
        edge_data = block_data.edge_data[edge]
        columns = edge_data.column_indices
        rows = edge_data.row_indices
        MOI.eval_constraint(edge, view(c, rows), view(x, columns))
    end

    # evaluate sub blocks
    Threads.@threads for i = 1:length(block.sub_blocks)
        sub_block = block.sub_blocks[i]
        sub_block_data = block_data.sub_block_data[sub_block.index]
        eval_constraint(sub_block, sub_block_data, c, x)
    end

    return
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