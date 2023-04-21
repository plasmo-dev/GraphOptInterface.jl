include(joinpath(@__DIR__,"utils.jl"))

using DataStructures
using SparseArrays

const MOIU = MOI.Utilities

struct _EmptyNLPEvaluator <: MOI.AbstractNLPEvaluator end

MOI.features_available(::_EmptyNLPEvaluator) = [:Grad, :Jac, :Hess]
MOI.initialize(::_EmptyNLPEvaluator, ::Any) = nothing
MOI.eval_constraint(::_EmptyNLPEvaluator, g, x) = nothing
MOI.jacobian_structure(::_EmptyNLPEvaluator) = Tuple{Int64,Int64}[]
MOI.hessian_lagrangian_structure(::_EmptyNLPEvaluator) = Tuple{Int64,Int64}[]
MOI.eval_constraint_jacobian(::_EmptyNLPEvaluator, J, x) = nothing
MOI.eval_hessian_lagrangian(::_EmptyNLPEvaluator, H, x, σ, μ) = nothing

# Track indices for each edge
struct EdgeIndexes
	column_indices::Union{UnitRange{Int64},Vector{Int64}}
    row_indices::Union{UnitRange{Int64},Vector{Int64}}
    nnzs_jac_inds::UnitRange{Int}
    nnzs_hess_inds::UnitRange{Int}
end

mutable struct BlockData
    num_variables::Int64
    num_constraints::Int64
    nnz_jac::Int64
    nnz_hess::Int64
    all_columns::UnitRange{Int64}
    all_rows::UnitRange{Int64}
    local_columns::UnitRange{Int64}
    local_rows::UnitRange{Int64}

    # node and edge row and column indice
    node_column_dict::OrderedDict{GOI.HyperNode,UnitRange{Int64}}
    edge_index_dict::OrderedDict{GOI.HyperEdge,EdgeIndexes}
    edge_model_dict::OrderedDict{GOI.HyperEdge,EdgeModel}
    sub_block_dict::OrderedDict{GOI.BlockIndex,BlockData}
    function BlockData()
        block_data = new(
            0,
            0,
            0,
            0,
            UnitRange{Int64}(0,0),
            UnitRange{Int64}(0,0),
            UnitRange{Int64}(0,0),
            UnitRange{Int64}(0,0),
            OrderedDict{GOI.HyperNode,UnitRange{Int64}}(),
            OrderedDict{GOI.HyperEdge,EdgeIndexes}(),
            OrderedDict{GOI.HyperEdge,EdgeModel}(),
            OrderedDict{GOI.BlockIndex,BlockData}()
        )
        return block_data
    end

    function BlockData(
        num_variables::Int64,
        num_constraints::Int64,
        nnz_jac::Int64,
        nnz_hess::Int64,
        all_columns::UnitRange{Int64},
        all_rows::UnitRange{Int64},
        node_column_dict::OrderedDict{GOI.HyperNode,UnitRange{Int64}},
        edge_index_dict::OrderedDict{GOI.HyperEdge,EdgeIndexes},
        edge_model_dict::OrderedDict{GOI.HyperEdge,EdgeModel},
        sub_block_dict::OrderedDict{GOI.BlockIndex,BlockData}
    )
        return new(
            num_variables,
            num_constraints,
            nnz_jac,
            nnz_hess,
            all_columns,
            all_rows,
            node_column_dict,
            edge_index_dict,
            edge_model_dict,
            sub_block_dict
        )
    end
end


"""
	Create `Evaluator`, a subtype of `MOI.AbstractNLPEvaluator`, from `Model`.
"""
mutable struct BlockNLPEvaluator <: MOI.AbstractNLPEvaluator
	graph::GOI.Graph
	block_data::BlockData
    eval_objective_timer::Float64
    eval_constraint_timer::Float64
    eval_objective_gradient_timer::Float64
    eval_constraint_jacobian_timer::Float64
    eval_hessian_lagrangian_timer::Float64
end
function BlockNLPEvaluator(graph::GOI.Graph)
    block_data = BlockData()
    return BlockNLPEvaluator(
        graph,
        block_data,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )
end

function _add_sublock_data!(block::GOI.Block, block_data::BlockData)
    for sub_block in block.sub_blocks
        sub_block_data = BlockData()
        block_data.sub_block_dict[sub_block.index] = sub_block_data
        _add_sublock_data!(sub_block, sub_block_data)
    end
    return
end

function _build_node_data!(
    block::GOI.Block,
    block_data::BlockData;
    offset_columns=0
)
    count_columns = 0
    offset_start = offset_columns
    node_columns = OrderedDict{GOI.HyperNode,UnitRange}()

    # get column ranges for each node
    for node in block.nodes
        num_vars = _num_variables(node)
        node_columns[node.index] = (count_columns+1:count_columns+num_vars).+offset_columns
        count_columns += num_vars
    end
    block_data.num_variables = count_columns
    block_data.node_column_dict = node_columns
    block_data.local_columns = offset_columns+1:offset_columns+count_columns
    
    # build sub-blocks. each block stores sub-block data too
    offset_columns = offset_columns+count_columns
    for sub_block in block.sub_blocks
        sub_block_data = block_data.sub_block_dict[sub_block.index]
        _build_node_data!(sub_block, sub_block_data; offset_columns=offset_columns)
        merge!(block_data.node_column_dict, sub_block_data.node_column_dict) 

        # update offset
        # offset_columns = offset_columns+sub_block_data.num_variables
        offset_columns += sub_block_data.num_variables

        # update root block for each sub-block
        block_data.num_variables += sub_block_data.num_variables
    end
    block_data.all_columns = offset_start+1:offset_start+block_data.num_variables
    return
end

function _build_edge_data!(
    block::GOI.Block,
    block_data::BlockData,
    requested_features::Vector{Symbol};
    offset_rows=0,
    offset_nnzj=0,
    offset_nnzh=0
)
    count_rows = 0
    count_nnzj = 0
    count_nnzh = 0
    edge_index_dict = OrderedDict{GOI.HyperEdge,EdgeIndexes}()
    edge_model_dict = OrderedDict{GOI.HyperEdge,EdgeModel}()
    offset_start = offset_rows
    
    # each edge contains constraints, jacobian, and hessian data
    for edge in block.edges
    	edge_model = build_edge_model(edge)
        nlp_data = edge_model.nlp_data
        MOI.initialize(nlp_data.evaluator, requested_features)

        ### map edge variable indices to evaluator columns
        
        # TODO: update for new data structure
        if GOI.is_self_edge(edge)
            # if self-edge, just pull unit ranges
            node_index = first(edge.index.vertices)
            columns = block_data.node_column_dict[node_index]
        else
            # look up column values
            node_variable_indices = GOI.node_variables(edge)
            columns = Int64[]
            for nvi in node_variable_indices
                node_index = nvi[1]
                variable_index = nvi[2]
                node_columns = block_data.node_column_dict[node_index]

                # grab element from unit range
                push!(columns, node_columns[variable_index.value])
            end
        end

        ### edge rows
        n_con_edge = _num_constraints(edge)
        rows = (count_rows+1:count_rows+n_con_edge).+offset_rows
        count_rows += n_con_edge

        ### edge jacobian indices
        if :Jac in requested_features
            jacobian_sparsity = MOI.jacobian_structure(edge_model)
            nnzs_jac_inds = (count_nnzj+1:count_nnzj+length(jacobian_sparsity)).+offset_nnzj
            count_nnzj += length(jacobian_sparsity)
        end

        ### edge hessian indices
        if :Hess in requested_features
            hessian_sparsity = MOI.hessian_lagrangian_structure(edge_model)
            nnzs_hess_inds = (count_nnzh+1:count_nnzh+length(hessian_sparsity)).+offset_nnzh
            count_nnzh += length(hessian_sparsity)
        end

        ### store indexes
        edge_index_dict[edge.index] = EdgeIndexes(columns, rows, nnzs_jac_inds, nnzs_hess_inds)
        edge_model_dict[edge.index] = edge_model
    end
    
    block_data.local_rows = offset_rows+1:offset_rows+count_rows
    block_data.edge_index_dict = edge_index_dict
    block_data.edge_model_dict = edge_model_dict
    block_data.num_constraints = count_rows
    block_data.nnz_jac = count_nnzj
    block_data.nnz_hess = count_nnzh

    # update offsets for sub-blocks
    offset_rows = offset_rows+count_rows
    offset_nnzj = offset_nnzj+count_nnzj
    offset_nnzh = offset_nnzh+count_nnzh

    for sub_block in block.sub_blocks
        sub_block_data = block_data.sub_block_dict[sub_block.index]
        _build_edge_data!(
            sub_block,
            sub_block_data,
            requested_features;
            offset_rows=offset_rows,
            offset_nnzj=offset_nnzj,
            offset_nnzh=offset_nnzh
        )
        offset_rows = offset_rows+sub_block_data.num_constraints
        offset_nnzj = offset_nnzj+sub_block_data.nnz_jac
        offset_nnzh = offset_nnzh+sub_block_data.nnz_hess

        # update root block for each sub-block
        block_data.num_constraints += sub_block_data.num_constraints
        block_data.nnz_jac += sub_block_data.nnz_jac
        block_data.nnz_hess += sub_block_data.nnz_hess
    end

    block_data.all_rows = offset_start+1:offset_start+block_data.num_constraints
    return   
end

function build_block_data(
    graph::GOI.Graph,
    requested_features::Vector{Symbol};
    #TODO: provide multiple AD backends
)
    
    block_data = BlockData()

    _add_sublock_data!(graph.block, block_data)

    _build_node_data!(graph.block, block_data)

    _build_edge_data!(graph.block, block_data, requested_features)

    block_data.all_columns = 1:block_data.num_variables
    block_data.all_rows = 1:block_data.num_constraints

    return block_data
end

# TODO: pass optional mapping of edge indices to AD backends
function MOI.initialize(evaluator::BlockNLPEvaluator, requested_features::Vector{Symbol})
    # create local evaluators
    evaluator.block_data = build_block_data(evaluator.graph, requested_features)
    return
end

### Eval_F_CB

function MOI.eval_objective(evaluator::BlockNLPEvaluator, x::AbstractArray{T}) where T
    return eval_objective(evaluator.graph.block, evaluator.block_data, x)
end

function eval_objective(block::GOI.Block, block_data::BlockData, x::AbstractArray)

    obj = Threads.Atomic{Float64}(0.)

    # evaluate root edges
    Threads.@threads for i = 1:length(block.edges)
    	edge_index = block.edges[i].index
        edge_model = block_data.edge_model_dict[edge_index]
        edge_indexes = block_data.edge_index_dict[edge_index]
        columns = edge_indexes.column_indices
        Threads.atomic_add!(obj, MOI.eval_objective(edge_model, view(x, columns)))
    end

    # evaluate sub blocks
    Threads.@threads for i = 1:length(block.sub_blocks)
        sub_block = block.sub_blocks[i]
        sub_block_data = block_data.sub_block_dict[sub_block.index]
        # TODO: think about how to pass only the block columns instead of `x`
        Threads.atomic_add!(
            obj,
            eval_objective(sub_block, sub_block_data, x)
        )
    end
    return obj.value
end


### Eval_Grad_F_CB

function MOI.eval_objective_gradient(
    evaluator::BlockNLPEvaluator,
    gradient::AbstractArray,
    x::AbstractArray
)
    gradient[:] .= 0.0
    eval_objective_gradient(evaluator.graph.block, evaluator.block_data, gradient, x)
end

function eval_objective_gradient(
    block::GOI.Block,
    block_data::BlockData,
    gradient::AbstractArray,
    x::AbstractArray
)
	#length(block.edges) == 0 && return

    # IDEA: fill each edge gradient as a sparse vector, then sum together
    edge_gradients = [spzeros(length(gradient)) for _ = 1:length(block.edges)]
    Threads.@threads for i = 1:length(block.edges)
        edge_index = block.edges[i].index
        edge_model = block_data.edge_model_dict[edge_index]
        edge_indexes = block_data.edge_index_dict[edge_index]
        columns = edge_indexes.column_indices
        MOI.eval_objective_gradient(edge_model, view(edge_gradients[i],columns), view(x,columns))
    end

    # errors if there are no edges on the block
    gradient[:] .+= sum(edge_gradients)

    # evaluate sub blocks
    Threads.@threads for i = 1:length(block.sub_blocks)
        sub_block = block.sub_blocks[i]
        sub_block_data = block_data.sub_block_dict[sub_block.index]
        eval_objective_gradient(sub_block, sub_block_data, gradient, x)
    end

    return
end

### Eval_G_CB

function MOI.eval_constraint(
    evaluator::BlockNLPEvaluator,
    c::AbstractArray,
    x::AbstractArray
)
    eval_constraint(evaluator.graph.block, evaluator.block_data, c, x)
end

function eval_constraint(
    block::GOI.Block,
    block_data::BlockData,
    c::AbstractArray,
    x::AbstractArray
)
    Threads.@threads for i = 1:length(block.edges)
        edge_index = block.edges[i].index
        edge_model = block_data.edge_model_dict[edge_index]
        edge_indexes = block_data.edge_index_dict[edge_index]
        columns = edge_indexes.column_indices
        rows = edge_indexes.row_indices
        MOI.eval_constraint(edge_model, view(c, rows), view(x, columns))
    end

    # evaluate sub blocks
    Threads.@threads for i = 1:length(block.sub_blocks)
        sub_block = block.sub_blocks[i]
        sub_block_data = block_data.sub_block_dict[sub_block.index]
        eval_constraint(sub_block, sub_block_data, c, x)
    end
    
    return
end

### Eval_Jac_G_CB

function MOI.jacobian_structure(evaluator::BlockNLPEvaluator)::Vector{Tuple{Int64,Int64}}

    I = Vector{Int64}(undef, evaluator.block_data.nnz_jac) # row indices
    J = Vector{Int64}(undef, evaluator.block_data.nnz_jac) # column indices

    jacobian_structure(evaluator.graph.block, evaluator.block_data, I, J)
    jacobian_sparsity = collect(zip(I, J))

    return jacobian_sparsity
end

function jacobian_structure(
    block::GOI.Block,
    block_data::BlockData,
    I::AbstractArray,
    J::AbstractArray
)

    Threads.@threads for i = 1:length(block.edges)
        edge_index = block.edges[i].index
        edge_model = block_data.edge_model_dict[edge_index]
        edge_indexes = block_data.edge_index_dict[edge_index]
        isempty(edge_indexes.nnzs_jac_inds) && continue

        localI = view(I, edge_indexes.nnzs_jac_inds)
        localJ = view(J, edge_indexes.nnzs_jac_inds)
        edge_jacobian_sparsity = MOI.jacobian_structure(edge_model)

        # update view with edge jacobian structure
        for (k,(row,col)) in enumerate(edge_jacobian_sparsity)
            # map coordinates to global indices
            # rows are ordered, so offset is first true row index
            localI[k] = row + edge_indexes.row_indices[1] - 1

            # local columns should ALWAYS be ordered [1...N], so just grab the index
            localJ[k] = edge_indexes.column_indices[col]
        end
    end

    Threads.@threads for i = 1:length(block.sub_blocks)
        sub_block = block.sub_blocks[i]
        sub_block_data = block_data.sub_block_dict[sub_block.index]
        jacobian_structure(sub_block, sub_block_data, I, J)
    end

    return
end

function MOI.eval_constraint_jacobian(
    evaluator::BlockNLPEvaluator,
    jac_values::AbstractArray,
    x::AbstractArray
)
    eval_constraint_jacobian(evaluator.graph.block, evaluator.block_data, jac_values, x)
end

function eval_constraint_jacobian(
    block::GOI.Block,
    block_data::BlockData,
    jac_values::AbstractArray,
    x::AbstractArray
)

    Threads.@threads for i = 1:length(block.edges)
        edge_index = block.edges[i].index
        edge_model = block_data.edge_model_dict[edge_index]
        edge_indexes = block_data.edge_index_dict[edge_index]
        MOI.eval_constraint_jacobian(
            edge_model,
            view(jac_values, edge_indexes.nnzs_jac_inds),
            view(x, edge_indexes.column_indices)
        )
    end

    Threads.@threads for i = 1:length(block.sub_blocks)
        sub_block = block.sub_blocks[i]
        sub_block_data = block_data.sub_block_dict[sub_block.index]
        eval_constraint_jacobian(sub_block, sub_block_data, jac_values, x)
    end

    return
end

### Eval_H_CB

function MOI.hessian_lagrangian_structure(
    evaluator::BlockNLPEvaluator
)::Vector{Tuple{Int64,Int64}}

    block = evaluator.graph.block
    block_data = evaluator.block_data
    I = Vector{Int64}(undef, block_data.nnz_hess) # row indices
    J = Vector{Int64}(undef, block_data.nnz_hess) # column indices

    hessian_lagrangian_structure(evaluator.graph.block, evaluator.block_data, I, J)
    hessian_sparsity = collect(zip(I, J))

    return hessian_sparsity
end

function hessian_lagrangian_structure(
    block::GOI.Block,
    block_data::BlockData,
    I::AbstractArray,
    J::AbstractArray
)

    Threads.@threads for i = 1:length(block.edges)
        edge_index = block.edges[i].index
        edge_model = block_data.edge_model_dict[edge_index]
        edge_indexes = block_data.edge_index_dict[edge_index]
        isempty(edge_indexes.nnzs_hess_inds) && continue

        localI = view(I, edge_indexes.nnzs_hess_inds)
        localJ = view(J, edge_indexes.nnzs_hess_inds)
        edge_hessian_sparsity = MOI.hessian_lagrangian_structure(edge_model)

        # update view with edge jacobian structure
        for (k,(row,col)) in enumerate(edge_hessian_sparsity)
            # map coordinates to global indices
            localI[k] = edge_indexes.column_indices[row]
            localJ[k] = edge_indexes.column_indices[col]
        end
    end

    Threads.@threads for i = 1:length(block.sub_blocks)
        sub_block = block.sub_blocks[i]
        sub_block_data = block_data.sub_block_dict[sub_block.index]
        hessian_lagrangian_structure(sub_block, sub_block_data, I, J)
    end

    return
end

function MOI.eval_hessian_lagrangian(
    evaluator::BlockNLPEvaluator, 
    hess_values::AbstractArray,
    x::AbstractArray,
    sigma::Float64,
    mu::AbstractArray
)
    eval_hessian_lagrangian(
        evaluator.graph.block,
        evaluator.block_data,
        hess_values,
        x,
        sigma,
        mu
    )
end

function eval_hessian_lagrangian(
    block::GOI.Block,
    block_data::BlockData,
    hess_values::AbstractArray,
    x::AbstractArray,
    sigma::Float64,
    mu::AbstractArray
)

    Threads.@threads for i = 1:length(block.edges)
        edge_index = block.edges[i].index
        edge_model = block_data.edge_model_dict[edge_index]
        edge_indexes = block_data.edge_index_dict[edge_index]
        MOI.eval_hessian_lagrangian(
            edge_model,
            view(hess_values, edge_indexes.nnzs_hess_inds),
            view(x, edge_indexes.column_indices),
            sigma,
            view(mu, edge_indexes.row_indices)
        )
    end

    Threads.@threads for i = 1:length(block.sub_blocks)
        sub_block = block.sub_blocks[i]
        sub_block_data = block_data.sub_block_dict[sub_block.index]
        eval_hessian_lagrangian(sub_block, sub_block_data, hess_values, x, sigma, mu)
    end

    return
end