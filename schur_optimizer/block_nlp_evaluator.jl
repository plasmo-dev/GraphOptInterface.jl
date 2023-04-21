# Track indices for each edge
struct EdgeIndexes
	column_indices::Union{UnitRange{Int64},Vector{Int64}}
    row_indices::Union{UnitRange{Int64},Vector{Int64}}
    nnzs_jac_inds::UnitRange{Int}
    nnzs_hess_inds::UnitRange{Int}
end

# map each edge in graph to a model we can evaluate
function build_edge_model(edge::GOI.Edge; _differentiation_backend=MOI.Nonlinear.SparseReverseMode())
	edge_model = EdgeModel()
	vars = MOI.ListOfVariableIndices(edge)
	
	# MOIU requires an index map to pass constraints. we assume a 1 to 1 mapping.
	var_map = MOIU.IndexMap()
	for var in vars
		var_map[var] = var
	end
	MOIU.pass_nonvariable_constraints(edge_model.qp_data, edge.moi_model, var_map)
	
	# create nlp-block if needed
	if edge.nonlinear_model != nothing
		edge_evaluator = MOI.Nonlinear.Evaluator(edge.nonlinear_model, _differentiation_backend, vars)
		edge_model.nlp_data = MOI.NLPBlockData(edge_evaluator)
	end

	return edge_model
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
    return new(
        graph,
        block_data,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )
end

function _add_sublock_data!(block::Block, block_data::BlockData)
    for sub_block in block.sub_blocks
        sub_block_data = BlockData()
        block_data.sub_block_data[sub_block.index] = sub_block_data
        _add_sublock_data!(sub_block, sub_block_data)
    end
    return
end

function _build_node_data!(
    block::Block,
    block_data::BlockData;
    offset_columns=0
)
    count_columns = 0
    offset_start = offset_columns
    node_columns = OrderedDict{GOI.Node,UnitRange}()

    # get column ranges for each node
    for node in block.nodes
        num_vars = _num_variables(node)
        node_columns[node.index] = (count_columns+1:count_columns+num_vars).+offset_columns
        count_columns += num_vars
    end
    block_data.num_variables = count_columns
    block_data.node_data_dict = node_columns
    block_data.local_columns = offset_columns+1:offset_columns+count_columns
    
    # build sub-blocks. each block stores sub-block data too
    offset_columns = offset_columns+count_columns
    for sub_block in block.sub_blocks
        sub_block_data = block_data.sub_block_data[sub_block.index]
        _build_node_data!(sub_block, sub_block_data; offset_columns=offset_columns)
        merge!(block_data.node_columns, sub_block_data.node_columns) 

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
    block::Block,
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
    edge_model_dict = OrderedDict{GOI.HyperEdge,EdgeModel}
    offset_start = offset_rows
    
    # each edge contains constraints, jacobian, and hessian data
    for edge in block.edges
    	edge_model = build_edge_model(edge)
        nlp_data = edge_model.nlp_data
        MOI.initialize(nlp_data.evaluator, requested_features)

        ### map edge variable indices to evaluator columns
        
        # TODO: update for new data structure
        if num_vertices(edge) == 1
            # if self-edge, just pull unit ranges
            node_index = edge.index.vertices[1]
            columns = block_data.node_data_dict[node_index]
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
            jacobian_sparsity = MOI.jacobian_structure(edge)
            nnzs_jac_inds = (count_nnzj+1:count_nnzj+length(jacobian_sparsity)).+offset_nnzj
            count_nnzj += length(jacobian_sparsity)
        end

        ### edge hessian indices
        if :Hess in requested_features
            hessian_sparsity = MOI.hessian_lagrangian_structure(edge)
            nnzs_hess_inds = (count_nnzh+1:count_nnzh+length(hessian_sparsity)).+offset_nnzh
            count_nnzh += length(hessian_sparsity)
        end

        ### store indexes
        edge_index_dict[edge.index] = EdgeIndexes(columns, row, nnzs_jac_inds, nnzs_hess_inds)
    end
    
    block_data.local_rows = offset_rows+1:offset_rows+count_rows
    block_data.edge_data_dict = edge_data_dict
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

    _add_sublock_data!(graph, block_data)

    _build_node_data!(graph, block_data)

    _build_edge_data!(graph, block_data, requested_features)

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
    return eval_objective(evaluator.block_data, x)
end

function eval_objective(block_data::BlockData, x::AbstractArray)

    obj = Threads.Atomic{Float64}(0.)

    # evaluate root edges
    Threads.@threads for i = 1:length(block_data.edges)
        edge = block_data.edges[i]
        edge_data = block_data.edge_data_dict[edge]
        columns = edge_data.column_indices
        Threads.atomic_add!(obj, MOI.eval_objective(edge_model, view(x, columns)))
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
