# TODO: Associate QPData + NLPData with each edge

struct EdgeData
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
    node_data::OrderedDict{Node,UnitRange{Int64}}
    edge_data::OrderedDict{Edge,EdgeData}
    sub_block_data::OrderedDict{BlockIndex,BlockData}
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
            OrderedDict{Node,UnitRange{Int64}}(),
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
        all_columns::UnitRange{Int64},
        all_rows::UnitRange{Int64},
        node_data::OrderedDict{Node,UnitRange{Int64}},
        edge_data::OrderedDict{Edge,EdgeData},
        sub_block_data::OrderedDict{BlockIndex,BlockData}
    )
        return new(
            num_variables,
            num_constraints,
            nnz_jac,
            nnz_hess,
            all_columns,
            all_rows,
            node_data,
            edge_data,
            sub_block_data
        )
    end
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
    # count up variables
    count_columns = 0
    offset_start = offset_columns

    node_columns = OrderedDict{Node,UnitRange}()
    for node in block.nodes
        num_vars = _num_variables(node)
        node_columns[node] = (count_columns+1:count_columns+num_vars).+offset_columns
        count_columns += num_vars
    end
    block_data.num_variables = count_columns
    block_data.node_data = node_columns
    block_data.local_columns = offset_columns+1:offset_columns+count_columns
    
    # build sub-blocks
    offset_columns = offset_columns+count_columns
    for sub_block in block.sub_blocks
        sub_block_data = block_data.sub_block_data[sub_block.index]
        _build_node_data!(sub_block, sub_block_data; offset_columns=offset_columns)

        # update offset
        offset_columns = offset_columns+sub_block_data.num_variables

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
    edge_data = OrderedDict{Edge,EdgeData}()
    offset_start = offset_rows
    
    # each edge contains constraints, jacobian, and hessian data
    for edge in block.edges
        nlp_data = MOI.get(edge, MOI.NLPBlock())
        MOI.initialize(nlp_data.evaluator, requested_features)

        ### map edge variable indices to evaluator columns
        
        if edge isa Edge{NTuple{1,Node}}
            # if self-edge, just pull unit ranges
            node = edge.elements[1]
            columns = block_data.node_data[node]
        else
            # do some searching to get the connected variable indices
            node_var_indices = node_variable_indices(block, edge)#::Vector{NodeVariableIndex}
            columns = Int64[]
            for nvi in node_var_indices
                node = nvi.node
                node_block = block.block_by_index[node.block_index]
                if node_block == block
                    # if the node is in `block`, just grab the data
                    node_columns = block_data.node_data[node]
                else
                    # get the sub-block and then get the node data
                    node_block_data = block_data.sub_block_data[node_block.index]
                    node_columns = node_block_data.node_data[node]
                end
                # grab element from unit range
                push!(columns, node_columns[nvi.index.value])
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

        edge_data[edge] = EdgeData(columns, rows, nnzs_jac_inds, nnzs_hess_inds)
    end
    
    block_data.local_rows = offset_rows+1:offset_rows+count_rows
    block_data.edge_data = edge_data
    block_data.num_constraints = count_rows
    block_data.nnz_jac = count_nnzj
    block_data.nnz_hess = count_nnzh

    # update offsets for sub-blocks
    offset_rows = offset_rows+count_rows
    offset_nnzj = offset_nnzj+count_nnzj
    offset_nnzh = offset_nnzh+count_nnzh

    for sub_block in block.sub_blocks
        sub_block_data = block_data.sub_block_data[sub_block.index]
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
    #block::Block,
    graph::GOI.Graph, 
    requested_features::Vector{Symbol}
)
    
    block_data = BlockData()

    _add_sublock_data!(graph, block_data)

    _build_node_data!(graph, block_data)

    _build_edge_data!(graph, block_data, requested_features)

    block_data.all_columns = 1:block_data.num_variables
    block_data.all_rows = 1:block_data.num_constraints

    return block_data
end

"""
    GraphNLPEvaluator(
        block::Block
    )
Create `Evaluator`, a subtype of `MOI.AbstractNLPEvaluator`, from `Model`.
"""
mutable struct GraphNLPEvaluator <: MOI.AbstractNLPEvaluator
    # The block containing nodes and edges
    block::Block
    block_data::BlockData
    
    eval_objective_timer::Float64
    eval_constraint_timer::Float64
    eval_objective_gradient_timer::Float64
    eval_constraint_jacobian_timer::Float64
    eval_hessian_lagrangian_timer::Float64

    function GraphNLPEvaluator(block::Block)
        block_data = BlockData()
        return new(
            block,
            block_data,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )
    end
end

function Base.string(evaluator::GraphNLPEvaluator)
    return """Block NLP Evaluator
    """
end
Base.print(io::IO, evaluator::GraphNLPEvaluator) = print(io, string(evaluator))
Base.show(io::IO, evaluator::GraphNLPEvaluator) = print(io, evaluator)


function MOI.initialize(evaluator::GraphNLPEvaluator, requested_features::Vector{Symbol})
    # create local evaluators. QPBlockData and NLPBlockData

    evaluator.block_data = build_block_data(evaluator.block, requested_features)
    return
end

### Eval_F_CB

function MOI.eval_objective(evaluator::GraphNLPEvaluator, x::AbstractArray)
    return eval_objective(evaluator.block, evaluator.block_data, x)
end

function eval_objective(block::Block, block_data::BlockData, x::AbstractArray)

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

function MOI.eval_objective_gradient(
    evaluator::GraphNLPEvaluator,
    gradient::AbstractArray,
    x::AbstractArray
)
    gradient[:] .= 0.0
    eval_objective_gradient(evaluator.block, evaluator.block_data, gradient, x)
end

function eval_objective_gradient(
    block::Block,
    block_data::BlockData,
    gradient::AbstractArray,
    x::AbstractArray
)
    # IDEA: fill each edge gradient as a sparse vector, then sum together
    edge_gradients = [spzeros(length(gradient)) for _ = 1:length(block.edges)]
    Threads.@threads for i = 1:length(block.edges)
        edge = block.edges[i]
        edge_data = block_data.edge_data[edge]
        columns = edge_data.column_indices
        MOI.eval_objective_gradient(edge, view(edge_gradients[i],columns), view(x,columns))
    end
    gradient[:] .+= sum(edge_gradients)

    # evaluate sub blocks
    Threads.@threads for i = 1:length(block.sub_blocks)
        sub_block = block.sub_blocks[i]
        sub_block_data = block_data.sub_block_data[sub_block.index]
        eval_objective_gradient(sub_block, sub_block_data, gradient, x)
    end

    return
end

### Eval_G_CB

function MOI.eval_constraint(
    evaluator::GraphNLPEvaluator,
    c::AbstractArray,
    x::AbstractArray
)
    eval_constraint(evaluator.block, evaluator.block_data, c, x)
end

function eval_constraint(
    block::Block,
    block_data::BlockData,
    c::AbstractArray,
    x::AbstractArray
)
    Threads.@threads for i = 1:length(block.edges)
        edge = block.edges[i]
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

function MOI.jacobian_structure(evaluator::GraphNLPEvaluator)::Vector{Tuple{Int64,Int64}}

    I = Vector{Int64}(undef, evaluator.block_data.nnz_jac) # row indices
    J = Vector{Int64}(undef, evaluator.block_data.nnz_jac) # column indices

    jacobian_structure(evaluator.block, evaluator.block_data, I, J)
    jacobian_sparsity = collect(zip(I, J))

    return jacobian_sparsity
end

function jacobian_structure(
    block::Block,
    block_data::BlockData,
    I::AbstractArray,
    J::AbstractArray
)

    Threads.@threads for i = 1:length(block.edges)
        edge = block.edges[i]
        edge_data = block_data.edge_data[edge]
        isempty(edge_data.nnzs_jac_inds) && continue

        localI = view(I, edge_data.nnzs_jac_inds)
        localJ = view(J, edge_data.nnzs_jac_inds)
        edge_jacobian_sparsity = MOI.jacobian_structure(edge)

        # update view with edge jacobian structure
        for (k,(row,col)) in enumerate(edge_jacobian_sparsity)
            # map coordinates to global indices
            # rows are ordered, so offset is first true row index
            localI[k] = row + edge_data.row_indices[1] - 1

            # local columns should ALWAYS be ordered [1...N], so just grab the index
            localJ[k] = edge_data.column_indices[col]
        end
    end

    Threads.@threads for i = 1:length(block.sub_blocks)
        sub_block = block.sub_blocks[i]
        sub_block_data = block_data.sub_block_data[sub_block.index]
        jacobian_structure(sub_block, sub_block_data, I, J)
    end

    return
end

function MOI.eval_constraint_jacobian(
    evaluator::GraphNLPEvaluator,
    jac_values::AbstractArray,
    x::AbstractArray
)
    eval_constraint_jacobian(evaluator.block, evaluator.block_data, jac_values, x)
end

function eval_constraint_jacobian(
    block::Block,
    block_data::BlockData,
    jac_values::AbstractArray,
    x::AbstractArray
)

    Threads.@threads for i = 1:length(block.edges)
        edge = block.edges[i]
        edge_data = block_data.edge_data[edge]
        MOI.eval_constraint_jacobian(
            edge,
            view(jac_values, edge_data.nnzs_jac_inds),
            view(x, edge_data.column_indices)
        )
    end

    Threads.@threads for i = 1:length(block.sub_blocks)
        sub_block = block.sub_blocks[i]
        sub_block_data = block_data.sub_block_data[sub_block.index]
        eval_constraint_jacobian(sub_block, sub_block_data, jac_values, x)
    end

    return
end

### Eval_H_CB

function MOI.hessian_lagrangian_structure(
    evaluator::GraphNLPEvaluator
)::Vector{Tuple{Int64,Int64}}

    block = evaluator.block
    block_data = evaluator.block_data
    I = Vector{Int64}(undef, block_data.nnz_hess) # row indices
    J = Vector{Int64}(undef, block_data.nnz_hess) # column indices

    hessian_lagrangian_structure(evaluator.block, evaluator.block_data, I, J)
    hessian_sparsity = collect(zip(I, J))

    return hessian_sparsity
end

function hessian_lagrangian_structure(
    block::Block,
    block_data::BlockData,
    I::AbstractArray,
    J::AbstractArray
)

    Threads.@threads for i = 1:length(block.edges)
        edge = block.edges[i]
        edge_data = block_data.edge_data[edge]
        isempty(edge_data.nnzs_hess_inds) && continue

        localI = view(I, edge_data.nnzs_hess_inds)
        localJ = view(J, edge_data.nnzs_hess_inds)
        edge_hessian_sparsity = MOI.hessian_lagrangian_structure(edge)

        # update view with edge jacobian structure
        for (k,(row,col)) in enumerate(edge_hessian_sparsity)
            # map coordinates to global indices
            localI[k] = edge_data.column_indices[row]
            localJ[k] = edge_data.column_indices[col]
        end
    end

    Threads.@threads for i = 1:length(block.sub_blocks)
        sub_block = block.sub_blocks[i]
        sub_block_data = block_data.sub_block_data[sub_block.index]
        hessian_lagrangian_structure(sub_block, sub_block_data, I, J)
    end

    return
end

function MOI.eval_hessian_lagrangian(
    evaluator::GraphNLPEvaluator, 
    hess_values::AbstractArray,
    x::AbstractArray,
    sigma::Float64,
    mu::AbstractArray
)
    eval_hessian_lagrangian(
        evaluator.block,
        evaluator.block_data,
        hess_values,
        x,
        sigma,
        mu
    )
end

function eval_hessian_lagrangian(
    block::Block,
    block_data::BlockData,
    hess_values::AbstractArray,
    x::AbstractArray,
    sigma::Float64,
    mu::AbstractArray
)

    Threads.@threads for i = 1:length(block.edges)
        edge = block.edges[i]
        edge_data = block_data.edge_data[edge]
        MOI.eval_hessian_lagrangian(
            edge,
            view(hess_values, edge_data.nnzs_hess_inds),
            view(x, edge_data.column_indices),
            sigma,
            view(mu, edge_data.row_indices)
        )
    end

    Threads.@threads for i = 1:length(block.sub_blocks)
        sub_block = block.sub_blocks[i]
        sub_block_data = block_data.sub_block_data[sub_block.index]
        eval_hessian_lagrangian(sub_block, sub_block_data, hess_values, x, sigma, mu)
    end

    return
end