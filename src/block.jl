# abstract type BlockModelLike <: MOI.ModelLike end
abstract type NodeModelLike <: MOI.ModelLike end
abstract type EdgeModelLike <: MOI.ModelLike end

### BlockOpt Interface Functions

function node_model(::AbstractBlockOptimizer)::NodeModelLike end

function edge_model(::AbstractBlockOptimizer)::EdgeModelLike end

### BlockIndex

struct BlockIndex
	value::Int64
end

### Node and Edge Definitions

"""
	Node

A set of variables. A node contains a `model` that implements the `NodeModelLike` interface.
"""
struct Node
	index::Int64
	block_index::BlockIndex
	model::NodeModelLike
end

struct NodeVariableIndex
	node::Node
	index::MOI.VariableIndex
end
function NodeVariableIndex(node::Node, index::Int64)
	return NodeVariableIndex(node, MOI.VariableIndex(index))
end

function Base.getindex(node::Node, index::Int64)
	@assert MOI.is_valid(node.model, MOI.VariableIndex(index))
	return MOI.VariableIndex(index)
end

function _column(nv::NodeVariableIndex)
	return nv.index.value
end

"""
	Edge{T<:Tuple}

An edge represents different types of coupling. For instance an Edge{Tuple{Node}} is an edge
the couple variables within a single node. An Edge{Tuple{N,Node}} couple variables across
one or more nodes.
"""
struct Edge{T<:Tuple}
	index::Int64
	block_index::BlockIndex
	elements::T # could be a node, nodes, blocks, or both
	model::EdgeModelLike
end

struct EdgeVariableIndex
	edge::Edge
	index::MOI.VariableIndex
end

struct NodeEdgeMap
	node_to_edge::OrderedDict{NodeVariableIndex,EdgeVariableIndex}
	edge_to_node::OrderedDict{EdgeVariableIndex,NodeVariableIndex}
end
function NodeEdgeMap()
	node_to_edge = OrderedDict{NodeVariableIndex,EdgeVariableIndex}()
	edge_to_node = OrderedDict{EdgeVariableIndex,NodeVariableIndex}()
	return NodeEdgeMap(node_to_edge, edge_to_node)
end

function Base.getindex(node_edge_map::NodeEdgeMap, index::NodeVariableIndex)
	return node_edge_map.node_to_edge[index]
end
function Base.getindex(node_edge_map::NodeEdgeMap, index::EdgeVariableIndex)
	return node_edge_map.edge_to_node[index]
end
function Base.setindex!(node_edge_map::NodeEdgeMap, nindex::NodeVariableIndex, eindex::EdgeVariableIndex)
	return node_edge_map.node_to_edge[nindex] = eindex
end
function Base.setindex!(node_edge_map::NodeEdgeMap, eindex::EdgeVariableIndex, nindex::NodeVariableIndex)
	return node_edge_map.edge_to_node[eindex] = nindex
end

### Block

mutable struct Block
	index::BlockIndex
	parent_index::Union{Nothing,BlockIndex}
	nodes::Vector{Node}
	edges::Vector{Edge}
	sub_blocks::Vector{Block}
	block_by_index::OrderedDict{BlockIndex,Block}
	edge_variable_map::Dict{Edge,NodeEdgeMap}
	function Block(index::Int64)
		block = new()
		block.index = BlockIndex(index)
		block.parent_index = nothing
		block.nodes = Vector{Node}()
		block.edges = Vector{Edge}()
		block.sub_blocks = Vector{Block}()
		block.block_by_index = OrderedDict()
		block.block_by_index[block.index] = block 
		block.edge_variable_map = Dict{Edge,NodeEdgeMap}()
		return block
	end
end

### add nodes and edges to a block

function add_node!(optimizer::AbstractBlockOptimizer, block::Block)::Node
	node_idx = length(block.nodes) + 1
	block_idx = block.index
	model = node_model(optimizer)
	node = Node(node_idx, block_idx, model)
	push!(block.nodes, node)
	return node
end

function add_node!(optimizer::AbstractBlockOptimizer)
	block = MOI.get(optimizer, BlockStructure())
	return add_node!(optimizer, block)
end

function add_edge!(optimizer::AbstractBlockOptimizer, node::Node)::Edge
	root_block = MOI.get(optimizer, BlockStructure())
	block_index = node.block_index
	block = root_block.block_by_index[block_index]
	edge_index = length(block.edges) + 1
	model = edge_model(optimizer)
	edge = Edge{NTuple{1,Node}}(edge_index, block_index, (node,), edge_model(optimizer))
	_add_edge!(block, edge)
	return edge
end

function add_edge!(
	optimizer::AbstractBlockOptimizer,
	block_index::BlockIndex,
	nodes::NTuple{N, Node} where N
)::Edge
	root_block = MOI.get(optimizer, BlockStructure())
	block = root_block.block_by_index[block_index]

	# check arguments make sense
	if !isempty(setdiff(nodes, block.nodes))
		error("all nodes must be within the same block")
	end

	# create and add edge
	edge_index = length(block.edges) + 1
	model = edge_model(optimizer)
	edge = Edge{NTuple{length(nodes),Node}}(edge_index, block_index, nodes, model)
	_add_edge!(block, edge)
	return edge
end

function add_edge!(
	optimizer::AbstractBlockOptimizer,
	block_index::BlockIndex,
	sub_blocks::NTuple{N, Block} where N
)::Edge
	root_block = MOI.get(optimizer, BlockStructure())
	block = root_block.block_by_index[block_index]

	# check arguments make sense
	if !isempty(setdiff(sub_blocks, block.sub_blocks))
		error("all sub blocks must be within the same block")
	end
	
	# create and add edge
	edge_idx = length(block.edges) + 1
	model = edge_model(optimizer)
	edge = Edge{NTuple{length(sub_blocks),Block}}(edge_index, block_index, nodes, model)
	_add_edge!(block, edge)
	return edge
end

function add_edge!(
	optimizer::AbstractBlockOptimizer,
	block_index::BlockIndex,
	node::Node,
	sub_block::Block
)::Edge
	root_block = MOI.get(optimizer, BlockStructure())
	block = root_block.block_by_index[block_index]

	# check arguments make sense
	node in block.nodes || error("Node must be within given block")
	sub_block in block.sub_blocks || error("Sub block must be within give block")

	# create and add edge
	edge_idx = length(block.edges) + 1
	model = edge_model(optimizer)
	edge = Edge{Tuple{Node, Block}}(edge_index, block_index, (node, sub_block), model)
	_add_edge!(block, edge)
	return edge
end

function _add_edge!(block::Block, edge::Edge)
	push!(block.edges,edge)
	block.edge_variable_map[edge] = NodeEdgeMap()
	return
end

function _add_edge!(block::Block, edge::Edge{NTuple{1,Node}})
	push!(block.edges,edge)
	return
end

function add_sub_block!(optimizer::AbstractBlockOptimizer, block::Block)::Block
	main_block = MOI.get(optimizer, Block())
	block_index = BlockIndex(length(main_block.block_by_index) + 1)
	sub_block = Block(block_index)
	push!(block.sub_blocks, sub_block)
	block.block_by_index[block_index] = sub_block
	main_block.block_by_index[block_index] = sub_block
	return sub_block
end

function add_sub_block!(optimizer::AbstractBlockOptimizer)::Block
	main_block = MOI.get(optimizer, Block())
	block = add_sub_block!(optimizer, main_block)
	return block
end

### variable index management

function add_edge_variable!(block::Block, edge::Edge, node::Node, vi::MOI.VariableIndex)
	@assert node in get_nodes(edge)
	nvi = NodeVariableIndex(node, vi)
	if !haskey(block.edge_variable_map[edge].node_to_edge, nvi)
		num_edge_vars = length(block.edge_variable_map[edge].node_to_edge)
		variable_index = MOI.VariableIndex(num_edge_vars + 1)
		edge_variable_index = EdgeVariableIndex(edge, variable_index)
		block.edge_variable_map[edge][nvi] = edge_variable_index
		block.edge_variable_map[edge][edge_variable_index] = nvi
		return variable_index
	else
		return block.edge_variable_map[edge][nvi].index
	end
end

function variable_indices(
	block::Block,
	edge::Edge{NTuple{1,Node}}
)::Vector{MOI.VariableIndex}
	node = edge.elements[1]
	return MOI.get(node, MOI.ListOfVariableIndices())
end

function variable_indices(
	block::Block,
	edge::Edge,
)::Vector{MOI.VariableIndex}
	node_var_indices = collect(values(block.edge_variable_map[edge].node_to_edge))
	return [nvi.index for nvi in node_var_indices]
end

function node_variable_indices(block::Block, edge::Edge)::Vector{NodeVariableIndex}
	nvis = collect(values(block.edge_variable_map[edge].edge_to_node))
end

### query functions

function get_nodes(block::Block)
	return block.nodes
end

function get_nodes(node::Node)
	return [node]
end

function get_edges(block::Block)
	return block.edges
end

function get_nodes(edge::Edge)
	nodes = []
	for element in edge.elements
		append!(nodes, get_nodes(element))
	end
	return nodes
end

function _num_variables(node::Node)
    return MOI.get(node, MOI.NumberOfVariables())
end

function _num_constraints(edge::Edge)
    n_con = 0
    for (F,S) in MOI.get(edge, MOI.ListOfConstraintTypesPresent())
        n_con += MOI.get(edge, MOI.NumberOfConstraints{F,S}())
    end
    nlp_block = MOI.get(edge, MOI.NLPBlock())
    n_con += length(nlp_block.constraint_bounds)
    return n_con
end

function _num_variables(block::Block)
    return MOI.get(block, MOI.NumberOfVariables())
end

function _num_constraints(block::Block)
	return sum(num_constraints(edge) for edge in all_edges(block))
end

function MOI.add_variable(optimizer::AbstractBlockOptimizer, node::Node)
	local_vi = MOI.add_variable(node.model)
	return local_vi
end

function MOI.add_variables(optimizer::AbstractBlockOptimizer, node::Node, n::Int64)
	vars = MOI.VariableIndex[]
	for _ = 1:n
		nvi = MOI.add_variable(optimizer, node)
		push!(vars, nvi)
	end
	return vars
end

function MOI.add_constraint(
	optimizer::AbstractBlockOptimizer,
	node::Node,
	vi::MOI.VariableIndex,
	set::S
) where {S <: MOI.AbstractSet}
	return MOI.add_constraint(node.model, vi, set)
end

# forward methods so Node and Edge call their underlying model
@forward Node.model (MOI.get, MOI.set)
@forward Edge.model (MOI.get, MOI.set, MOI.eval_objective, MOI.eval_objective_gradient,
	MOI.eval_constraint, MOI.jacobian_structure, MOI.eval_constraint_jacobian,
	MOI.hessian_lagrangian_structure, MOI.eval_hessian_lagrangian
)

function MOI.add_constraint(
	optimizer::AbstractBlockOptimizer, 
	edge::Edge,
	func::F,
	set::S
) where {F <: MOI.AbstractFunction, S <: MOI.AbstractSet}
	
	local_ci = MOI.add_constraint(edge.model, func, set)
	block_index = MOI.get(optimizer.block, MOI.NumberOfConstraints{F,S}())
	block_ci = MOI.ConstraintIndex{F,S}(block_index)
	#edge.constraint_map[local_ci] = block_ci
	return block_ci
end

### Block attributes

function MOI.get(block::Block, attr::MOI.ListOfConstraintTypesPresent)
	ret = []
	for node in get_nodes(block)
		append!(ret, MOI.get(node.model, attr))
	end
	for edge in get_edges(block)
		append!(ret, MOI.get(edge.model, attr))
	end
    return unique(ret)
end

function MOI.get(block::Block, attr::MOI.NumberOfVariables)
    return sum(MOI.get(node, attr) for node in get_nodes(block))
end

function MOI.get(block::Block, attr::MOI.ListOfVariableIndices)
	var_list = []
	for node in get_nodes(block)
		append!(var_list, MOI.get(node, attr))
	end
    return var_list
end

function MOI.get(
    block::Block,
    attr::MOI.NumberOfConstraints{F,S}
) where {F <: MOI.AbstractFunction, S <: MOI.AbstractSet}

    return sum(MOI.get(edge, attr) for edge in get_edges(block))
end

function Base.string(block::Block)
    return """Block
    $(length(block.nodes)) nodes
    $(length(block.edges)) edges
    $(length(block.sub_blocks)) sub-blocks
    """
end
Base.print(io::IO, block::Block) = print(io, string(block))
Base.show(io::IO, block::Block) = print(io, block)