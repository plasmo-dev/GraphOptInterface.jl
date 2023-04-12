abstract type AbstractBlock end
abstract type NodeModelLike <: MOI.ModelLike end
abstract type EdgeModelLike <: MOI.ModelLike end

### BlockOpt Interface Functions

function node_model(::AbstractBlockOptimizer)::NodeModelLike end

function edge_model(::AbstractBlockOptimizer)::EdgeModelLike end

### BlockIndex

struct BlockIndex
	value::Int64
end

struct NodeIndex
	value::Int64
end

struct EdgeIndex
	value::Int64
end

function index_value(index::Union{BlockIndex,NodeIndex,EdgeIndex})
	return index.value
end

### Node and Edge Definitions

"""
	Node

A set of variables. A node contains a `model` that implements the `NodeModelLike` interface.
"""
struct Node
	index::NodeIndex
	block_index::BlockIndex
	model::NodeModelLike
end

function node_index(node::Node)
	return node.index
end

function Base.getindex(node::Node, index::Int64)
	@assert MOI.is_valid(node.model, MOI.VariableIndex(index))
	return MOI.VariableIndex(index)
end

"""
	Edge{N}

An edge represents different types of coupling. For instance an Edge{Tuple{Node}} is an edge
the couple variables within a single node. An Edge{Tuple{N,Node}} couple variables across
one or more nodes.
"""
struct Edge{N}
	index::EdgeIndex
	block_index::BlockIndex
	elements::NTuple{N,NodeIndex} # could be any number of nodes
	model::EdgeModelLike

	# local edge variables
	# coupled_variables::Union{Nothing,OrderedDict{MOI.VariableIndex,Tuple{NodeIndex,MOI.VariableIndex}}}
	edge_variable_map::OrderedDict{MOI.VariableIndex,Tuple{NodeIndex,MOI.VariableIndex}}
	node_variable_map::OrderedDict{Tuple{NodeIndex,MOI.VariableIndex},MOI.VariableIndex}
end

### Blocks
mutable struct Block{T<:AbstractBlock} <: AbstractBlock
	index::BlockIndex
	parent_index::BlockIndex
	nodes::Vector{Node}
	edges::Vector{Edge}
	sub_blocks::Vector{T}
	node_by_index::OrderedDict{NodeIndex,Node}
	edge_by_index::OrderedDict{EdgeIndex,Edge}
	block_by_index::OrderedDict{BlockIndex,Block}
	function Block(parent_index::Int64, block_index::Int64)
		block = new{Block}()
		block.index = BlockIndex(block_index)
		block.parent_index = BlockIndex(parent_index)
		block.nodes = Vector{Node}()
		block.edges = Vector{Edge}()
		block.sub_blocks = Vector{Block}()
		block.node_by_index = OrderedDict{NodeIndex,Node}()
		block.edge_by_index = OrderedDict{EdgeIndex,Edge}()
		block.block_by_index = OrderedDict{BlockIndex,Block}()
		block.block_by_index[block.index] = block
		return block
	end
end

mutable struct RootBlock <: AbstractBlock
	index::BlockIndex
	nodes::Vector{Node}
	edges::Vector{Edge}
	sub_blocks::Vector{Block}
	node_by_index::OrderedDict{NodeIndex,Node}
	edge_by_index::OrderedDict{EdgeIndex,Edge}
	block_by_index::OrderedDict{BlockIndex,Union{RootBlock,Block}}
	graph::HyperGraph # NOTE: vertices match node indices
	function RootBlock()
		block = new()
		block.index = BlockIndex(0)
		block.nodes = Vector{Node}()
		block.edges = Vector{Edge}()
		block.sub_blocks = Vector{Block}()
		block.node_by_index = OrderedDict{NodeIndex,Node}()
		block.edge_by_index = OrderedDict{EdgeIndex,Edge}()
		block.block_by_index = OrderedDict{BlockIndex,Union{RootBlock,Block}}()
		block.block_by_index[block.index] = block
		block.graph = HyperGraph()
		return block
	end
end

### Add nodes and edges

"""
	add_node!(optimizer::AbstractBlockOptimizer, block::T where T<:AbstractBlock)::Node

Add a node to `block` on `optimizer`. The index of the node is determined by the root block.

	add_node!(optimizer::AbstractBlockOptimizer)

Add a node to `optimizer` root block.
"""
function add_node!(optimizer::AbstractBlockOptimizer, block::T where T<:AbstractBlock)::Node
	# get the root block to determine the new index
	root = MOI.get(optimizer, BlockStructure())
	num_vertices = Graphs.nv(root.graph)
	node_idx = NodeIndex(num_vertices + 1)

	# create model and node
	model = node_model(optimizer)
	node = Node(node_idx, block.index, model)
	push!(block.nodes, node)
	root.node_by_index[node_idx] = node

	# add vertex to the hypergraph
	Graphs.add_vertex!(root.graph)

	nverts = Graphs.nv(root.graph)
	node_id = node_idx.value

	@assert (Graphs.nv(root.graph) === node_idx.value)
	return node
end

# add node to the root
function add_node!(optimizer::AbstractBlockOptimizer)::Node
	root = MOI.get(optimizer, BlockStructure())
	return add_node!(optimizer, root)
end

function add_edge!(optimizer::AbstractBlockOptimizer, node::Node)::Edge
	root = MOI.get(optimizer, BlockStructure())
	num_edges = Graphs.ne(root.graph)
	
	# get the corresponding block
	block_index = node.block_index

	# create the edge
	edge_index = EdgeIndex(num_edges + 1)
	model = edge_model(optimizer)
	edge = _add_edge!(root, edge_index, block_index, (node.index,), model)
	
	return edge
end

function add_edge!(optimizer::AbstractBlockOptimizer, block::T, nodes::Node...)::Edge where T <: AbstractBlock
	root = MOI.get(optimizer, BlockStructure())
	num_edges = Graphs.ne(root.graph)
	
	# check the edge is allowed on the corresponding block
	# block = root.block_by_index[block_index]
	@assert isempty(setdiff(nodes, get_nodes_one_layer(block)))

	node_indices = node_index.(nodes)
	edge_index = EdgeIndex(num_edges + 1)
	model = edge_model(optimizer)

	edge = _add_edge!(root, edge_index, block.index, node_indices, model)

	return edge
end

function _add_edge!(
	root::RootBlock,
	edge_index::EdgeIndex,
	block_index::BlockIndex, 
	node_indices::NTuple{N,NodeIndex} where N,
	edge_model::EdgeModelLike
)
	edge = Edge{length(node_indices)}(
		edge_index,
		block_index,
		node_indices, 
		edge_model,
		OrderedDict{MOI.VariableIndex,Tuple{NodeIndex,MOI.VariableIndex}}(),
		OrderedDict{Tuple{NodeIndex,MOI.VariableIndex},MOI.VariableIndex}()
	)

	# add edge to block
	block = root.block_by_index[block_index]
	push!(block.edges,edge)

	block.edge_by_index[edge.index] = edge
	root.edge_by_index[edge.index] = edge

	# update the graph
	vertices = index_value.(node_indices)
	Graphs.add_edge!(root.graph, vertices...)
	return edge
end

function add_sub_block!(optimizer::AbstractBlockOptimizer, parent::T)::Block where T <: AbstractBlock
	# get the root block
	root = MOI.get(optimizer, BlockStructure())

	# define a new block index, create the sub-block
	new_block_index = length(root.block_by_index)
	sub_block = Block(parent.index.value, new_block_index)
	push!(parent.sub_blocks, sub_block)

	# track new block index
	parent.block_by_index[sub_block.index] = sub_block
	root.block_by_index[sub_block.index] = sub_block
	return sub_block
end

function add_sub_block!(optimizer::AbstractBlockOptimizer)::Block
	root = MOI.get(optimizer, BlockStructure())
	block = add_sub_block!(optimizer, root)
	return block
end

### Query Functions

function get_nodes(block::T) where T <: AbstractBlock
	nodes = block.nodes
	return block.nodes
end

function get_nodes_one_layer(block::T) where T <: AbstractBlock
	nodes = block.nodes
	for sub_block in block.sub_blocks
		nodes = [nodes; sub_block.nodes]
	end
	return nodes
end


function get_edges(block::T) where T <: AbstractBlock
	return block.edges
end

function all_nodes(block::T) where T <: AbstractBlock
	nodes = block.nodes
	for sub_block in block.sub_blocks
		append!(nodes, all_nodes(sub_block))
	end
	return nodes
end

# edges attached to one node
function self_edges(block::Block)
	return filter((edge) -> edge isa Edge{1}, block.edges)
end

# edges that connect nodes
function linking_edges(block::Block)
	return filter((edge) -> !(edge isa Edge{1}), block.edges)
end

function connected_nodes(optimizer::AbstractBlockOptimizer, edge::Edge)
	root = MOI.get(optimizer, MOI.BlockStructure())
	graph = root.graph
	vertices = edge.elements
	return getindex.(Ref(root.node_by_index), vertices)
end

### Neighbors

# every neighbor including parent and child neighbors
function all_neighbors(optimizer::AbstractBlockOptimizer, nodes::Vector{Node})::Vector{Node}
	root = MOI.get(optimizer, BlockStructure())
	vertices = index_value.(node_index.(nodes))
	neighbor_vertices = Graphs.all_neighbors(root.graph, vertices...)
	return_indices = NodeIndex.(neighbor_vertices)
	return_nodes = getindex.(Ref(root.node_by_index), return_indices)
	return return_nodes
end

function all_neighbors(optimizer::AbstractBlockOptimizer, block::AbstractBlock)::Vector{Node}
	return all_neighbors(optimizer, block.nodes)
end

# neighbors in parent block
function parent_neighbors(optimizer::AbstractBlockOptimizer, block::AbstractBlock)::Vector{Node}
	root = MOI.get(optimizer, BlockStructure())
	parent_block = root.block_by_index[block.parent_index]
	neighbors = all_neighbors(optimizer, block)
	parent_nodes = intersect(parent_block.nodes, neighbors)
	return parent_nodes
end

function neighbors(optimizer::AbstractBlockOptimizer, block::AbstractBlock)::Vector{Node}
	root = MOI.get(optimizer, BlockStructure())
	parent_block = root.block_by_index[block.parent_index]
	neighbors = all_neighbors(optimizer, block)
	neighbor_nodes = setdiff(neighbors, parent_block.nodes)
	return neighbor_nodes
end

### Incident Edges

# all edges incident to a block
function all_incident_edges(optimizer::AbstractBlockOptimizer, block::AbstractBlock)::Vector{Edge}
	root = MOI.get(optimizer, BlockStructure())
	node_indices = index_value.(node_index.((all_nodes(block))))
	hyperedges = incident_edges(root.graph, node_indices)

	# TODO: make cleaner hypergraph implementation
	edge_indices = EdgeIndex.([root.graph.hyperedges[h.vertices] for h in hyperedges])
	edges = getindex.(Ref(root.edge_by_index), edge_indices)
	return edges
end

# edges that connect this block to a parent block
function parent_incident_edges(optimizer::AbstractBlockOptimizer, block::AbstractBlock)::Vector{Edge}
	root = MOI.get(optimizer, BlockStructure())
	parent_block = root.block_by_index[block.parent_index]
	inc_edges = all_incident_edges(optimizer, block)
	parent_edges = intersect(parent_block.edges, inc_edges)
	return parent_edges
end

# edges that connect this block to other blocks
function incident_edges(optimizer::AbstractBlockOptimizer, block::AbstractBlock)::Vector{Edge}
	root = MOI.get(optimizer, BlockStructure())
	parent_block = root.block_by_index[block.parent_index]
	inc_edges = all_incident_edges(optimizer, block)
	external_edges = filter((edge) -> any(node -> node in node_index.(parent_block.nodes), edge.elements), inc_edges)
	return external_edges
end

# edges that connect this block to children blocks
# TODO: we might just label these edges with direction for easy look-up
function children_incident_edges(optimizer::AbstractBlockOptimizer, block::AbstractBlock)::Vector{Edge}
	children_edges = []
	for sub_block in block.sub_blocks
		append!(children_edges, parent_incident_edges(sub_block))
	end
	return children_edges
end

### Coupling Variables

function add_coupling_variable!(edge::Edge, node::Node, vi::MOI.VariableIndex)::MOI.VariableIndex
	@assert node.index in edge.elements
	if length(edge.elements) === 1
		error("self-edges cannot have coupling variables")
	end
	# if edge variable already exists, return it
	if (node.index,vi) in keys(edge.node_variable_map)
		return edge.node_variable_map[(node.index,vi)]
	else
		num_edge_variables = length(edge.edge_variable_map)
		edge_vi = MOI.VariableIndex(num_edge_variables + 1)
		edge.edge_variable_map[edge_vi] = (node.index,vi)
		edge.node_variable_map[(node.index,vi)] = edge_vi
		return edge_vi
	end
end

### MOI Functions

function MOI.get(
	edge::Edge{1},
	attr::MOI.ListOfVariableIndices
)::Vector{MOI.VariableIndex}
	node = edge.elements[1]
	return MOI.get(node, MOI.ListOfVariableIndices())
end

# get list of variable indices on the edge
function MOI.get(
	edge::Edge,
	attr::MOI.ListOfVariableIndices
)::Vector{MOI.VariableIndex}
	return collect(values(edge.edge_variables))
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

function MOI.add_constraint(
	optimizer::AbstractBlockOptimizer, 
	edge::Edge,
	func::F,
	set::S
) where {F <: MOI.AbstractFunction, S <: MOI.AbstractSet}
	local_ci = MOI.add_constraint(edge.model, func, set)
	block_index = MOI.get(optimizer.block, MOI.NumberOfConstraints{F,S}())
	block_ci = MOI.ConstraintIndex{F,S}(block_index)
	return block_ci
end

### Forward methods so Node and Edge objects call their underlying MOI model

# forward node methods
@forward Node.model (MOI.get, MOI.set)

# forward edge methods
@forward Edge.model (MOI.get, MOI.set, MOI.eval_objective, MOI.eval_objective_gradient,
	MOI.eval_constraint, MOI.jacobian_structure, MOI.eval_constraint_jacobian,
	MOI.hessian_lagrangian_structure, MOI.eval_hessian_lagrangian
)

### Block attributes

function MOI.get(block::T, attr::MOI.ListOfConstraintTypesPresent) where T <: AbstractBlock
	ret = []
	for node in get_nodes(block)
		append!(ret, MOI.get(node.model, attr))
	end
	for edge in get_edges(block)
		append!(ret, MOI.get(edge.model, attr))
	end
    return unique(ret)
end

function MOI.get(block::T, attr::MOI.NumberOfVariables) where T <: AbstractBlock
    return sum(MOI.get(node, attr) for node in get_nodes(block))
end

function MOI.get(block::T, attr::MOI.ListOfVariableIndices) where T <: AbstractBlock
	var_list = []
	for node in get_nodes(block)
		append!(var_list, MOI.get(node, attr))
	end
    return var_list
end

function MOI.get(
    block::T,
    attr::MOI.NumberOfConstraints{F,S}
) where {T <: AbstractBlock, F <: MOI.AbstractFunction, S <: MOI.AbstractSet}

    return sum(MOI.get(edge, attr) for edge in get_edges(block))
end

function Base.string(block::AbstractBlock)
    return """Block
    $(length(block.nodes)) nodes
    $(length(block.edges)) edges
    $(length(block.sub_blocks)) sub-blocks
    """
end
Base.print(io::IO, block::AbstractBlock) = Base.print(io, Base.string(block))
Base.show(io::IO, block::AbstractBlock) = Base.print(io, block)

### Utility funcs

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


# function _add_edge!(block::Block, edge::Edge)
# 	push!(block.edges,edge)
# 	block.edge_variable_map[edge] = NodeEdgeMap()
# 	vertices = [node.index for node in edge.elements]
# 	for vertex in vertices
# 		for other_vertex in setdiff(vertices,[vertex])
# 			Graphs.add_edge!(block.graph, vertex, other_vertex)
# 		end
# 	end
# 	return
# end


# coupling edge
# function add_edge!(
# 	optimizer::AbstractBlockOptimizer,
# 	block_index::BlockIndex,
# 	nodes::NTuple{N, Node} where N
# )::Edge

# 	root_block = MOI.get(optimizer, BlockStructure())
# 	block = root_block.block_by_index[block_index]


# 	# check arguments make sense
# 	# if !isempty(setdiff(nodes, block.nodes))
# 	# 	error("all nodes must be within the same block")
# 	# end

# 	# create and add edge
# 	edge_index = length(block.edges) + 1
# 	model = edge_model(optimizer)
# 	edge = Edge{NTuple{length(nodes),Node}}(edge_index, block_index, nodes, model)


# 	_add_edge!(block, edge)
# 	return edge
# end

# UPDATE: let's not do edges between nodes and blocks
# function add_edge!(
# 	optimizer::AbstractBlockOptimizer,
# 	block_index::BlockIndex,
# 	sub_blocks::NTuple{N, Block} where N
# )::Edge
# 	root_block = MOI.get(optimizer, BlockStructure())
# 	block = root_block.block_by_index[block_index]

# 	# check arguments make sense
# 	if !isempty(setdiff(sub_blocks, block.sub_blocks))
# 		error("all sub blocks must be within the same block")
# 	end
	
# 	# create and add edge
# 	edge_index = length(block.edges) + 1
# 	model = edge_model(optimizer)
# 	edge = Edge{NTuple{length(sub_blocks),Block}}(edge_index, block_index, sub_blocks, model)
# 	_add_edge!(block, edge)
# 	return edge
# end

# UPDATE: let's not do edges between nodes and blocks
# function add_edge!(
# 	optimizer::AbstractBlockOptimizer,
# 	block_index::BlockIndex,
# 	node::Node,
# 	sub_block::Block
# )::Edge
# 	root_block = MOI.get(optimizer, BlockStructure())
# 	block = root_block.block_by_index[block_index]

# 	# check arguments make sense
# 	node in block.nodes || error("Node must be within given block")
# 	sub_block in block.sub_blocks || error("Sub block must be within give block")

# 	# create and add edge
# 	edge_index = length(block.edges) + 1
# 	model = edge_model(optimizer)
# 	edge = Edge{Tuple{Node, Block}}(edge_index, block_index, (node, sub_block), model)
# 	_add_edge!(block, edge)
# 	return edge
# end


# function _add_edge!(block::Block, edge::Edge{NTuple{1,Node}})
# 	push!(block.edges,edge)
# 	return
# end

# # wrapper for a coupling variable
# struct EdgeVariableIndex
# 	edge::Edge
# 	index::MOI.VariableIndex
# end

# function Base.string(ev::EdgeVariableIndex)
#     return """
#     	Edge (Block $(ev.edge.block_index.value), Edge $(ev.edge.index), Index $(ev.index.value))
#     """
# end
# Base.print(io::IO, ev::EdgeVariableIndex) = print(io, string(ev))
# Base.show(io::IO, ev::EdgeVariableIndex) = print(io, ev)

# # blocks retain mapping between node and edge variables
# # TODO: do we need this? we can just lookup the vertices on the edge.
# struct NodeEdgeMap
# 	node_to_edge::OrderedDict{NodeVariableIndex,EdgeVariableIndex}
# 	edge_to_node::OrderedDict{EdgeVariableIndex,NodeVariableIndex}
# end
# function NodeEdgeMap()
# 	node_to_edge = OrderedDict{NodeVariableIndex,EdgeVariableIndex}()
# 	edge_to_node = OrderedDict{EdgeVariableIndex,NodeVariableIndex}()
# 	return NodeEdgeMap(node_to_edge, edge_to_node)
# end

# function Base.getindex(node_edge_map::NodeEdgeMap, index::NodeVariableIndex)
# 	return node_edge_map.node_to_edge[index]
# end
# function Base.getindex(node_edge_map::NodeEdgeMap, index::EdgeVariableIndex)
# 	return node_edge_map.edge_to_node[index]
# end
# function Base.setindex!(node_edge_map::NodeEdgeMap, nindex::NodeVariableIndex, eindex::EdgeVariableIndex)
# 	return node_edge_map.node_to_edge[nindex] = eindex
# end
# function Base.setindex!(node_edge_map::NodeEdgeMap, eindex::EdgeVariableIndex, nindex::NodeVariableIndex)
# 	return node_edge_map.edge_to_node[eindex] = nindex
# end