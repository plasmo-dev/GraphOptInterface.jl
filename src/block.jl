abstract type AbstractBlock end
abstract type NodeModelLike <: MOI.ModelLike end
abstract type EdgeModelLike <: MOI.ModelLike end

### BlockIndex

struct BlockIndex
	value::Int64
end

function raw_index(index::BlockIndex)
	return index.value
end

### Node and Edge Definitions

struct Node <: NodeModelLike
	index::HyperNode
	block_index::BlockIndex
	variables::MOIU.VariablesContainer
	var_attr::Dict{MOI.AbstractVariableAttribute,Dict{MOI.VariableIndex, Any}}
end

function node_index(node::Node)::HyperNode
	return node.index
end

"""
	Edge

An edge represents different types of coupling. For instance an Edge{Tuple{Node}} is an edge
the couple variables within a single node. An Edge{Tuple{N,Node}} couple variables across
one or more nodes.
"""
mutable struct Edge{N} <: EdgeModelLike
	index::HyperEdge
	block_index::BlockIndex
	moi_model::MOIU.UniversalFallback{MOIU.Model{Float64}}
	nonlinear_model::Union{Nothing,MOI.Nonlinear.Model}
end
function Edge(hyperedge::HyperEdge, block_index::BlockIndex)
	# if Graphs.nv(hyperedge) > 1
	# 	variable_map = OrderedDict{MOI.VariableIndex,Tuple{HyperNode,MOI.VariableIndex}}()
	# else
	# 	variable_map = nothing
	# end
	moi_model = MOIU.UniversalFallback{MOIU.Model{Float64}}()
	return Edge{length(hyperedge.vertices)}(index, block_index, variable_map, moi_model, nothing)
end

function edge_index(edge::Edge)::HyperEdge
	return edge.index
end

struct OptiGraph
	hypergraph::HyperGraph
	block::Block
end
function OptiGraph()
	return OptiGraph(Block(0), HyperGraph())
end

### Blocks
# a block contains nodes, edges, and sub-blocks

mutable struct Block{T<:AbstractBlock} <: AbstractBlock
	index::BlockIndex
	parent_index::Union{Nothing,BlockIndex}
	nodes::Vector{Node}
	edges::Vector{Edge}
	sub_blocks::Vector{T}
	block_by_index::OrderedDict{BlockIndex,Block}
	edge_variable_map::OrderedDict{Tuple{HyperEdge,MOI.VariableIndex},Tuple{HyperNode,MOI.VariableIndex}}}
	function Block(block_index::Int64)
		block = new{Block}()
		block.index = BlockIndex(block_index)
		block.parent_index = nothing
		block.sub_blocks = Vector{Block}()
		block.block_by_index = OrderedDict{BlockIndex,Block}()
		block.block_by_index[block.index] = block
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
function add_node(graph::OptiGraph, block::Block)::Node
	hypernode = Graphs.add_vertex!(graph.hypergraph)
	node = Node(hypernode)
	push!(block.nodes, node)
	return node
end

function add_node(graph::OptiGraph)::Node
	return add_node!(graph.block)
end

function add_edge(graph::OptiGraph, block::Block, nodes::NTuple{N,Node})::Edge
	# nodes 
	@assert isempty(setdiff(nodes, get_nodes_one_layer(block)))
	hypernodes = node_index.(nodes)
	hyperedge = add_edge!(graph.hypergraph, hypernodes)
	edge = Edge(hyperedge, block_index)
	push!(block.edges,edge)
	return edge
end

function add_edge(graph::OptiGraph, node::Node)
	return add_edge(graph, graph.block, (node,))
end

function add_sub_block(graph::OptiGraph, parent::Block)::Block
	# get the root block
	root = graph.block

	# define a new block index, create the sub-block
	new_block_index = length(root.block_by_index)
	sub_block = Block(parent.index.value, new_block_index)
	push!(parent.sub_blocks, sub_block)

	# track new block index
	parent.block_by_index[sub_block.index] = sub_block
	root.block_by_index[sub_block.index] = sub_block
	return sub_block
end

function add_sub_block!(graph::OptiGraph)::Block
	root = graph.block
	block = add_sub_block!(graph, root)
	return block
end


### Query Functions

function get_nodes(block::Block)
	nodes = block.nodes
	return block.nodes
end

function get_nodes_one_layer(block::Block)
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

function add_coupling_variable(edge::Edge, node::Node, vi::MOI.VariableIndex)::MOI.VariableIndex
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

# function MOI.get(
# 	edge::Edge{1},
# 	attr::MOI.ListOfVariableIndices
# )::Vector{MOI.VariableIndex}
# 	node = edge.elements[1]
# 	return MOI.get(node, MOI.ListOfVariableIndices())
# end

# # get list of variable indices on the edge
# function MOI.get(
# 	edge::Edge,
# 	attr::MOI.ListOfVariableIndices
# )::Vector{MOI.VariableIndex}
# 	return collect(values(edge.edge_variables))
# end

function get_node_variables(graph::OptiGraph, edge::Edge{1})
	block = graph.block.block_by_index[edge.block_index]
	node = node_by_index(graph, edge.hyperedge.vertices[1])
	variables = MOI.get(node, MOI.ListOfVariableIndices())
	return (node.index, variables)
end

function get_node_variables(graph::OptiGraph, edge::Edge)
	return graph.block.variables[edge]
end

function MOI.add_variable(node::Node)
	return MOI.add_variable(node.variables)
end

function MOI.add_variables(node::Node, n::Int64)
	return MOI.add_variables(node.variables, n)
end

function MOI.add_constraint(
	node::Node,
	vi::MOI.VariableIndex,
	set::S
) where {S <: MOI.AbstractSet}
	return MOI.add_constraint(node.variables, vi, set)
end

function MOI.add_constraint(
	edge::Edge,
	func::F,
	set::S
) where {F <: MOI.AbstractFunction, S <: MOI.AbstractSet}
	ci = MOI.add_constraint(edge.moi_model, func, set)
	return ci
end

function MOI.Nonlinear.add_constraint(
	edge::Edge,
	func::Expr,
	set::S
) where {S <: MOI.AbstractSet}
	edge.nonlinear_model == nothing && edge.nonlinear_model = MOI.Nonlinear.Model()
	ci = MOI.add_constraint(edge.nonlinear_model, expr, set)
	return ci
end

function MOI.set(
    edge::Edge,
    ::MOI.NLPBlockDualStart,
    values::Union{Nothing,Vector},
)
    MOI.set(edge.nonlinear_model,values)
    return
end

MOI.get(edge::EdgeModel, ::MOI.NLPBlockDualStart) = edge.nlp_dual_start

### Forward methods so Node and Edge objects call their underlying MOI model

# forward node methods
@forward Node.variables (MOI.get, MOI.set)

# forward edge methods
@forward Edge.model (MOI.get, MOI.set, MOI.eval_objective, MOI.eval_objective_gradient,
	MOI.eval_constraint, MOI.jacobian_structure, MOI.eval_constraint_jacobian,
	MOI.hessian_lagrangian_structure, MOI.eval_hessian_lagrangian
)

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