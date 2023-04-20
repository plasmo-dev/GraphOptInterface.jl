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

### Node

"""
	Node

A node represents a set of variables and associated bounds. A node maintains both 
internal and external indexes for each variable. The internal indexes are used for 
interoperability with MOI objects. The external variable indexes represent the actual 
indices used to define constraints in an GraphModel.
"""
struct Node <: NodeModelLike
	index::HyperNode
	block_index::BlockIndex
	variables::MOIU.VariablesContainer
	# dual starts, etc...
	var_attr::Dict{MOI.AbstractVariableAttribute,Dict{MOI.VariableIndex, Any}}
end

function node_index(node::Node)::HyperNode
	return node.index
end

### MOI Node Functions

function MOI.add_variable(node::Node)
	return MOI.add_variable(node.variables)
end

function MOI.add_variables(node::Node, n::Int64)
	return MOI.add_variables(node.variables, n)
end

function MOI.add_constraint(
	node::Node,
	variable::MOI.VariableIndex,
	set::S
) where {S <: MOI.AbstractSet}
	return MOI.add_constraint(node.variables, variable, set)
end

### Edge

struct LinkVariableMap
	int_to_ext::MOIU.CleverDicts.CleverDict{MOI.VariableIndex,Tuple{HyperNode,MOI.VariableIndex}}
	ext_to_int::MOIU.CleverDicts.CleverDict{Tuple{HyperNode,MOI.VariableIndex},MOI.VariableIndex}
end

struct SingleNodeMap
	node::HyperNode
end

function node_variables(var_map::SingleNodeMap)
	return MOI.get(var_map.node, MOI.ListOfVariableIndices())
end

function node_variables(var_map::LinkVariableMap)
	return collect(values(var_map.int_to_ext))
end

"""
	Edge

An edge represents different types of coupling. For instance an Edge{Tuple{Node}} is an edge
the couple variables within a single node. An Edge{Tuple{N,Node}} couple variables across
one or more nodes.
"""
mutable struct Edge <: EdgeModelLike
	index::HyperEdge
	block_index::BlockIndex
	# NOTE: moi_model only stores constraints
	moi_model::MOIU.UniversalFallback{MOIU.Model{Float64}}
	nonlinear_model::Union{Nothing,MOI.Nonlinear.Model}
	variable_map::Union{LinkMap,SingleNodeMap}
end

function Edge(hyperedge::HyperEdge, block_index::BlockIndex)
	moi_model = MOIU.UniversalFallback{MOIU.Model{Float64}}()
	return Edge(index, block_index, variable_map, moi_model, nothing)
end

function edge_index(edge::Edge)::HyperEdge
	return edge.index
end

function is_self_edge(edge::Edge)
	return length(edge.index.vertices) == 1
end

function node_variables(edge::Edge)
	return node_variables(edge.variable_map)
end

### MOI Edge Functions

function MOI.add_variable(edge::Edge, node::Node, variable::MOI.VariableIndex)
	if is_self_edge(edge)
		error("Self edge variables should be added directly to the specified node")
	end
	internal_variable = MOI.add_variable(edge.moi_model)
	edge.variable_map[internal_variable] = (node.index, variable)
	return internal_variable
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
	expr::Expr,
	set::S
) where {S <: MOI.AbstractSet}
	edge.nonlinear_model == nothing && edge.nonlinear_model = MOI.Nonlinear.Model()
	ci = MOI.add_constraint(edge.nonlinear_model, expr, set)
	return ci
end

### Forward methods so Node and Edge objects call their underlying MOI models

# forward node methods
@forward Node.variables (MOI.get, MOI.set)

# forward edge methods
@forward Edge.moi_model (MOI.get, MOI.set)

### Blocks

mutable struct Block{T<:AbstractBlock} <: AbstractBlock
	index::BlockIndex
	parent_index::Union{Nothing,BlockIndex}
	nodes::Vector{Node}
	edges::Vector{Edge}
	sub_blocks::Vector{T}
	node_by_index::OrderedDict{HyperNode,Node}
	edge_by_index::OrderedDict{HyperEdge,Edge}
	block_by_index::OrderedDict{BlockIndex,Block}
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




### Block MOI Functions

function MOI.get(block::Block, attr::MOI.ListOfConstraintTypesPresent)
	ret = []
	for node in all_nodes(block)
		append!(ret, MOI.get(node.model, attr))
	end
	for edge in all_edges(block)
		append!(ret, MOI.get(edge.model, attr))
	end
    return unique(ret)
end

function MOI.get(block::Block, attr::MOI.NumberOfVariables)
    return sum(MOI.get(node, attr) for node in all_nodes(block))
end

function MOI.get(block::Block, attr::MOI.ListOfVariableIndices)
	return vcat(MOI.get(node, attr) for node in all_nodes(block))
end

function MOI.get(
    block::Block,
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

### GraphModel

struct GraphModel <: MOI.ModelLike
	hypergraph::HyperGraph
	block::Block
end
function GraphModel()
	return GraphModel(HyperGraph(), Block(0))
end

function number_of_blocks(graph::GraphModel)
	return length(graph.block_by_index)
end

"""
	add_node!(graph::GraphModel, block::T where T<:AbstractBlock)::Node

Add a node to `block` on `graph`. The index of the node is determined by the central graph.

	add_node!(graph::GraphModel)

Add a node to root graph block
"""
function add_node(graph::GraphModel, block::Block)::Node
	hypernode = Graphs.add_vertex!(graph.hypergraph)
	node = Node(hypernode)
	push!(block.nodes, node)
	return node
end

function add_node(graph::GraphModel)::Node
	return add_node!(graph.block)
end

### add_edge

function add_edge(graph::GraphModel, block::Block, nodes::NTuple{N,Node})::Edge
	# we do not allow arbitrary edges. at most between two layers.
	@assert isempty(setdiff(nodes, get_nodes_one_layer(block)))
	hypernodes = node_index.(nodes)
	hyperedge = add_edge!(graph.hypergraph, hypernodes)
	edge = Edge(hyperedge, block_index)
	push!(block.edges,edge)
	return edge
end

function add_edge(graph::GraphModel, node::Node)
	return add_edge(graph, graph.block, (node,))
end

### add_sub_block

function add_sub_block(graph::GraphModel, parent::Block)::Block
	# get the root block
	root = graph.block

	# define a new block index, create the sub-block
	new_block_index = number_of_blocks(graph)
	sub_block = Block(parent.index.value, new_block_index)
	push!(parent.sub_blocks, sub_block)

	# track new block index on root and block
	parent.block_by_index[sub_block.index] = sub_block
	if root.index != parent.index
		root.block_by_index[sub_block.index] = sub_block
	end
	return sub_block
end

function add_sub_block(graph::GraphModel)::Block
	root = graph.block
	block = add_sub_block(graph, root)
	return block
end

### Query graph

function get_nodes(block::Block)
	nodes = block.nodes
	return block.nodes
end

function get_nodes_to_depth(block::Block, n_layers::Int=0)
	nodes = block.nodes
	if n_layers > 0
		for sub_block in block.sub_blocks
			nodes = get_nodes_to_depth(sub_block, n_layers-1)
			nodes = [nodes; sub_block.nodes]
		end
	end
	return nodes
end

function get_edges(block::Block)
	return block.edges
end

function all_nodes(block::Block)
	nodes = block.nodes
	if !isempty(block.sub_blocks)
		for sub_block in block.sub_blocks
			append!(nodes, all_nodes(sub_block))
		end
	end
	return nodes
end

# edges attached to a single node
function self_edges(block::Block)
	return filter((edge) -> length(edge.index.vertices) == 1, block.edges)
end

# edges that connect nodes
function linking_edges(block::Block)
	return filter((edge) -> length(edge.index.vertices) > 1, block.edges)
end

# nodes connected to edge
function connected_nodes(graph::GraphModel, edge::Edge)
	vertices = edge.vertices
	return getindex.(Ref(graph.nodes_by_index), vertices)
end





### Coupling Variables

# function add_coupling_variable(edge::Edge, node::Node, vi::MOI.VariableIndex)::MOI.VariableIndex
# 	@assert node.index in edge.elements
# 	if length(edge.elements) === 1
# 		error("self-edges cannot have coupling variables")
# 	end
# 	# if edge variable already exists, return it
# 	if (node.index,vi) in keys(edge.node_variable_map)
# 		return edge.node_variable_map[(node.index,vi)]
# 	else
# 		num_edge_variables = length(edge.edge_variable_map)
# 		edge_vi = MOI.VariableIndex(num_edge_variables + 1)
# 		edge.edge_variable_map[edge_vi] = (node.index,vi)
# 		edge.node_variable_map[(node.index,vi)] = edge_vi
# 		return edge_vi
# 	end
# end

### MOI Functions

# function MOI.get(
# 	edge::Edge,
# 	attr::MOI.ListOfVariableIndices
# )::Vector{MOI.VariableIndex}
# 	# TODO: get variable indices from terms
# 	return MOI.get(node, MOI.ListOfVariableIndices())
# end

# # get list of variable indices on the edge
# function MOI.get(
# 	edge::Edge,
# 	attr::MOI.ListOfVariableIndices
# )::Vector{MOI.VariableIndex}
# 	return collect(values(edge.edge_variables))
# end

# function get_node_variables(graph::GraphModel, edge::Edge{1})
# 	block = graph.block.block_by_index[edge.block_index]
# 	node = node_by_index(graph, edge.hyperedge.vertices[1])
# 	variables = MOI.get(node, MOI.ListOfVariableIndices())
# 	return (node.index, variables)
# end

# function get_node_variables(graph::GraphModel, edge::Edge)
# 	return graph.block.variables[edge]
# end


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


### Other coupling ideas...
# NOTE: I think this is too much customization on top of MOI. We would have to redefine every single struct.

# struct NodeAffineTerm{T} <: MOI.AbstractScalarFunction
#     coefficient::T
#     variable::NodeVariableIndex
# end

# function ScalarAffineTerm(coefficient::T, variable::NodeVariableIndex)
# 	return ScalarNodeAffineTerm(coefficient, variable)
# end

# struct LinkAffineFunction{T} <: MOI.AbstractScalarFunction
# 	terms::Vector{NodeAffineTerm{T}}
# 	constant::T
# end

# function MOI.add_constraint(
# 	block::Block,
# 	edge::Edge,
# 	func::ScalarLinkingFunction,
# 	set::S
# ) where {S <: MOI.AbstractSet}
# 	# map func.terms to local edge variables
	
# 	# add a scalar affine function to the edge

# 	ci = MOI.add_constraint(edge.moi_model, func, set)

# 	# update edge mapping

# 	return ci
# end

# TODO
# coefficient
# term_indices
# term_pair

	# external_var = MOI.VariableIndex(MOI.get(graph, MOI.NumberOfVariables()))
	# node.int_to_ext[internal_var] = external_var
	# node.ext_to_int[external_var] = internal_var
	# return external_var