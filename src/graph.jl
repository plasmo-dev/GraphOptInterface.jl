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

A node represents a set of variables and associated attributes. 
"""
struct Node
	index::HyperNode
	block_index::BlockIndex
	variables::MOIU.VariablesContainer
	# variable attributes primal starts, etc...
	var_attr::Dict{MOI.AbstractVariableAttribute,Dict{MOI.VariableIndex, Any}}
end
function Node(vertex::HyperNode, block_index::BlockIndex)
	var_attributes = Dict{MOI.AbstractVariableAttribute,Dict{MOI.VariableIndex, Any}}()
	var_attributes[MOI.VariablePrimalStart()] = Dict{MOI.VariableIndex,Any}()
	return Node(
		vertex,
		block_index,
		MOIU.VariablesContainer{Float64}(),
		var_attributes
	)
end

function node_index(node::Node)::HyperNode
	return node.index
end

### MOI Node Functions

function MOI.add_variable(node::Node)
	var = MOI.add_variable(node.variables)
	MOI.set(node, MOI.VariablePrimalStart(), var, nothing)
	return var
end

function MOI.add_variables(node::Node, n::Int64)
	vars = MOI.VariableIndex[]
	for _ = 1:n
		push!(vars, MOI.add_variable(node))
	end
	return vars
end

function MOI.add_constraint(
	node::Node,
	variable::MOI.VariableIndex,
	set::S
) where {S <: MOI.AbstractSet}
	return MOI.add_constraint(node.variables, variable, set)
end

function MOI.set(node::Node, attr::MOI.VariablePrimalStart, vi::MOI.VariableIndex, value::Any)
	node.var_attr[attr][vi] = value
end

function MOI.get(node::Node, attr::MOI.VariablePrimalStart, vi::MOI.VariableIndex)
	return node.var_attr[attr][vi]
end

### Edge
abstract type AbstractNodeVariableMap end

struct NodeIndexMap <: AbstractNodeVariableMap
	int_to_ext::OrderedDict{MOI.VariableIndex,Tuple{HyperNode,MOI.VariableIndex}}
	ext_to_int::OrderedDict{Tuple{HyperNode,MOI.VariableIndex},MOI.VariableIndex}
end
function NodeIndexMap()
	int_to_ext = OrderedDict{MOI.VariableIndex,Tuple{HyperNode,MOI.VariableIndex}}()
	ext_to_int = OrderedDict{Tuple{HyperNode,MOI.VariableIndex},MOI.VariableIndex}()
	return NodeIndexMap(int_to_ext, ext_to_int)
end

struct SelfEdgeMap <: AbstractNodeVariableMap
	node::Node
end

function node_variables(var_map::NodeIndexMap)
	return collect(values(var_map.int_to_ext))
end

function node_variables(var_map::SelfEdgeMap)
	return (var_map.node.index, MOI.get(var_map.node, MOI.ListOfVariableIndices()))
end

"""
	Edge

An edge represents different types of coupling. For instance an Edge{Tuple{Node}} is an edge
the couple variables within a single node. An Edge{Tuple{N,Node}} couple variables across
one or more nodes.
"""
mutable struct Edge{T <: AbstractNodeVariableMap}
	index::HyperEdge
	block_index::BlockIndex
	# NOTE: moi_model only stores constraints, not variable indexes
	moi_model::MOIU.UniversalFallback{MOIU.Model{Float64}}
	nonlinear_model::Union{Nothing,MOI.Nonlinear.Model}
	variable_map::T
end

function Edge(
	hyperedge::HyperEdge,
	block_index::BlockIndex,
	var_map::T
) where T <: AbstractNodeVariableMap
	moi_model = MOIU.UniversalFallback(MOIU.Model{Float64}())
	return Edge{typeof(var_map)}(hyperedge, block_index, moi_model, nothing, var_map)
end

function edge_index(edge::Edge)::HyperEdge
	return edge.index
end

function is_self_edge(edge::Edge)
	return length(edge.index.vertices) === 1
end

function node_variables(edge::Edge)
	return node_variables(edge.variable_map)
end

### MOI Edge Functions

function MOI.add_variable(edge::Edge{NodeIndexMap}, node::Node, variable::MOI.VariableIndex)
	internal_variable = MOI.add_variable(edge.moi_model)
	edge.variable_map.int_to_ext[internal_variable] = (node.index, variable)
	edge.variable_map.ext_to_int[(node.index, variable)] = internal_variable
	return internal_variable
end

function MOI.get(edge::Edge{SelfEdgeMap}, attr::MOI.ListOfVariableIndices)
	return MOI.get(edge.variable_map.node, attr)
end

function MOI.get(edge::Edge{NodeIndexMap}, ::MOI.ListOfVariableIndices)
	return collect(keys(edge.variable_map.int_to_ext))
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
	edge.nonlinear_model === nothing && (edge.nonlinear_model = MOI.Nonlinear.Model())
	constraint_index = MOI.Nonlinear.add_constraint(edge.nonlinear_model, expr, set)
	return constraint_index
end

### Forward methods so Node and Edge objects call their underlying MOI models

@forward Node.variables (MOI.get, MOI.set)

@forward Edge.moi_model (MOI.get, MOI.set)

### Blocks

mutable struct Block
	index::BlockIndex
	parent_index::Union{Nothing,BlockIndex}
	nodes::Vector{Node}
	edges::Vector{Edge}
	sub_blocks::Vector{Block}
	node_by_index::OrderedDict{HyperNode,Node}
	edge_by_index::OrderedDict{HyperEdge,Edge}
	block_by_index::OrderedDict{BlockIndex,Block}
	function Block(block_index::Int64)
		block = new()
		block.index = BlockIndex(block_index)
		block.nodes = Node[]
		block.edges = Edge[]
		block.parent_index = nothing
		block.sub_blocks = Vector{Block}()
		block.node_by_index = OrderedDict{HyperNode,Node}()
		block.edge_by_index = OrderedDict{HyperEdge,Edge}()
		block.block_by_index = OrderedDict{BlockIndex,Block}()
		block.block_by_index[block.index] = block
		return block
	end
	function Block(parent_index::Int64, block_index::Int64)
		block = Block(block_index)
		block.parent_index = BlockIndex(parent_index)
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
Base.print(io::IO, block::Block) = Base.print(io, Base.string(block))
Base.show(io::IO, block::Block) = Base.print(io, block)

### Graph

struct Graph <: MOI.ModelLike
	hypergraph::HyperGraph
	block::Block
end
function Graph()
	return Graph(HyperGraph(), Block(0))
end

function number_of_blocks(graph::Graph)
	return length(graph.block.block_by_index)
end

"""
	add_node!(graph::Graph, block::T where T<:Block)::Node

Add a node to `block` on `graph`. The index of the node is determined by the central graph.

	add_node!(graph::Graph)

Add a node to root graph block
"""
function add_node(graph::Graph, block::Block)::Node
	hypernode = Graphs.add_vertex!(graph.hypergraph)
	node = Node(hypernode, block.index)
	push!(block.nodes, node)
	return node
end

function add_node(graph::Graph)::Node
	return add_node(graph, graph.block)
end

### add_edge

function add_edge(graph::Graph, nodes::NTuple{N,Node})::Edge where N
	return add_edge(graph, graph.block, nodes)
end

function add_edge(graph::Graph, block::Block, nodes::NTuple{N,Node})::Edge where N
	# we do not allow arbitrary edges. at most between two layers.
	@assert isempty(setdiff(nodes, get_nodes_to_depth(block, 1)))
	hypernodes = node_index.(nodes)
	hyperedge = add_edge!(graph.hypergraph, hypernodes...)

	if length(nodes) === 1
		edge = Edge(hyperedge, block.index, SelfEdgeMap(nodes[1]))
	else
		edge = Edge(hyperedge, block.index, NodeIndexMap())
	end
	push!(block.edges,edge)
	return edge
end

function add_edge(graph::Graph, node::Node)
	block = graph.block.block_by_index[node.block_index]
	return add_edge(graph, block, (node,))
end

### add_sub_block

function add_sub_block(graph::Graph, parent::Block)::Block
	root = graph.block

	# define a new block index, create the sub-block
	new_block_index = number_of_blocks(graph)
	sub_block = Block(parent.index.value, new_block_index)
	push!(parent.sub_blocks, sub_block)

	# track new block index on root and block
	root.block_by_index[BlockIndex(new_block_index)] = sub_block
	if root.index != parent.index
		parent.block_by_index[BlockIndex(new_block_index)] = sub_block
	end
	return sub_block
end

function add_sub_block(graph::Graph)::Block
	root = graph.block
	block = add_sub_block(graph, root)
	return block
end