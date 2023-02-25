# abstract type BlockModelLike <: MOI.ModelLike end
abstract type NodeModelLike <: MOI.ModelLike end
abstract type EdgeModelLike <: MOI.ModelLike end


"""
	Node
A block that supports variable data
"""
struct Node #<: MOI.ModelLike
	index::Int64
	model::NodeModelLike
end

# An edge can represent different types of coupling
struct Edge{T<:Tuple} #<: MOI.ModelLike
	index::Int64
	elements::T # could be a node, nodes, blocks, or both
	model::EdgeModelLike
end

struct BlockIndex
	value::Int64
end

mutable struct Block #<: BlockModelLike
	index::BlockIndex
	nodes::Vector{Node}
	edges::Vector{Edge}
	sub_blocks::Vector{Block}
	block_by_index::OrderedDict{BlockIndex,Block}
	function Block(index::Int64)
		block = new()
		block.index = BlockIndex(index)
		block.nodes = Vector{Node}()
		block.edges = Vector{Edge}()
		block.sub_blocks = Vector{Block}()
		block.block_by_index = OrderedDict()
		block.block_by_index[block.index] = block 
		return block
	end
end

function all_nodes(block::Block)
	return block.nodes
end

# Interface Functions
# add a node to a model on `AbstractBlock`
function add_node!(::AbstractBlockOptimizer, ::BlockIndex) end

# self-edge, one node
function add_edge!(::AbstractBlockOptimizer, ::BlockIndex, ::Node) end

# edge between nodes
function add_edge!(::AbstractBlockOptimizer, ::BlockIndex, ::NTuple{N, Node}) where N end

# edge containing sub-blocks
function add_edge!(::AbstractBlockOptimizer, ::BlockIndex, ::NTuple{N, Block}) where N end

# edge containing node and sub-block
function add_edge!(::AbstractBlockOptimizer, ::BlockIndex, ::Node, ::Block) end

# add a sub-block to block with `BlockIndex`
function add_sub_block!(::AbstractBlockOptimizer, ::BlockIndex) end


@forward Node.model (MOI.add_variable, MOI.add_variables, MOI.get, MOI.set)
@forward Edge.model (MOI.get, MOI.set, MOI.add_constraint)

# Block functions
function add_node!(block::Block, model::NodeModelLike)::Node
	node = Node(length(block.nodes) + 1, model)
	push!(block.nodes, node)
	return node
end

# Edges
function add_edge!(block::Block, node::Node, model::EdgeModelLike)::Edge
	index = length(block.edges) + 1
	edge = Edge{NTuple{1,Node}}(index, (node,), model)
	push!(block.edges, edge)
	return edge
end

function add_edge!(block::Block, nodes::NTuple{N, Node} where N, model::EdgeModelLike)::Edge
	index = length(block.edges) + 1
	edge = Edge{NTuple{length(nodes),Node}}(index, nodes, model)
	push!(block.edges, edge)
	return edge
end

function add_edge!(block::Block, blocks::NTuple{N, Block} where N, model::EdgeModelLike)::Edge
	index = length(block.edges) + 1
	edge = Edge{NTuple{length(blocks),Node}}(index, nodes, model)
	push!(block.edges, edge)
	return
end

function add_edge!(block::Block, node::Node, sub_block::Block, model::EdgeModelLike)::Edge
	index = length(block.edges) + 1
	edge = Edge{Tuple{Node, Block}}(index, (node, sub_block), model)
	push!(block.edges, edge)
	return edge
end

function add_sub_block!(block::Block)::Block
	sub_block = Block()
	push!(block.sub_blcoks, sub_block)
	return sub_block
end


# MOI functions over a `Block`
function MOI.get(block::Block, attr::MOI.ListOfConstraintTypesPresent)
	ret = []
	for node in all_nodes(block)
		append!(ret, MOI.get(node.model, attr))
	end
	for edge in get_edges(block)
		append!(ret, MOI.get(edge.model, attr))
	end
    return unique(ret)
end

function MOI.get(
    block::Block,
    attr::MOI.NumberOfVariables,
)
    return sum(MOI.get(node.model, attr) for node in all_nodes(block))
end


function MOI.get(
    model::Node,
    attr::MOI.ListOfVariableIndices,
)
	var_list = []
	for node in all_nodes(block)
		append!(var_list, MOI.get(node.model))
	end
    return MOI.get(model.variables, attr)
end

