# abstract type BlockModelLike <: MOI.ModelLike end
abstract type NodeModelLike <: MOI.ModelLike end
abstract type EdgeModelLike <: MOI.ModelLike end


"""
	Node
A set of variables. A node contains a `model` that implements the `NodeModelLike` interface.
"""
struct Node
	index::Int64
	model::NodeModelLike
	# local variable inxed to block index
	variable_map::OrderedDict{MOI.VariableIndex,MOI.VariableIndex}
end

# An edge can represent different types of coupling
struct Edge{T<:Tuple}
	index::Int64
	elements::T # could be a node, nodes, blocks, or both
	model::EdgeModelLike

	# local constraint index to block index
	constraint_map::OrderedDict{MOI.ConstraintIndex,MOI.ConstraintIndex}
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

# TODO: recursively check blocks
function all_nodes(block::Block)
	return block.nodes
end

function all_nodes(node::Node)
	return [node]
end

function all_edges(block::Block)
	return block.edges
end

function all_nodes(edge::Edge)
	nodes = []
	for element in edge.elements
		append!(nodes, all_nodes(element))
	end
	return nodes
end

### Interface Functions

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

# Block functions
function add_node!(block::Block, model::NodeModelLike)::Node
	node = Node(length(block.nodes) + 1, model, Dict{MOI.VariableIndex,MOI.VariableIndex}())
	push!(block.nodes, node)
	return node
end

function add_sub_block!(block::Block)::Block
	sub_block = Block(length(block.sub_blocks) + 1)
	push!(block.sub_blocks, sub_block)
	return sub_block
end

# IDEA: one method that takes the edge object
# Edges
function add_edge!(block::Block, node::Node, model::EdgeModelLike)::Edge
	index = length(block.edges) + 1
	edge = Edge{NTuple{1,Node}}(index, (node,), model, Dict{MOI.ConstraintIndex,MOI.ConstraintIndex}())
	push!(block.edges, edge)
	return edge
end

function add_edge!(block::Block, nodes::NTuple{N, Node} where N, model::EdgeModelLike)::Edge
	index = length(block.edges) + 1
	edge = Edge{NTuple{length(nodes),Node}}(index, nodes, model, Dict{MOI.ConstraintIndex,MOI.ConstraintIndex}())
	push!(block.edges, edge)
	return edge
end

function add_edge!(block::Block, blocks::NTuple{N, Block} where N, model::EdgeModelLike)::Edge
	index = length(block.edges) + 1
	edge = Edge{NTuple{length(blocks),Node}}(index, nodes, model, Dict{MOI.ConstraintIndex,MOI.ConstraintIndex}())
	push!(block.edges, edge)
	return
end

function add_edge!(block::Block, node::Node, sub_block::Block, model::EdgeModelLike)::Edge
	index = length(block.edges) + 1
	edge = Edge{Tuple{Node, Block}}(index, (node, sub_block), model, Dict{MOI.ConstraintIndex,MOI.ConstraintIndex}())
	push!(block.edges, edge)
	return edge
end

### Block Functions

column(x::MOI.VariableIndex) = x.value

function MOI.add_variable(optimizer::AbstractBlockOptimizer, node::Node)
	local_vi = MOI.add_variable(node.model)
	# block index is the total number of variables
	block_index = MOI.get(optimizer.block, MOI.NumberOfVariables())
	block_vi = MOI.VariableIndex(block_index)
	node.variable_map[local_vi] = block_vi
	return block_vi
end

function MOI.add_variables(optimizer::AbstractBlockOptimizer, node::Node, n::Int64)
	vars = MOI.VariableIndex[]
	for _ = 1:n
		vi = MOI.add_variable(optimizer, node)
		push!(vars, vi)
	end
	return vars
end


# forward methods so Node and Edge call their underlying model
@forward Node.model (MOI.get, MOI.set)
@forward Edge.model (MOI.get, MOI.set, MOI.eval_objective, MOI.eval_objective_gradient,
	MOI.eval_constraint, MOI.jacobian_structure, MOI.eval_constraint_jacobian,
	MOI.hessian_lagrangian_structure, MOI.eval_hessian_lagrangian
)

function MOI.add_constraint(optimizer::AbstractBlockOptimizer, 
	edge::Edge,
	func::F,
	set::S
) where {F <: MOI.AbstractFunction, S <: MOI.AbstractSet}
	
	local_ci = MOI.add_constraint(edge.model, func, set)
	block_index = MOI.get(optimizer.block, MOI.NumberOfConstraints{F,S}())
	block_ci = MOI.ConstraintIndex{F,S}(block_index)
	edge.constraint_map[local_ci] = block_ci
	return block_ci
end

function column_inds(node::Node)
	return column.(values(node.variable_map))
end

function column_inds(edge::Edge)
	nodes = all_nodes(edge)
	inds = Int64[]
	for node in nodes
		append!(inds, column_inds(node))
	end
	return inds
end


### Block attributes
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

function MOI.get(block::Block, attr::MOI.NumberOfVariables)
    return sum(MOI.get(node, attr) for node in all_nodes(block))
end

function MOI.get(block::Block, attr::MOI.ListOfVariableIndices)
	var_list = []
	for node in all_nodes(block)
		append!(var_list, MOI.get(node, attr))
	end
    return var_list
end

function MOI.get(
    block::Block,
    attr::MOI.NumberOfConstraints{F,S}
) where {F <: MOI.AbstractFunction, S <: MOI.AbstractSet}

    return sum(MOI.get(edge, attr) for edge in all_edges(block))
end