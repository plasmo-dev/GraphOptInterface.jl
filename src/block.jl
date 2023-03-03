# abstract type BlockModelLike <: MOI.ModelLike end
abstract type NodeModelLike <: MOI.ModelLike end
abstract type EdgeModelLike <: MOI.ModelLike end


"""
	Node

A set of variables. A node contains a `model` that implements the `NodeModelLike` interface.
"""
struct Node
	index::Int64
	variables::Vector{MOI.VariableIndex}
	# model::NodeModelLike
	# local variable indexed to block index
	# variable_map::OrderedDict{MOI.VariableIndex,MOI.VariableIndex}
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
	return NodeVariableIndex(node,MOI.VariableIndex(index))
end

function num_variables(node::Node)
    return MOI.get(node, MOI.NumberOfVariables())
end

"""
	Edge{T<:Tuple}

An edge represents different types of coupling. For instance an Edge{Tuple{Node}} is an edge
the couple variables within a single node. An Edge{Tuple{N,Node}} couple variables across
one or more nodes.
"""
struct Edge{T<:Tuple}
	index::Int64
	elements::T # could be a node, nodes, blocks, or both
	model::EdgeModelLike
	# local constraint index to block index
	# constraint_map::OrderedDict{MOI.ConstraintIndex,MOI.ConstraintIndex}
end

struct EdgeVariableIndex
	edge::Edge
	index::MOI.VariableIndex
end

const NodeEdgeMap = OrderedDict{NodeVariableIndex,EdgeVariableIndex}

function num_constraints(edge::Edge)
    n_con = 0
    for (F,S) in MOI.get(edge, MOI.ListOfConstraintTypesPresent())
        n_con += MOI.get(edge, MOI.NumberOfConstraints{F,S}())
    end
    nlp_block = MOI.get(edge, MOI.NLPBlock())
    n_con += length(nlp_block.constraint_bounds)
    return n_con
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
	edge_variable_map::Dict{Edge,NodeEdgeMap}
	function Block(index::Int64)
		block = new()
		block.index = BlockIndex(index)
		block.nodes = Vector{Node}()
		block.edges = Vector{Edge}()
		block.sub_blocks = Vector{Block}()
		block.block_by_index = OrderedDict()
		block.block_by_index[block.index] = block 
		block.edge_variable_map = Dict{Edge,NodeEdgeMap}()
		return block
	end
end

### edge_variable

function edge_variable(block::Block, edge::Edge{NTuple{1,Node}}, nv::NodeVariableIndex)
	@assert edge.elements == (nv.node,)
	return nv.index
end

function edge_variable(block::Block, edge::Edge, nvi::NodeVariableIndex)
	@assert nvi.node in all_nodes(edge)
	if !haskey(block.edge_variable_map[edge], nvi)
		variable_index = MOI.VariableIndex(length(block.edge_variable_map[edge])+1)
		block.edge_variable_map[edge][nvi] = EdgeVariableIndex(edge, variable_index)
		return variable_index
	else
		return block.edge_variable_map[edge][nvi].index
	end
end

### edge_variables

function edge_variables(block::Block, edge::Edge)
	error("Edge connects multiple nodes or blocks. You must provide specific node indices.")
end

function edge_variables(block::Block, edge::Edge{NTuple{1,Node}})
	return MOI.get(edge.elements[1], MOI.ListOfVariableIndices())
end

function edge_variables(block::Block, edge::Edge, nvis::Vector{NodeVariableIndex})
	return edge_variable.(Ref(block), Ref(edge), nvis)
end

### getters

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
	node = Node(length(block.nodes) + 1, model)
	push!(block.nodes, node)
	return node
end

function add_sub_block!(block::Block)::Block
	sub_block = Block(length(block.sub_blocks) + 1)
	push!(block.sub_blocks, sub_block)
	return sub_block
end

# IDEA: one method that takes the edge object
function _add_edge!(block::Block, edge::Edge)
	push!(block.edges,edge)
	block.edge_variable_map[edge] = NodeEdgeMap()
	return
end
# Edges
function add_edge!(block::Block, node::Node, model::EdgeModelLike)::Edge
	index = length(block.edges) + 1
	edge = Edge{NTuple{1,Node}}(index, (node,), model)
	_add_edge!(block, edge)
	return edge
end

function add_edge!(block::Block, nodes::NTuple{N, Node} where N, model::EdgeModelLike)::Edge
	index = length(block.edges) + 1
	edge = Edge{NTuple{length(nodes),Node}}(index, nodes, model)
	_add_edge!(block, edge)
	return edge
end

function add_edge!(block::Block, blocks::NTuple{N, Block} where N, model::EdgeModelLike)::Edge
	index = length(block.edges) + 1
	edge = Edge{NTuple{length(blocks),Node}}(index, nodes, model)
	_add_edge!(block, edge)
	return
end

function add_edge!(block::Block, node::Node, sub_block::Block, model::EdgeModelLike)::Edge
	index = length(block.edges) + 1
	edge = Edge{Tuple{Node, Block}}(index, (node, sub_block), model)
	_add_edge!(block, edge)
	return edge
end

### Block Functions

column(x::MOI.VariableIndex) = x.value

function MOI.add_variable(optimizer::AbstractBlockOptimizer, node::Node)
	local_vi = MOI.add_variable(node.model)
	# block index is the total number of variables
	#block_index = MOI.get(optimizer.block, MOI.NumberOfVariables())
	#block_vi = MOI.VariableIndex(block_index)
	#node.variable_map[local_vi] = block_vi
	return NodeVariableIndex(node,local_vi)
end

function MOI.add_variables(optimizer::AbstractBlockOptimizer, node::Node, n::Int64)
	vars = NodeVariableIndex[]
	for _ = 1:n
		nvi = MOI.add_variable(optimizer, node)
		push!(vars, nvi)
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
	#edge.constraint_map[local_ci] = block_ci
	return block_ci
end

# function column_inds(node::Node)
# 	return column.(values(node.variable_map))
# end

# function column_inds(edge::Edge)
# 	nodes = all_nodes(edge)
# 	inds = Int64[]
# 	for node in nodes
# 		append!(inds, column_inds(node))
# 	end
# 	return inds
# end


### Block attributes
function MOI.get(block::Block, attr::MOI.ListOfConstraintTypesPresent)
	ret = []
	# for node in all_nodes(block)
	# 	append!(ret, MOI.get(node.model, attr))
	# end
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

# NOTE: we map the evaluation to the actual column indices on the edge
# function MOI.eval_objective(edge::Edge, x)
# 	col_inds = column_inds(edge)
# 	x_eval = SparseArrays.sparsevec(col_inds, x)
# 	return MOI.eval_objective(edge.model, x_eval)
# end

# function MOI.eval_constraint(edge::Edge, g, x)
# 	col_inds = column_inds(edge)
# 	x_eval = SparseArrays.sparsevec(col_inds, x)
# 	return MOI.eval_constraint(edge.model, g, x_eval)
# end
