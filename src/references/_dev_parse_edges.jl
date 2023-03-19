# function BOI.add_node!(optimizer::SchurOptimizer, index::BOI.BlockIndex)
#     block = optimizer.block.block_by_index[index]
#     node = BOI.add_node!(block, NodeModel(optimizer))
# end

# function BOI.add_edge!(optimizer::SchurOptimizer, index::BOI.BlockIndex, node::BOI.Node)
#     block = optimizer.block.block_by_index[index]
#     return BOI.add_edge!(block, node, EdgeModel(optimizer))
# end

# function BOI.add_edge!(optimizer::SchurOptimizer, index::BOI.BlockIndex, nodes::NTuple{N, BOI.Node} where N)
#     block = optimizer.block.block_by_index[index]
#     return BOI.add_edge!(block, nodes, EdgeModel(optimizer))
# end

# function BOI.add_edge!(optimizer::SchurOptimizer, index::BOI.BlockIndex, blocks::NTuple{N, BOI.Block} where N)
#     block = optimizer.block.block_by_index[index]
#     return BOI.add_edge!(block, blocks, EdgeModel(optimizer))
# end

# function BOI.add_edge!(optimizer::SchurOptimizer, index::BOI.BlockIndex, node::BOI.Node, block::BOI.Block)
#     parent_block = optimizer.block.block_by_index[index]
#     return BOI.add_edge!(parent_block, node, block, EdgeModel(optimizer))
# end

# TODO: figure out whether we can do these
# MOI.supports_incremental_interface(::SchurOptimizer) = false
# function MOI.copy_to(model::SchurOptimizer, src::BOI.ModelLike)
#     return MOI.Utilities.default_copy_to(model, src)
# end