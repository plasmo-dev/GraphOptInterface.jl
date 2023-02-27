# Optimizers that inherit from AbstractBlockOptimizer use `Block` in addition to other MOI attributes.



"""
    AbstractBlockOptimizerAttribute
Abstract supertype for attribute objects that can be used to set or get attributes (properties) of the block-structure-exploiting optimizer.
"""
# abstract type AbstractBlockAttribute <: MOI.AbstractModelAttribute end



"""
    BlockStructure()
An [`AbstractModelAttribute`](@ref) that stores a [`GraphBlockData`](@ref),
representing structural model information.

Example:
`MOI.set(optimizer, GraphBlock(), create_graph_block_data(graph::Plasmo.OptiGraph))`
"""
# struct BlockStructure <: AbstractBlockAttribute end

