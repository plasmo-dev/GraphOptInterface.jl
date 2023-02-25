# Optimizers that inherit from AbstractBlockOptimizer use `Block` in addition to other MOI attributes.

"""
    AbstractBlockOptimizer
Abstract supertype for block-structure-exploiting optimizers.

e.g. create a Schur optimizer
mutable struct SchurOptimizer <: AbstractBlockOptimizer end
"""
abstract type AbstractBlockOptimizer <: MOI.AbstractOptimizer end


"""
    AbstractBlockOptimizerAttribute
Abstract supertype for attribute objects that can be used to set or get attributes (properties) of the block-structure-exploiting optimizer.
"""
abstract type AbstractBlockAttribute <: MOI.AbstractModelAttribute end

# Block optimizer interface
function supports_block_interface(::MOI.AbstractOptimizer)
    return false
end

"""
    BlockStructure()
An [`AbstractModelAttribute`](@ref) that stores a [`GraphBlockData`](@ref),
representing structural model information.

Example:
`MOI.set(optimizer, GraphBlock(), create_graph_block_data(graph::Plasmo.OptiGraph))`
"""
# struct BlockStructure <: AbstractBlockAttribute end

