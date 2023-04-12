module BlockOptInterface

using MathOptInterface
using DataStructures
using SparseArrays
using Lazy
using Graphs

const MOI = MathOptInterface
const MOIU = MOI.Utilities
const MOIB = MOI.Bridges

"""
    AbstractBlockOptimizer
Abstract supertype for block-structure-exploiting optimizers.
"""
abstract type AbstractBlockOptimizer <: MOI.AbstractOptimizer end

abstract type AbstractBlockAttribute <: MOI.AbstractModelAttribute end

struct BlockStructure <: AbstractBlockAttribute end

# Block optimizer interface
function supports_block_interface(::MOI.AbstractOptimizer)
    return false
end

function supports_block_interface(::AbstractBlockOptimizer)
    return true
end

include("hypergraph.jl")

include("block.jl")

include("block_nlp_evaluator.jl")

end 
