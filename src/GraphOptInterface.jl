module GraphOptInterface

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
abstract type AbstractGraphOptimizer <: MOI.AbstractOptimizer end

abstract type AbstractGraphAttribute <: MOI.AbstractModelAttribute end

struct GraphInterface <: AbstractBlockAttribute end

# Block optimizer interface
function supports_graph_interface(::MOI.AbstractOptimizer)
    return false
end

function supports_graph_interface(::AbstractBlockOptimizer)
    return true
end

include("hypergraph.jl")

include("block.jl")

include("block_nlp_evaluator.jl")

end 
