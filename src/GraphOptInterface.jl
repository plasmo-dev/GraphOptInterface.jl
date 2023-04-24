module GraphOptInterface

using MathOptInterface
const MOI = MathOptInterface
const MOIU = MOI.Utilities

using DataStructures
using SparseArrays
using Lazy
using Graphs

"""
    AbstractBlockOptimizer
Abstract supertype for block-structure-exploiting optimizers.
"""
abstract type AbstractGraphOptimizer <: MOI.AbstractOptimizer end

struct GraphStructure <: MOI.AbstractModelAttribute end

# Block optimizer interface
function supports_graph_interface(::MOI.AbstractOptimizer)
    return false
end

function supports_graph_interface(::AbstractGraphOptimizer)
    return true
end

include("hypergraph.jl")

include("graph.jl")

include("graph_functions.jl")

end 
