module GraphOptInterface

using MathOptInterface
const MOI = MathOptInterface
const MOIU = MOI.Utilities

using DataStructures
using SparseArrays
using Lazy
using Graphs

"""
    AbstractGraphOptimizer
Abstract supertype for block-structure-exploiting optimizers.
"""
abstract type AbstractGraphOptimizer <: MOI.AbstractOptimizer end

# Block optimizer interface
function supports_graph_interface(::MOI.AbstractOptimizer)
    return false
end

function supports_graph_interface(::AbstractGraphOptimizer)
    return true
end

include("GraphViews/hypergraph.jl")

include("GraphViews/bipartite.jl")

include("optigraph.jl")

include("graph_functions.jl")

end 
