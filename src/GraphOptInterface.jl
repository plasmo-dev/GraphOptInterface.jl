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

"""
    supports_graph_interface(optimizer::MOI.AbstractOptimizer)

Check if the given optimizer supports the graph interface. Returns `false` for standard 
MOI optimizers and `true` for `AbstractGraphOptimizer` subtypes.
"""
function supports_graph_interface(::MOI.AbstractOptimizer)
    return false
end

function supports_graph_interface(::AbstractGraphOptimizer)
    return true
end

include("GraphViews/hypergraph.jl")

include("GraphViews/bipartite.jl")

include("optigraph.jl")

include("hypergraph_interface.jl")

end
