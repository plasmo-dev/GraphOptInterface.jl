struct GraphIndex
    value::Symbol
end

### Node

struct NodeIndex
    value::Int64
end

"""
    Node

A node represents a set of variables and associated attributes. 
"""
mutable struct Node
    graph_index::GraphIndex
    index::NodeIndex
    moi_model::MOIU.UniversalFallback{MOIU.Model{Float64}}
    nonlinear_model::Union{Nothing,MOI.Nonlinear.Model}
end
function Node(graph_index::GraphIndex, node_index::NodeIndex)
    moi_model = MOIU.UniversalFallback(MOIU.Model{Float64}())
    return Node(
        graph_index,
        node_index,
        moi_model,
        nothing
    )
end

function node_index(node::Node)::NodeIndex
    return node.index
end

function raw_index(node::Node)
    return node.index.value
end

### MOI Node Functions

function MOI.add_variable(node::Node)
    var = MOI.add_variable(node.moi_model)
    return var
end

function MOI.add_variables(node::Node, n::Int64)
    vars = MOI.VariableIndex[]
    for _ = 1:n
        push!(vars, MOI.add_variable(node))
    end
    return vars
end

function MOI.add_constraint(
    node::Node,
    func::F,
    set::S
) where {F <: MOI.AbstractFunction, S <: MOI.AbstractSet}
    ci = MOI.add_constraint(node.moi_model, func, set)
    return ci
end

function MOI.Nonlinear.add_constraint(
    node::Node,
    expr::Expr,
    set::S
) where {S <: MOI.AbstractSet}
    node.nonlinear_model === nothing && (node.nonlinear_model = MOI.Nonlinear.Model())
    constraint_index = MOI.Nonlinear.add_constraint(node.nonlinear_model, expr, set)
    return constraint_index
end

# forward methods so Nodes call their underlying MOI models

@forward Node.moi_model (MOI.get, MOI.set)

### Edges

struct EdgeIndex
    value::Int64
end

struct EdgeIndexMap
    edge_to_node_map::OrderedDict{MOI.VariableIndex,Tuple{Node,MOI.VariableIndex}}
    node_to_edge_map::OrderedDict{Tuple{Node,MOI.VariableIndex},MOI.VariableIndex}
end
function EdgeIndexMap()
    edge_to_node_map = OrderedDict{MOI.VariableIndex,Tuple{Node,MOI.VariableIndex}}()
    node_to_edge_map = OrderedDict{Tuple{Node,MOI.VariableIndex},MOI.VariableIndex}()
    return EdgeIndexMap(edge_to_node_map, node_to_edge_map)
end

"""
    Edge

An edge represents different types of coupling. For instance an Edge{Tuple{Node}} is an edge
the couple variables within a single node. An Edge{Tuple{N,Node}} couple variables across
one or more nodes.
"""
mutable struct Edge
    graph_index::GraphIndex
    index::EdgeIndex
    nodes::OrderedSet{Node}
    index_map::EdgeIndexMap
    moi_model::MOIU.UniversalFallback{MOIU.Model{Float64}}
    nonlinear_model::Union{Nothing,MOI.Nonlinear.Model}
end
function Edge(
    graph_index::GraphIndex, 
    edge_index::EdgeIndex, 
    nodes::NTuple{N,Node} where N
)
    moi_model = MOIU.UniversalFallback(MOIU.Model{Float64}())
    return Edge(
        graph_index, 
        edge_index, 
        OrderedSet(nodes), 
        EdgeIndexMap(), 
        moi_model, 
        nothing
    )
end

function edge_index(edge::Edge)::EdgeIndex
    return edge.index
end

function raw_index(edge::Edge)
    return edge.index.value
end

"""
    node_variables(edge::Edge)::Vector{Tuple{Node,MOI.VariableIndex}}

Return a vector of tuples where each tuple contains the node and variable index
associated with each edge variable.
"""
function node_variables(edge::Edge)::Vector{Tuple{Node,MOI.VariableIndex}}
    return collect(values(edge.index_map.edge_to_node_map))
end

### MOI Edge Functions

function MOI.add_variable(edge::Edge, node::Node, variable::MOI.VariableIndex)
    edge_variable = MOI.add_variable(edge.moi_model)
    edge.index_map.edge_to_node_map[edge_variable] = (node,variable)
    edge.index_map.node_to_edge_map[(node,variable)] = edge_variable
    return edge_variable
end

function MOI.add_constraint(
    edge::Edge,
    func::F,
    set::S
) where {F <: MOI.AbstractFunction, S <: MOI.AbstractSet}
    # TODO: check that the variables exist on the edge index map
    ci = MOI.add_constraint(edge.moi_model, func, set)
    return ci
end

function MOI.Nonlinear.add_constraint(
    edge::Edge,
    expr::Expr,
    set::S
) where {S <: MOI.AbstractSet}
    edge.nonlinear_model === nothing && (edge.nonlinear_model = MOI.Nonlinear.Model())
    constraint_index = MOI.Nonlinear.add_constraint(edge.nonlinear_model, expr, set)
    return constraint_index
end

function get_nodes(edge::Edge)
    return collect(edge.nodes)
end

function get_edge_variable(edge::Edge, node::Node, variable::MOI.VariableIndex)
    return edge.index_map.node_to_edge_map[(node,variable)]
end

function get_node_variable(edge::Edge, variable::MOI.VariableIndex)
    return edge.index_map.edge_to_node_map[variable]
end

# forward methods so Edges call their underlying MOI models

@forward Edge.moi_model (MOI.get, MOI.set)

### OptiGraph

mutable struct OptiGraph
    index::GraphIndex
    parent::Union{Nothing,OptiGraph}
    nodes::Vector{Node}
    edges::Vector{Edge}
    subgraphs::Vector{OptiGraph}

    # lookup elements by their indices
    node_by_index::OrderedDict{NodeIndex,Node}
    edge_by_index::OrderedDict{EdgeIndex,Edge}
    subgraph_by_index::OrderedDict{GraphIndex,OptiGraph}
    function OptiGraph()
        graph = new()
        graph.index = GraphIndex(gensym())
        graph.parent = nothing
        graph.nodes = Node[]
        graph.edges = Edge[]
        graph.subgraphs = Vector{OptiGraph}()
        graph.node_by_index = OrderedDict{NodeIndex,Node}()
        graph.edge_by_index = OrderedDict{EdgeIndex,Edge}()
        graph.subgraph_by_index = OrderedDict{GraphIndex,OptiGraph}()
        return graph
    end
end

function Base.string(graph::OptiGraph)
    return """Graph
    $(length(graph.nodes)) nodes
    $(length(graph.edges)) edges
    $(length(graph.subgraphs)) subgraphs
    """
end
Base.print(io::IO, graph::OptiGraph) = Base.print(io, Base.string(graph))
Base.show(io::IO, graph::OptiGraph) = Base.print(io, graph)

function graph_index(graph::OptiGraph)
    return graph.index
end

function raw_index(graph::OptiGraph)
    return graph.index.value
end

function num_nodes(graph::OptiGraph)
    return length(graph.nodes)
end

function num_all_nodes(graph::OptiGraph)
    num = num_nodes(graph)
    for subgraph in get_subgraphs(graph)
        num += num_nodes(subgraph)
    end
    return num
end

function all_nodes(graph::OptiGraph)
    nodes = Node[]
    append!(nodes, graph.nodes)
    if !isempty(graph.subgraphs)
        for subgraph in graph.subgraphs
            append!(nodes, all_nodes(subgraph))
        end
    end
    return nodes
end

function num_edges(graph::OptiGraph)
    return length(graph.edges)
end

### Graph MOI Functions

function MOI.get(graph::OptiGraph, attr::MOI.ListOfConstraintTypesPresent)
    ret = []
    for node in all_nodes(graph)
        append!(ret, MOI.get(node.model, attr))
    end
    for edge in all_edges(graph)
        append!(ret, MOI.get(edge.model, attr))
    end
    return unique(ret)
end

function MOI.get(graph::OptiGraph, attr::MOI.NumberOfVariables)
    return sum(MOI.get(node, attr) for node in all_nodes(graph))
end

function MOI.get(graph::OptiGraph, attr::MOI.ListOfVariableIndices)
    return vcat(MOI.get(node, attr) for node in all_nodes(graph))
end

function MOI.get(
    graph::OptiGraph,
    attr::MOI.NumberOfConstraints{F,S}
) where {F <: MOI.AbstractFunction, S <: MOI.AbstractSet}
    return sum(MOI.get(edge, attr) for edge in all_edges(graph)) + 
        sum(MOI.get(node, attr) for node in all_nodes(graph))
end

### OptiGraph functions

"""
    add_node(graph::OptiGraph)::Node

Add a node to `graph`. The index of the node is determined by the central graph.
"""
function add_node(graph::OptiGraph)::Node
    node_index = NodeIndex(num_nodes(graph) + 1)
    node = Node(graph.index, node_index)
    push!(graph.nodes, node)
    return node
end

function get_nodes(graph::OptiGraph)
    return graph.nodes
end

function get_nodes_to_depth(graph::OptiGraph, depth::Int=0)
    nodes = graph.nodes
    if depth > 0
        for subgraph in graph.subgraphs
            inner_nodes = get_nodes_to_depth(subgraph, depth-1)
            nodes = [nodes; inner_nodes]
        end
    end
    return nodes
end

"""
    add_edge(graph::OptiGraph, nodes::NTuple{N,Node})::Edge where N

Add an edge to `graph`.
"""
function add_edge(graph::OptiGraph, nodes::NTuple{N,Node})::Edge where N
    # we do not allow arbitrary edges. at most between two layers.
    @assert isempty(setdiff(nodes, get_nodes_to_depth(graph, 1)))
    edge_index = EdgeIndex(num_edges(graph) + 1)
    edge = Edge(graph.index, edge_index, nodes)
    graph.edge_by_index[edge_index] = edge
    push!(graph.edges, edge)
    return edge
end

function get_edges(graph::OptiGraph)
    return graph.edges
end

function all_edges(graph::OptiGraph)
    edges = graph.edges
    if !isempty(graph.edges)
        for subgraph in graph.subgraphs
            append!(edges, all_edges(subgraph))
        end
    end
    return edges
end

"""
    add_subgraph(graph::OptiGraph)

Add a new subgraph to `graph`.
"""
function add_subgraph(graph::OptiGraph)
    subgraph = OptiGraph()
    push!(graph.subgraphs, subgraph)
    graph.subgraph_by_index[subgraph.index] = subgraph
    subgraph.parent = graph
    return subgraph
end

function get_subgraphs(graph::OptiGraph)
    return graph.subgraphs
end

function num_subgraphs(graph::OptiGraph)
    return length(graph.subgraphs)
end

function num_all_subgraphs(graph::OptiGraph)
    num = num_subgraphs(graph)
    for subgraph in get_subgraphs(graph)
        num += num_subgraphs(subgraph)
    end
    return num
end