### GraphIndex

struct GraphIndex
    value::Int64
end

function raw_index(index::GraphIndex)
    return index.value
end

struct NodeIndex
    value::Int64
end

function raw_index(index::GraphIndex)
    return index.value
end

struct EdgeIndex
    value::Int64
end

function raw_index(index::GraphIndex)
    return index.value
end

"""
    Node

A node represents a set of variables and associated attributes. 
"""
struct Node
    graph::OptiGraph
    index::NodeIndex
    moi_model::MOIU.UniversalFallback{MOIU.Model{Float64}}
    nonlinear_model::Union{Nothing,MOI.Nonlinear.Model}
end
function Node(graph::OptiGraph, index::HyperNode)
    moi_model = MOIU.UniversalFallback(MOIU.Model{Float64}())
    return Node(
        graph,
        graph_index,
        moi_model,
        nothing
    )
end

function node_index(node::Node)::HyperNode
    return node.index
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

# forward methods so Nodes call their underlying MOI models

@forward Node.moi_model (MOI.get, MOI.set)


### Edges

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
    graph::OptiGraph
    index::EdgeIndex
    nodes::OrderedSet{Node}
    index_map::EdgeIndexMap
    moi_model::MOIU.UniversalFallback{MOIU.Model{Float64}}
    nonlinear_model::Union{Nothing,MOI.Nonlinear.Model}
end

function Edge(
    graph::OptiGraph,
    index::HyperEdge,
    nodes::NTuple
) where T <: AbstractNodeVariableMap
    moi_model = MOIU.UniversalFallback(MOIU.Model{Float64}())
    return Edge(graph, index, nodes, EdgeIndexMap(), moi_model, nothing)
end

function edge_index(edge::Edge)::HyperEdge
    return edge.index
end

function node_variables(edge::Edge)::Tuple
    return collect(values(edge.index_map.edge_to_node_map))
end

### MOI Edge Functions

# forward methods so Edges call their underlying MOI models

@forward Edge.moi_model (MOI.get, MOI.set)

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

### OptiGraph

mutable struct OptiGraph
    index::Int64
    parent::Union{Nothing,OptiGraph}
    nodes::Vector{Node}
    edges::Vector{Edge}
    subgraphs::Vector{OptiGraph}

    # lookup elements by their indices
    node_by_index::OrderedDict{HyperNode,Node}
    edge_by_index::OrderedDict{HyperEdge,Edge}
    subgraph_by_index::OrderedDict{GraphIndex,OptiGraph}
    function OptiGraph()
        graph = new()
        graph.index = GraphIndex(0)
        graph.parent = nothing
        graph.hypergraph = HyperGraph()
        graph.nodes = Node[]
        graph.edges = Edge[]
        graph.subgraphs = Vector{OptiGraph}()
        graph.node_by_index = OrderedDict{HyperNode,Node}()
        graph.edge_by_index = OrderedDict{HyperEdge,Edge}()
        graph.subgraph_by_index = OrderedDict{GraphIndex,OptiGraph}()
        return graph
    end
end

# nodes connected to edge
function connected_nodes(graph::OptiGraph, edge::Edge)
    vertices = edge.vertices
    return getindex.(Ref(graph.nodes_by_index), vertices)
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

function num_subgraphs(graph::OptiGraph)
    return length(graph.subgraphs)
end

function num_all_subgraphs(graph::OptiGraph)
    return length(graph.subgraph_by_index)
end

"""
    add_node(graph::OptiGraph)::Node

Add a node to `graph`. The index of the node is determined by the central graph.
"""
function add_node(graph::OptiGraph)::Node
    #hypernode = Graphs.add_vertex!(graph.hypergraph)
    node_index = num_nodes(graph) + 1
    node = Node(graph, node_index)
    push!(graph.nodes, node)
    return node
end

"""
    add_edge(graph::OptiGraph, nodes::NTuple{N,Node})::Edge where N

Add an edge to `graph`.
"""
function add_edge(graph::OptiGraph, nodes::NTuple{N,Node})::Edge where N
    # we do not allow arbitrary edges. at most between two layers.
    @assert isempty(setdiff(nodes, get_nodes_to_depth(graph, 1)))
    #hypernodes = node_index.(nodes)
    #hyperedge = Graphs.add_edge!(graph.hypergraph, hypernodes...)
    edge_index = EdgeIndex(num_edges(graph) + 1)
    edge = Edge(graph, edge_index, nodes)
    graph.edge_by_index[edge_index] = edge
    push!(graph.edges, edge)
    return edge
end

"""
    add_subgraph(graph::OptiGraph)

Add a new subgraph to `graph`.
"""
function add_subgraph(graph::OptiGraph)
    subgraph = OptiGraph()
    push!(graph.subgraphs, subgraph)
    graph.graph_by_index[subgraph.index] = subgraph
    subgraph.parent = graph
    subgraph.index = GraphIndex(number_of_subgraphs(graph))
    return subgraph

    # TODO: update parent graphs
end

function get_nodes(graph::OptiGraph)
    return graph.nodes
end

function get_nodes_to_depth(block::Block, n_layers::Int=0)
    nodes = block.nodes
    if n_layers > 0
        for sub_block in block.sub_blocks
            inner_nodes = get_nodes_to_depth(sub_block, n_layers-1)
            nodes = [nodes; inner_nodes]
        end
    end
    return nodes
end

function get_edges(block::Block)
    return block.edges
end

function all_nodes(block::Block)
    nodes = block.nodes
    if !isempty(block.sub_blocks)
        for sub_block in block.sub_blocks
            append!(nodes, all_nodes(sub_block))
        end
    end
    return nodes
end

