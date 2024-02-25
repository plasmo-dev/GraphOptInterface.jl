"""
	HyperMap

A mapping from an OptiGraph to a graph view that supports various graph query functions.
Currently supports a `HyperGraph` as the graph view.
"""
struct HyperMap
	optigraph::OptiGraph
	hypergraph::HyperGraph
	hypernode_to_node_map::OrderedDict{HyperNode,Node}
	hyperedge_to_edge_map::OrderedDict{HyperEdge,Edge}
	node_to_hypernode_map::OrderedDict{Node,HyperNode}
	edge_to_hyperedge_map::OrderedDict{Edge,HyperEdge}
end
function HyperMap(graph::OptiGraph, hypergraph::HyperGraph)
	return HyperMap(
		graph,
		hypergraph,
		OrderedDict{HyperNode,Node}(),
		OrderedDict{HyperEdge,Edge}(),
		OrderedDict{Node,HyperNode}(),
		OrderedDict{Edge,HyperEdge}()
	)
end

function Base.getindex(hyper_map::HyperMap, vertex::HyperNode)
    return hyper_map.hypernode_to_node_map[vertex]
end

function Base.setindex!(hyper_map::HyperMap, vertex::HyperNode, node::Node)
    return hyper_map.hypernode_to_node_map[vertex] = node
end

function Base.getindex(hyper_map::HyperMap, node::Node)
    return hyper_map.node_to_hypernode_map[node]
end

function Base.setindex!(hyper_map::HyperMap, node::Node, vertex::HyperNode)
    return hyper_map.node_to_hypernode_map[node] = vertex
end

function Base.getindex(hyper_map::HyperMap, hyper_edge::HyperEdge)
    return hyper_map.hyperedge_to_edge_map[hyper_edge]
end

function Base.setindex!(hyper_map::HyperMap, hyper_edge::HyperEdge, edge::Edge)
    return hyper_map.hyperedge_to_edge_map[edge] = hyperedge
end

function Base.getindex(hyper_map::HyperMap, edge::Edge)
    return hyper_map.edge_to_hyperedge_map[edge]
end

function Base.setindex!(hyper_map::HyperMap, edge::Edge, hyper_edge::HyperEdge)
    return hyper_map.edge_to_hyperedge_map[edge] = hyper_edge
end

function build_hypergraph_view(graph::OptiGraph)
	hypergraph = HyperGraph()
	hyper_map = HyperMap(graph, hypergraph)
	for node in all_nodes(graph)
		hypernode = Graphs.add_vertex!(hypergraph)
		hyper_map[hypernode] = node 
		hyper_map[node] = hypernode
	end
	for edge in all_edges(graph)
        nodes = get_nodes(edge)
        hypernodes = [hyper_map.node_to_hypernode_map[node] for node in nodes]
        @assert length(hypernodes) >= 2
        hyperedge = Graphs.add_edge!(hypergraph, hypernodes...)
        hyper_map[hyperedge] = edge
        hyper_map[edge] = hyperedge
    end
    return hyper_map
end

"""
	get_mapped_nodes(hyper_map::HyperMap, nodes::Vector{Node})

Get the hypernode elements that correspond to the supplied optigraph `nodes`.
"""
function get_mapped_nodes(hyper_map::HyperMap, nodes::Vector{Node})
	return getindex.(Ref(hyper_map.node_to_hypernode_map), nodes)
end

function get_mapped_nodes(hyper_map::HyperMap, nodes::Vector{HyperNode})
	return getindex.(Ref(hyper_map.hypernode_to_node_map), nodes)
end

function get_mapped_edges(hyper_map::HyperMap, edges::Vector{Edge})
	return getindex.(Ref(hyper_map.node_to_hypernode_map), edges)
end

function get_mapped_edges(hyper_map::HyperMap, edges::Vector{HyperEdge})
	return getindex.(Ref(hyper_map.hyperedge_to_edge_map), edges)
end

### Neighbors

"""
	all_neighbors(hyper_map::HyperMap, nodes::Vector{Node})::Vector{Node}

Return the neighbor nodes within the optigraph in `hyper_map` to the vector of supplied 
optigraph nodes.  
"""
function Graphs.all_neighbors(hyper_map::HyperMap, nodes::Vector{Node})::Vector{Node}
	vertices = get_mapped_nodes(nodes)
	neighbor_vertices = Graphs.all_neighbors(hyper_map.hypergraph, vertices...)
	return get_mapped_nodes(hyper_map, vertices)
end

"""
	parent_neighbors(hyper_map::HyperMap, subgraph::OptiGraph)::Vector{Node}

Return the neighbor nodes in `subgraph` based on the optigraph in `hyper_map` that are only
in the parent graph of `subgraph. 
"""
function parent_neighbors(hyper_map::HyperMap, subgraph::OptiGraph)::Vector{Node}
	@assert subgraph in all_subgraphs(hyper_map.optigraph)
	parent_graph = subgraph.parent
	neighbors = all_neighbors(hyper_map, all_nodes(subgraph))
	parent_nodes = intersect(parent_graph.nodes, neighbors)
	return parent_nodes
end

"""
	non_parent_neighbors(hyper_map::HyperMap, subgraph::OptiGraph)::Vector{Node}

Return the neighbor nodes in `subgraph` based on the optigraph in `hyper_map` that are not 
in the parent graph of `subgraph. 
"""
function non_parent_neighbors(hyper_map::HyperMap, subgraph::OptiGraph)::Vector{Node}
	@assert subgraph in all_subgraphs(hyper_map.optigraph)
	parent_graph = subgraph.parent
	neighbors = all_neighbors(hyper_map, all_nodes(subgraph))
	neighbor_nodes = setdiff(neighbors, parent_graph.nodes)
	return neighbor_nodes
end

### Incident Edges

"""
	all_incident_edges(hyper_map::HyperMap, nodes::Vector{Node})::Vector{Edge}

Return all of the edges within the optigraph in `hyper_map` that are incident 
to the vector of supplied `nodes`.
"""
function all_incident_edges(hyper_map::HyperMap, nodes::Vector{Node})::Vector{Edge}
	vertices = get_mapped_nodes(nodes)
	hyperedges = incident_edges(hyper_map.optigraph, vertices)
	return get_mapped_edges(hyper_map, hyperedges)
end

"""
	parent_incident_edges(hyper_map::HyperMap, subgraph::OptiGraph)::Vector{Edge}

Return all of the optigraph edges that are incident to the supplied `subgraph` that 
are strictly parent connections.
"""
function parent_incident_edges(hyper_map::HyperMap, subgraph::OptiGraph)::Vector{Edge}
	@assert subgraph in all_subgraphs(hyper_map.optigraph)
	parent_graph = subgraph.parent
	incident_edges = all_incident_edges(hyper_map, all_edges(subgraph))
	parent_edges = intersect(parent_graph.edges, incident_edges)
	return parent_edges
end

"""
	non_parent_incident_edges(hyper_map::HyperMap, subgraph::OptiGraph)::Vector{Edge}

Return all of the optigraph edges that are incident to the supplied `subgraph` that 
are strictly not parent connections.
"""
function non_parent_incident_edges(hyper_map::HyperMap, subgraph::OptiGraph)::Vector{Edge}
	@assert subgraph in all_subgraphs(hyper_map.optigraph)
	parent_graph = subgraph.parent
	incident_edges = all_incident_edges(hyper_map, all_edges(subgraph))
	external_edges = setdiff(incident_edges, parent_graph.edges)
	return external_edges
end

"""
	children_incident_edges(hyper_map::HyperMap, graph::OptiGraph)::Vector{Edge}

Return all of the optigraph edges that are incident to the supplied `graph` that are 
strictly child connections.
"""
function children_incident_edges(hyper_map::HyperMap, graph::OptiGraph)::Vector{Edge}
	children_edges = []
	for subgraph in graph.subgraphs
		append!(children_edges, parent_incident_edges(subgraph))
	end
	return children_edges
end