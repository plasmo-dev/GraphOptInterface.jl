### Neighbors

# every neighbor including parent and child neighbors
function all_neighbors(graph::OptiGraph, nodes::Vector{Node})::Vector{Node}
	root = MOI.get(optimizer, BlockStructure())
	vertices = index_value.(node_index.(nodes))
	neighbor_vertices = Graphs.all_neighbors(root.graph, vertices...)
	return_indices = NodeIndex.(neighbor_vertices)
	return_nodes = getindex.(Ref(root.node_by_index), return_indices)
	return return_nodes
end

function all_neighbors(graph::Graph, block::Block)::Vector{Node}
	return all_neighbors(optimizer, block.nodes)
end

# neighbors in parent block
function parent_neighbors(graph::Graph, block::Block)::Vector{Node}
	root = MOI.get(optimizer, BlockStructure())
	parent_block = root.block_by_index[block.parent_index]
	neighbors = all_neighbors(optimizer, block)
	parent_nodes = intersect(parent_block.nodes, neighbors)
	return parent_nodes
end

function neighbors(graph::Graph, block::Block)::Vector{Node}
	root = MOI.get(optimizer, BlockStructure())
	parent_block = root.block_by_index[block.parent_index]
	neighbors = all_neighbors(optimizer, block)
	neighbor_nodes = setdiff(neighbors, parent_block.nodes)
	return neighbor_nodes
end

### Incident Edges

# all edges incident to a block
function all_incident_edges(graph::Graph, block::Block)::Vector{Edge}
	root = MOI.get(optimizer, BlockStructure())
	node_indices = index_value.(node_index.((all_nodes(block))))
	hyperedges = incident_edges(root.graph, node_indices)

	# TODO: make cleaner hypergraph implementation
	edge_indices = EdgeIndex.([root.graph.hyperedges[h.vertices] for h in hyperedges])
	edges = getindex.(Ref(root.edge_by_index), edge_indices)
	return edges
end

# edges that connect this block to a parent block
function parent_incident_edges(graph::Graph, block::Block)::Vector{Edge}
	root = MOI.get(optimizer, BlockStructure())
	parent_block = root.block_by_index[block.parent_index]
	inc_edges = all_incident_edges(optimizer, block)
	parent_edges = intersect(parent_block.edges, inc_edges)
	return parent_edges
end

# edges that connect this block to other blocks
function incident_edges(graph::Graph, block::Block)::Vector{Edge}
	root = MOI.get(optimizer, BlockStructure())
	parent_block = root.block_by_index[block.parent_index]
	inc_edges = all_incident_edges(optimizer, block)
	external_edges = filter((edge) -> any(node -> node in node_index.(parent_block.nodes), edge.elements), inc_edges)
	return external_edges
end

# edges that connect this block to children blocks
# TODO: we might just label these edges with direction for easy look-up
function children_incident_edges(graph::Graph, block::Block)::Vector{Edge}
	children_edges = []
	for sub_block in block.sub_blocks
		append!(children_edges, parent_incident_edges(sub_block))
	end
	return children_edges
end