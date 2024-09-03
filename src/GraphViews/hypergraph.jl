const HyperNode = Int64

struct HyperEdge <: Graphs.AbstractEdge{Int64}
    vertices::Set{HyperNode}
end

HyperEdge(t::Vector{HyperNode}) = HyperEdge(Set(t))

HyperEdge(t::HyperNode...) = HyperEdge(Set(collect(t)))

"""
    HyperGraph

A very simple hypergraph type.  Contains attributes for vertices and hyperedges.
"""
mutable struct HyperGraph <: Graphs.AbstractGraph{Int64}
    vertices::OrderedSet{HyperNode}
    hyperedge_map::OrderedDict{Int64,HyperEdge}
    hyperedges::OrderedDict{Set{HyperNode},Int64}
    node_map::OrderedDict{HyperNode,Vector{HyperEdge}}
end
function HyperGraph()
    return HyperGraph(
        OrderedSet{HyperNode}(),
        OrderedDict{Int64,HyperEdge}(),
        OrderedDict{Set{HyperNode},HyperEdge}(),
        OrderedDict{HyperNode,Vector{HyperEdge}}(),
    )
end

function Base.getindex(hypergraph::HyperGraph, node::HyperNode)
    return node
end

function Base.getindex(hypergraph::HyperGraph, edge::HyperEdge)
    return hypergraph.hyperedges[edge.vertices]
end

Base.reverse(e::HyperEdge) = error("`HyperEdge` does not support reverse.")

function Base.:(==)(h1::HyperEdge, h2::HyperEdge)
    return collect(h1.vertices) == collect(h2.vertices)
end

function get_hyperedge(hypergraph::HyperGraph, edge_index::Int64)
    return hypergraph.hyperedge_map[edge_index]
end

function get_hyperedge(hypergraph::HyperGraph, hypernodes::Set)
    edge_index = hypergraph.hyperedges[hypernodes]
    return hypergraph.hyperedge_map[edge_index]
end

function Graphs.add_vertex!(hypergraph::HyperGraph)
    # test for overflow
    (Graphs.nv(hypergraph) + one(Int) <= Graphs.nv(hypergraph)) && return false
    v = length(hypergraph.vertices) + 1
    hypernode = v
    push!(hypergraph.vertices, hypernode)
    hypergraph.node_map[hypernode] = HyperEdge[]
    return hypernode
end

function Graphs.add_edge!(hypergraph::HyperGraph, hypernodes::HyperNode...)
    @assert length(hypernodes) > 1
    hypernodes = Set(collect(hypernodes))
    if Graphs.has_edge(hypergraph, hypernodes)
        return get_hyperedge(hypergraph, hypernodes)
        #return hypergraph.hyperedges[hypernodes]
    else
        index = Graphs.ne(hypergraph) + 1
        hyperedge = HyperEdge(hypernodes...)
        for hypernode in hypernodes
            push!(hypergraph.node_map[hypernode], hyperedge)
        end
        hypergraph.hyperedges[hypernodes] = index
        hypergraph.hyperedge_map[index] = hyperedge
        return hyperedge
    end
end

Graphs.edges(hypergraph::HyperGraph) = values(hypergraph.hyperedge_map)

Graphs.edgetype(graph::HyperGraph) = HyperEdge

function Graphs.has_edge(graph::HyperGraph, edge::HyperEdge)
    return edge in values(graph.hyperedge_map)
end

function Graphs.has_edge(graph::HyperGraph, hypernodes::Set{HyperNode})
    return haskey(graph.hyperedges, hypernodes)
end

Graphs.has_vertex(graph::HyperGraph, v::Integer) = v in Graphs.vertices(graph)

Graphs.is_directed(graph::HyperGraph) = false

Graphs.is_directed(::Type{HyperGraph}) = false

Graphs.ne(graph::HyperGraph) = length(graph.hyperedge_map)

Graphs.nv(graph::HyperGraph) = length(graph.vertices)

Graphs.vertices(graph::HyperGraph) = collect(graph.vertices)

Graphs.vertices(hyperedge::HyperEdge) = collect(hyperedge.vertices)

Graphs.degree(g::HyperGraph, v::Int) = length(Graphs.all_neighbors(g, v))

function Graphs.all_neighbors(g::HyperGraph, node::HyperNode)
    hyperedges = g.node_map[node]  #incident hyperedges to the hypernode
    neighbors = HyperNode[]
    for edge in hyperedges
        append!(neighbors, [vert for vert in edge.vertices if vert != node])
    end
    return unique(neighbors)
end

"""
    Graphs.incidence_matrix(hypergraph::HyperGraph)

Obtain the incidence matrix representation of `hypergraph`.  Rows correspond to vertices. Columns correspond to hyperedges.
Returns a sparse matrix.
"""
function Graphs.incidence_matrix(hypergraph::HyperGraph)
    I = []
    J = []
    for (edge_index, hyperedge) in hypergraph.hyperedge_map
        node_indices = sort(collect(hyperedge.vertices))
        for node_index in node_indices
            push!(I, node_index)
            push!(J, edge_index)
        end
    end
    V = Int.(ones(length(I)))
    m = length(hypergraph.vertices)
    n = length(hypergraph.hyperedge_map)
    return SparseArrays.sparse(I, J, V, m, n)
end

SparseArrays.sparse(hypergraph::HyperGraph) = Graphs.incidence_matrix(hypergraph)

"""
    Graphs.adjacency_matrix(hypergraph::HyperGraph)

Obtain the adjacency matrix from `hypergraph.` Returns a sparse matrix.
"""
function Graphs.adjacency_matrix(hypergraph::HyperGraph)
    I = []
    J = []
    for vertex in Graphs.vertices(hypergraph)
        for neighbor in Graphs.all_neighbors(hypergraph, vertex)
            push!(I, vertex)
            push!(J, neighbor)
        end
    end
    V = Int.(ones(length(I)))
    return SparseArrays.sparse(I, J, V)
end

"""
    incident_edges(hypergraph::HyperGraph,hypernode::HyperNode)

Identify the incident hyperedges to a `HyperNode`.
"""
function incident_edges(hypergraph::HyperGraph, node::HyperNode)
    return hypergraph.node_map[node]
end

"""
    induced_edges(hypergraph::HyperGraph,hypernodes::Vector{HyperNode})

Identify the induced hyperedges to a vector of `HyperNode`s.

NOTE: This currently does not support hypergraphs with unconnected nodes
"""
function induced_edges(hypergraph::HyperGraph, hypernodes::Vector{HyperNode})
    # get vertices in hypergraph that are not in hypernodes
    external_nodes = setdiff(hypergraph.vertices, hypernodes)

    #Create partition matrix
    I = []
    J = []
    for hypernode in hypernodes
        j = getindex(hypergraph, hypernode)
        push!(I, 1)
        push!(J, j)
    end
    for hypernode in external_nodes
        j = getindex(hypergraph, hypernode)
        push!(I, 2)
        push!(J, j)
    end
    V = Int.(ones(length(J)))
    G = sparse(I, J, V)  #Node partition matrix
    A = sparse(hypergraph)
    C = G * A  #Edge partitions

    # find shared edges; get indices of shared edges
    sum_vector = sum(C; dims=1)
    max_vector = maximum(C; dims=1)
    cross_vector = sum_vector - max_vector

    # nonzero indices of the cross vector; these are edges that cross partitions.
    indices = findall(cross_vector .!= 0)
    indices = [indices[i].I[2] for i in 1:length(indices)]

    inds = findall(C[1, :] .!= 0)
    new_inds = filter(x -> !(x in indices), inds) #these are edge indices
    induced_edges = HyperEdge[get_hyperedge(hypergraph, new_ind) for new_ind in new_inds]

    return induced_edges
end

"""
    incident_edges(hypergraph::HyperGraph,hypernodes::Vector{HyperNode})

Identify the incident hyperedges to a vector of `HyperNode`s.
"""
function incident_edges(hypergraph::HyperGraph, hypernodes::Vector{HyperNode})
    # get vertices in hypergraph that are not in hypernodes
    external_nodes = setdiff(hypergraph.vertices, hypernodes) #nodes in hypergraph that aren't in hypernodes

    # create partition matrix
    I = []
    J = []
    for j in hypernodes
        push!(I, 1)
        push!(J, j)
    end
    for j in external_nodes
        push!(I, 2)
        push!(J, j)
    end

    V = Int.(ones(length(J)))
    # node partition matrix
    G = sparse(I, J, V)
    A = sparse(hypergraph)
    # edge partitions
    C = G * A

    # find shared edges; get indices of shared edges
    sum_vector = sum(C; dims=1)
    max_vector = maximum(C; dims=1)
    cross_vector = sum_vector - max_vector
    # nonzero indices of the cross vector; these are edges that cross partitions.
    indices = findall(cross_vector .!= 0)
    indices = [indices[i].I[2] for i in 1:length(indices)]
    incident_edges = HyperEdge[get_hyperedge(hypergraph, index) for index in indices]

    return incident_edges
end

"""
    identify_edges(hypergraph::HyperGraph,partitions::Vector{Vector{HyperNode}})

Identify both induced partition edges and cut edges given a partition of `HyperNode` vectors.
"""
function identify_edges(hypergraph::HyperGraph, partitions::Vector{Vector{HyperNode}})
    nparts = length(partitions)

    # create node partition matrix
    I = []
    J = []
    for i in 1:nparts
        for hypernode in partitions[i]
            j = hypernode
            push!(I, i)
            push!(J, j)
        end
    end
    V = Int.(ones(length(J)))
    G = sparse(I, J, V)
    A = Graphs.incidence_matrix(hypergraph)
    C = G * A

    # find shared edges; i.e. get indices of shared edges
    sum_vector = sum(C; dims=1)
    max_vector = maximum(C; dims=1)
    cross_vector = sum_vector - max_vector
    # get nonzero indices of the cross vector; these are edges that cross partitions.
    indices = findall(cross_vector .!= 0)
    indices = [indices[i].I[2] for i in 1:length(indices)]
    shared_edges = HyperEdge[]
    for index in indices
        push!(shared_edges, get_hyperedge(hypergraph, index))
    end

    # get induced partition edges (i.e. edges local to each partition)
    partition_edges = Vector[Vector{HyperEdge}() for _ in 1:nparts]
    for i in 1:nparts
        inds = findall(C[i, :] .!= 0)
        new_inds = filter(x -> !(x in indices), inds) #these are edge indices
        for new_ind in new_inds
            push!(partition_edges[i], get_hyperedge(hypergraph, new_ind))
        end
    end

    return partition_edges, shared_edges
end

"""
    identify_nodes(hypergraph::HyperGraph,partitions::Vector{Vector{HyperEdge}})

Identify both induced partition nodes and cut nodes given a partition of `HyperEdge` vectors.
"""
function identify_nodes(hypergraph::HyperGraph, partitions::Vector{Vector{HyperEdge}})
    nparts = length(partitions)

    # create edge partition matrix
    I = []
    J = []
    for i in 1:nparts
        for hyperedge in partitions[i]
            j = getindex(hypergraph, hyperedge)
            push!(I, i)
            push!(J, j)
        end
    end
    V = Int.(ones(length(J)))
    G = sparse(I, J, V)
    A = Graphs.incidence_matrix(hypergraph)
    C = A * G'

    # find the shared vertices; i.e. get indices of shared vertices
    sum_vector = sum(C; dims=2)
    max_vector = maximum(C; dims=2)
    cross_vector = sum_vector - max_vector
    indices = findall(cross_vector .!= 0)
    indices = [indices[i].I[1] for i in 1:length(indices)]
    shared_nodes = HyperNode[]
    for index in indices
        push!(shared_nodes, index)#getnode(hypergraph, index))
    end

    # get induced partition vertices (i.e. get vertices local to each partition)
    partition_nodes = Vector[Vector{HyperNode}() for _ in 1:nparts]
    for i in 1:nparts
        inds = findall(C[:, i] .!= 0)
        new_inds = filter(x -> !(x in indices), inds) #these are edge indices
        for new_ind in new_inds
            push!(partition_nodes[i], new_ind)#getnode(hypergraph, new_ind))
        end
    end

    return partition_nodes, shared_nodes
end

function induced_elements(hypergraph::HyperGraph, partitions::Vector{Vector{HyperNode}})
    return partitions
end

"""
    neighborhood(g::HyperGraph,nodes::Vector{OptiNode},distance::Int64)

Retrieve the neighborhood within `distance` of `nodes`.  Returns a vector of the original vertices and added vertices
"""
function neighborhood(g::HyperGraph, nodes::Vector{HyperNode}, distance::Int64)
    V = collect(nodes)
    nbr = copy(V)
    newnbr = copy(V) #neighbors to check
    addnbr = Int64[]
    for k in 1:distance
        for i in newnbr
            append!(addnbr, Graphs.all_neighbors(g, i)) #NOTE: union! is slow
        end
        newnbr = setdiff(addnbr, nbr)
    end
    nbr = unique([nbr; addnbr])
    return nbr
end

function expand(g::HyperGraph, nodes::Vector{HyperNode}, distance::Int64)
    new_nodes = neighborhood(g, nodes, distance)
    new_edges = induced_edges(g, new_nodes)
    return new_nodes, new_edges
end

####################################
#Print Functions
####################################
function string(graph::HyperGraph)
    return "Hypergraph: " * "($(nv(graph)) , $(ne(graph)))"
end
print(io::IO, graph::HyperGraph) = print(io, string(graph))
show(io::IO, graph::HyperGraph) = print(io, graph)

function string(edge::HyperEdge)
    return "HyperEdge: " * "$(collect(edge.vertices))"
end
print(io::IO, edge::HyperEdge) = print(io, string(edge))
show(io::IO, edge::HyperEdge) = print(io, edge)

### TODO

function Graphs.rem_edge!(g::HyperGraph, e::HyperEdge)
    throw(error("Edge removal not yet supported on hypergraphs"))
end
function Graphs.rem_vertex!(g::HyperGraph)
    throw(error("Vertex removal not yet supported on hypergraphs"))
end
