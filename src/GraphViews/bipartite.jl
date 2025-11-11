"""
    BipartiteGraph

A simple bipartite graph.  Contains two vertex sets to enforce bipartite structure.
"""
mutable struct BipartiteGraph <: Graphs.AbstractGraph{Int64}
    graph::Graphs.Graph
    vertexset1::Vector{Int64}
    vertexset2::Vector{Int64}
end
"""
    BipartiteGraph()

Construct an empty `BipartiteGraph` with no vertices or edges.
"""
function BipartiteGraph()
    return BipartiteGraph(Graphs.Graph(), Vector{Int64}(), Vector{Int64}())
end

"""
    Graphs.add_vertex!(bgraph::BipartiteGraph; bipartite=1)

Add a vertex to the bipartite graph. The `bipartite` keyword argument specifies
which vertex set to add the vertex to (1 or 2). Returns the success status.
"""
function Graphs.add_vertex!(bgraph::BipartiteGraph; bipartite=1)
    added = Graphs.add_vertex!(bgraph.graph)
    vertex = Graphs.nv(bgraph.graph)
    if bipartite == 1
        push!(bgraph.vertexset1, vertex)
    else
        @assert bipartite == 2
        push!(bgraph.vertexset2, vertex)
    end
    return added
end

"""
    Graphs.add_edge!(bgraph::BipartiteGraph, from::Int64, to::Int64)

Add an edge between vertices `from` and `to`. Enforces bipartite structure by
requiring that the vertices be in different vertex sets.
"""
function Graphs.add_edge!(bgraph::BipartiteGraph, from::Int64, to::Int64)
    length(intersect((from, to), bgraph.vertexset1)) == 1 ||
        error("$from and $to must be in separate vertex sets")
    return Graphs.add_edge!(bgraph.graph, from, to)
end

"""
    Graphs.edges(bgraph::BipartiteGraph)

Return an iterator over all edges in the bipartite graph.
"""
Graphs.edges(bgraph::BipartiteGraph) = Graphs.edges(bgraph.graph)

"""
    Graphs.edgetype(bgraph::BipartiteGraph)

Return the edge type for a bipartite graph (`SimpleEdge{Int64}`).
"""
Graphs.edgetype(bgraph::BipartiteGraph) = Graphs.SimpleGraphs.SimpleEdge{Int64}

"""
    Graphs.has_edge(bgraph::BipartiteGraph, from::Int64, to::Int64)

Check if the bipartite graph has an edge from `from` to `to`.
"""
function Graphs.has_edge(bgraph::BipartiteGraph, from::Int64, to::Int64)
    return Graphs.has_edge(bgraph.graph, from, to)
end

"""
    Graphs.has_vertex(bgraph::BipartiteGraph, v::Integer)

Check if the bipartite graph contains vertex `v`.
"""
function Graphs.has_vertex(bgraph::BipartiteGraph, v::Integer)
    return Graphs.has_vertex(bgraph.graph, v)
end

"""
    Graphs.is_directed(bgraph::BipartiteGraph)

Bipartite graphs are undirected. Always returns `false`.
"""
Graphs.is_directed(bgraph::BipartiteGraph) = false

"""
    Graphs.is_directed(::Type{BipartiteGraph})

Bipartite graphs are undirected. Always returns `false`.
"""
Graphs.is_directed(::Type{BipartiteGraph}) = false

"""
    Graphs.ne(bgraph::BipartiteGraph)

Return the number of edges in the bipartite graph.
"""
Graphs.ne(bgraph::BipartiteGraph) = Graphs.ne(bgraph.graph)

"""
    Graphs.nv(bgraph::BipartiteGraph)

Return the number of vertices in the bipartite graph.
"""
Graphs.nv(bgraph::BipartiteGraph) = Graphs.nv(bgraph.graph)

"""
    Graphs.vertices(bgraph::BipartiteGraph)

Return a vector of all vertices in the bipartite graph.
"""
Graphs.vertices(bgraph::BipartiteGraph) = Graphs.vertices(bgraph.graph)

"""
    Graphs.adjacency_matrix(bgraph::BipartiteGraph)

Return the adjacency matrix of the bipartite graph. Rows correspond to vertices
in the first set, columns correspond to vertices in the second set.
"""
function Graphs.adjacency_matrix(bgraph::BipartiteGraph)
    n_v1 = length(bgraph.vertexset1)
    n_v2 = length(bgraph.vertexset2)
    A = spzeros(n_v1, n_v2)
    for edge in Graphs.edges(bgraph.graph)
        A[edge.src, edge.dst - n_v1] = 1
    end
    return A
end

"""
    identify_separators(bgraph::BipartiteGraph, partitions::Vector; cut_selector=Graphs.degree)

Identify separator elements (cut vertices or cut edges) that separate the given `partitions`.
The `cut_selector` parameter determines how to assign boundary elements to cuts. It can be:
- A function (e.g., `Graphs.degree`) that compares element degrees
- `:vertex` to always assign vertices as separators
- `:edge` to always assign edges as separators

Returns a tuple `(partition_elements, cross_elements)` where `partition_elements` are the 
elements local to each partition and `cross_elements` are the separator elements.
"""
function identify_separators(
    bgraph::BipartiteGraph, partitions::Vector; cut_selector=Graphs.degree
)
    nparts = length(partitions)

    # create partition matrix
    I = []
    J = []
    for i in 1:nparts
        for vertex in partitions[i]
            push!(I, i)
            push!(J, vertex)
        end
    end
    V = Int.(ones(length(J)))
    # node partition matrix
    G = sparse(I, J, V)
    A = Graphs.incidence_matrix(bgraph.graph)
    # bipartite edge partitions
    C = G * A

    # find shared nodes; i.e. get indices of shared nodes
    sum_vector = sum(C; dims=1)
    max_vector = maximum(C; dims=1)
    cross_vector = sum_vector - max_vector
    indices = findall(cross_vector .!= 0)
    indices = [indices[i].I[2] for i in 1:length(indices)]

    # assign boundary vertices to actual cross cuts 
    # (a vertex is a cut node or a cut edge)
    es = collect(Graphs.edges(bgraph.graph))
    cut_edges = es[indices]
    cross_elements = Int64[]
    for edge in cut_edges
        src = edge.src #src is always the true hypernode
        dst = edge.dst #dest is always the true hyperedge
        if cut_selector == :vertex
            push!(cross_elements, src)
        elseif cut_selector == :edge
            push!(cross_elements, dst)
        else #use selection function
            if cut_selector(bgraph.graph, src) >= cut_selector(bgraph.graph, dst)
                push!(cross_elements, src) #tie goes to vertex
            else
                push!(cross_elements, dst)
            end
        end
    end

    # get induced elements: need to remove the cut element from these lists
    partition_elements = Vector[Vector{Int64}() for _ in 1:nparts]
    for i in 1:nparts
        new_inds = filter(x -> !(x in cross_elements), partitions[i])
        for new_ind in new_inds
            push!(partition_elements[i], new_ind)
        end
    end

    return partition_elements, cross_elements
end

"""
    induced_elements(bgraph::BipartiteGraph, partitions::Vector; cut_selector=Graphs.degree)

Get the induced elements for each partition (i.e., elements local to each partition,
excluding separators). This is a convenience function that returns only the first element
of `identify_separators`.

# Arguments
- `bgraph::BipartiteGraph`: The bipartite graph
- `partitions::Vector`: Vector of partitions
- `cut_selector=Graphs.degree`: Selector for determining cut assignment (function, `:vertex`, or `:edge`)

# Returns
A vector of vectors, where each inner vector contains the induced elements for that partition.
"""
function induced_elements(
    bgraph::BipartiteGraph, partitions::Vector; cut_selector=Graphs.degree
)
    return identify_separators(bgraph, partitions; cut_selector=cut_selector)[1]
end
