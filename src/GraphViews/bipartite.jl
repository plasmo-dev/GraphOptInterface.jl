"""
    BipartiteGraph

A simple bipartite graph.  Contains two vertex sets to enforce bipartite structure.
"""
mutable struct BipartiteGraph <: Graphs.AbstractGraph{Int64}
    graph::Graphs.Graph
    vertexset1::Vector{Int64}
    vertexset2::Vector{Int64}
end
function BipartiteGraph()
    return BipartiteGraph(Graphs.Graph(), Vector{Int64}(), Vector{Int64}())
end

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

function Graphs.add_edge!(bgraph::BipartiteGraph, from::Int64, to::Int64)
    length(intersect((from, to), bgraph.vertexset1)) == 1 ||
        error("$from and $to must be in separate vertex sets")
    return Graphs.add_edge!(bgraph.graph, from, to)
end

Graphs.edges(bgraph::BipartiteGraph) = Graphs.edges(bgraph.graph)

Graphs.edgetype(bgraph::BipartiteGraph) = Graphs.SimpleGraphs.SimpleEdge{Int64}

function Graphs.has_edge(bgraph::BipartiteGraph, from::Int64, to::Int64)
    return Graphs.has_edge(bgraph.graph, from, to)
end

function Graphs.has_vertex(bgraph::BipartiteGraph, v::Integer)
    return Graphs.has_vertex(bgraph.graph, v)
end

Graphs.is_directed(bgraph::BipartiteGraph) = false

Graphs.is_directed(::Type{BipartiteGraph}) = false

Graphs.ne(bgraph::BipartiteGraph) = Graphs.ne(bgraph.graph)

Graphs.nv(bgraph::BipartiteGraph) = Graphs.nv(bgraph.graph)

Graphs.vertices(bgraph::BipartiteGraph) = Graphs.vertices(bgraph.graph)

function Graphs.adjacency_matrix(bgraph::BipartiteGraph)
    n_v1 = length(bgraph.vertexset1)
    n_v2 = length(bgraph.vertexset2)
    A = spzeros(n_v1, n_v2)
    for edge in Graphs.edges(bgraph.graph)
        A[edge.src, edge.dst - n_v1] = 1
    end
    return A
end

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

function induced_elements(
    bgraph::BipartiteGraph, partitions::Vector; cut_selector=Graphs.degree
)
    return identify_separators(bgraph, partitions; cut_selector=cut_selector)[1]
end
