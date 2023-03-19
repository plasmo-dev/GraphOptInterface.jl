mutable struct TwoStagePartition
    nparts::Int
    part::Vector{Int}
end


function TwoStagePartition(
    csc::SparseMatrixCSC,
    part::Vector{Int},
    nparts::Int
)
    # if isempty(part) || findfirst(x->x==0.,part) == nothing
    #     g = Graph(csc)
    #     isempty(part) && (part = partition(g, nparts, alg=:KWAY))
    #     mark_boundary!(g,part)
    # end
    return TwoStagePartition(nparts,part)
end

# convert sparse matrix to graph
Graph(csc::SparseMatrixCSC) = Graph(getelistcsc(csc.colptr,csc.rowval))

# get edge list
getelistcsc(colptr,rowval) = [Edge(i,Int(j)) for i=1:length(colptr)-1 for j in @view rowval[colptr[i]:colptr[i+1]-1]]


function mark_boundary!(g,part)
    for e in edges(g)
        (part[src(e)]!=part[dst(e)] && part[src(e)]!= 0 && part[dst(e)] != 0) &&
            (part[src(e)] = 0; part[dst(e)] = 0)
    end
end

    # if isempty(part) || findfirst(x->x==0.,part) == nothing
    #     g = Graph(csc)
    #     isempty(part) && (part = partition(g, nparts, alg=:KWAY))
    #     mark_boundary!(g,part)
    # end