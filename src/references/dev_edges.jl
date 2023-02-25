abstract type AbstractNode end
abstract type AbstractEdge end

struct Node <: AbstractNode
	index::Int64
end

struct Block <: AbstractNode
	index::Int64
end

struct Edge{T<:Tuple}
    nodes::T
end

node1 = Node(1)
node2 = Node(2)
node3 = Node(3)
block1 = Block(1)
block2 = Block(2)

##################################
# two nodes
edge1 = Edge((node1, node2))

# three nodes
edge2 = Edge((node1, node2, node3))

# a self edge
edge3 = Edge((node1,))


edge4 = Edge((node1, block1))

edge5 = Edge((block1, block2))

# print functions for each type of edge
function print_test(edge::Edge{NTuple{N, Node}} where N) 
    println("This Edge connects $(length(edge.nodes)) nodes of type $(Node)")
end

function print_test(edge::Edge{NTuple{N, Block}}) where N
    println("This Edge connects $(length(edge.nodes)) blocks of type $(Node)")
end

# Define a function that takes an Edge that connects a single node to itself
function print_test(edge::Edge{NTuple{1,Node}})
    println("This Edge is a self-edge and connects a single node")
end

# Define a function that takes an Edge that connects two nodes of different types
function print_test(edge::Edge{Tuple{Node, Block}})
    println("This Edge connects a node to a sub-block")
end
