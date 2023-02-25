module BlockOptInterface

using MathOptInterface
using DataStructures
using Lazy

const MOI = MathOptInterface
const MOIU = MOI.Utilities
const MOIB = MOI.Bridges

include("block_optimizer.jl")

include("block.jl")

# include("block_nlp_evaluator.jl")

end 
