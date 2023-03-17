push!(LOAD_PATH, joinpath(@__DIR__, "../"))

import MadNLP: MadNLPSolver, AbstractNLPModel, MadNLPExecutionStats, QPBlockData

using MathOptInterface
const MOI = MathOptInterface

using BlockOptInterface
const BOI = BlockOptInterface
using NLPModels

struct _EmptyNLPEvaluator <: MOI.AbstractNLPEvaluator end

MOI.features_available(::_EmptyNLPEvaluator) = [:Grad, :Jac, :Hess]
MOI.initialize(::_EmptyNLPEvaluator, ::Any) = nothing
MOI.eval_constraint(::_EmptyNLPEvaluator, g, x) = nothing
MOI.jacobian_structure(::_EmptyNLPEvaluator) = Tuple{Int64,Int64}[]
MOI.hessian_lagrangian_structure(::_EmptyNLPEvaluator) = Tuple{Int64,Int64}[]
MOI.eval_constraint_jacobian(::_EmptyNLPEvaluator, J, x) = nothing
MOI.eval_hessian_lagrangian(::_EmptyNLPEvaluator, H, x, σ, μ) = nothing

"""
    SchurOptimizer()
Create a new MadNLP Schur optimizer.
"""
mutable struct SchurOptimizer <: BOI.AbstractBlockOptimizer
    solver::Union{Nothing,MadNLPSolver}   #interior point solver
    nlp::Union{Nothing,AbstractNLPModel}  #e.g. graph model
    result::Union{Nothing,MadNLPExecutionStats{Float64}}

    name::String
    invalid_model::Bool
    silent::Bool
    options::Dict{Symbol,Any}
    solve_time::Float64
    solve_iterations::Int
    sense::MOI.OptimizationSense

    block::BOI.Block
end

function SchurOptimizer(; kwargs...)
    option_dict = Dict{Symbol, Any}()
    for (name, value) in kwargs
        option_dict[name] = value
    end
    return SchurOptimizer(
        nothing,
        nothing,
        nothing,
        "",
        false,
        false,
        option_dict,
        NaN,
        0,
        MOI.FEASIBILITY_SENSE,
        BOI.Block(0)
    )
end

mutable struct NodeModel <: BOI.NodeModelLike
    variables::MOI.Utilities.VariablesContainer{Float64}
    variable_primal_start::Vector{Union{Nothing,Float64}}
    mult_x_L::Vector{Union{Nothing,Float64}}
    mult_x_U::Vector{Union{Nothing,Float64}}
end
function NodeModel(optimizer::SchurOptimizer)
    return NodeModel(
        MOI.Utilities.VariablesContainer{Float64}(),
        Union{Nothing,Float64}[],
        Union{Nothing,Float64}[],
        Union{Nothing,Float64}[]
    )
end

mutable struct EdgeModel <: BOI.EdgeModelLike
    qp_data::QPBlockData{Float64}
    nlp_dual_start::Union{Nothing,Vector{Float64}}
    nlp_data::MOI.NLPBlockData
    sense::MOI.OptimizationSense
end
function EdgeModel(optimizer::SchurOptimizer)
    return EdgeModel(
        QPBlockData{Float64}(),
        nothing,
        MOI.NLPBlockData([], _EmptyNLPEvaluator(), false),
        MOI.FEASIBILITY_SENSE
    )
end

function MOI.supports(optimizer::SchurOptimizer, ::BOI.BlockStructure)
    return true
end

function MOI.get(optimizer::SchurOptimizer, ::BOI.BlockStructure)
    return optimizer.block
end

function BOI.node_model(optimizer::SchurOptimizer)
    return NodeModel(optimizer)
end

function BOI.edge_model(optimizer::SchurOptimizer)
    return EdgeModel(optimizer)
end



const _SETS = Union{MOI.GreaterThan{Float64},MOI.LessThan{Float64},MOI.EqualTo{Float64}}

const _FUNCTIONS = Union{
    MOI.ScalarAffineFunction{Float64},
    MOI.ScalarQuadraticFunction{Float64},
}



# function BOI.add_node!(optimizer::SchurOptimizer, index::BOI.BlockIndex)
#     block = optimizer.block.block_by_index[index]
#     node = BOI.add_node!(block, NodeModel(optimizer))
# end

# function BOI.add_edge!(optimizer::SchurOptimizer, index::BOI.BlockIndex, node::BOI.Node)
#     block = optimizer.block.block_by_index[index]
#     return BOI.add_edge!(block, node, EdgeModel(optimizer))
# end

# function BOI.add_edge!(optimizer::SchurOptimizer, index::BOI.BlockIndex, nodes::NTuple{N, BOI.Node} where N)
#     block = optimizer.block.block_by_index[index]
#     return BOI.add_edge!(block, nodes, EdgeModel(optimizer))
# end

# function BOI.add_edge!(optimizer::SchurOptimizer, index::BOI.BlockIndex, blocks::NTuple{N, BOI.Block} where N)
#     block = optimizer.block.block_by_index[index]
#     return BOI.add_edge!(block, blocks, EdgeModel(optimizer))
# end

# function BOI.add_edge!(optimizer::SchurOptimizer, index::BOI.BlockIndex, node::BOI.Node, block::BOI.Block)
#     parent_block = optimizer.block.block_by_index[index]
#     return BOI.add_edge!(parent_block, node, block, EdgeModel(optimizer))
# end

# TODO: figure out whether we can do these
# MOI.supports_incremental_interface(::SchurOptimizer) = false
# function MOI.copy_to(model::SchurOptimizer, src::BOI.ModelLike)
#     return MOI.Utilities.default_copy_to(model, src)
# end

MOI.get(::SchurOptimizer, ::MOI.SolverName) = "MadNLP.Schur"

### MOI.Name

MOI.supports(::SchurOptimizer, ::MOI.Name) = true

function MOI.set(model::SchurOptimizer, ::MOI.Name, value::String)
    model.name = value
    return
end

MOI.get(model::SchurOptimizer, ::MOI.Name) = model.name

### MOI.Silent

MOI.supports(::SchurOptimizer, ::MOI.Silent) = true

function MOI.set(model::SchurOptimizer, ::MOI.Silent, value)
    model.silent = value
    return
end

MOI.get(model::SchurOptimizer, ::MOI.Silent) = model.silent

### MOI.TimeLimitSec

MOI.supports(::SchurOptimizer, ::MOI.TimeLimitSec) = true

function MOI.set(model::SchurOptimizer, ::MOI.TimeLimitSec, value::Real)
    MOI.set(model, MOI.RawSchurOptimizerAttribute("max_cpu_time"), Float64(value))
    return
end

function MOI.set(model::SchurOptimizer, ::MOI.TimeLimitSec, ::Nothing)
    delete!(model.options, "max_cpu_time")
    return
end

function MOI.get(model::SchurOptimizer, ::MOI.TimeLimitSec)
    return get(model.options, "max_cpu_time", nothing)
end

### MOI.RawOptimizerAttribute

MOI.supports(::SchurOptimizer, ::MOI.RawOptimizerAttribute) = true

function MOI.set(model::SchurOptimizer, p::MOI.RawOptimizerAttribute, value)
    model.options[Symbol(p.name)] = value
    # No need to reset model.solver because this gets handled in optimize!.
    return
end

function MOI.get(model::SchurOptimizer, p::MOI.RawOptimizerAttribute)
    if !haskey(model.options, p.name)
        error("RawParameter with name $(p.name) is not set.")
    end
    return model.options[p.name]
end


### Variables / Nodes
function MOI.copy_to(node::NodeModel, src::MOI.ModelLike)
    return MOI.Utilities.default_copy_to(node, src)
end

column(x::MOI.VariableIndex) = x.value

function MOI.add_variable(node::NodeModel)
    push!(node.variable_primal_start, nothing)
    push!(node.mult_x_L, nothing)
    push!(node.mult_x_U, nothing)
    return MOI.add_variable(node.variables)
end

function MOI.is_valid(node::NodeModel, x::MOI.VariableIndex)
    return MOI.is_valid(node.variables, x)
end

function MOI.get(
    node::NodeModel,
    attr::Union{MOI.NumberOfVariables,MOI.ListOfVariableIndices},
)
    return MOI.get(node.variables, attr)
end


function MOI.is_valid(
    node::NodeModel,
    ci::MOI.ConstraintIndex{MOI.VariableIndex,<:_SETS},
)
    return MOI.is_valid(node.variables, ci)
end

function MOI.get(
    model::NodeModel,
    attr::Union{
        MOI.NumberOfConstraints{MOI.VariableIndex,<:_SETS},
        MOI.ListOfConstraintIndices{MOI.VariableIndex,<:_SETS},
    },
)
    return MOI.get(node.variables, attr)
end

function MOI.get(
    node::NodeModel,
    attr::Union{MOI.ConstraintFunction,MOI.ConstraintSet},
    ci::MOI.ConstraintIndex{MOI.VariableIndex,<:_SETS},
)
    return MOI.get(node.variables, attr, ci)
end

function MOI.add_constraint(node::NodeModel, x::MOI.VariableIndex, set::_SETS)
    index = MOI.add_constraint(node.variables, x, set)
    # model.solver = nothing
    return index
end

function MOI.set(
    node::NodeModel,
    ::MOI.ConstraintSet,
    ci::MOI.ConstraintIndex{MOI.VariableIndex,S},
    set::S,
) where {S<:_SETS}
    MOI.set(node.variables, MOI.ConstraintSet(), ci, set)
    #model.solver = nothing
    return
end

function MOI.delete(
    node::NodeModel,
    ci::MOI.ConstraintIndex{MOI.VariableIndex,<:_SETS},
)
    MOI.delete(node.variables, ci)
    #model.solver = nothing
    return
end

# Constraints / Edges
### ScalarAffineFunction and ScalarQuadraticFunction constraints
function MOI.copy_to(edge::EdgeModel, src::MOI.ModelLike)
    return MOI.Utilities.default_copy_to(edge, src)
end

function MOI.supports_constraint(
    ::EdgeModel,
    ::Type{<:Union{MOI.VariableIndex,_FUNCTIONS}},
    ::Type{<:_SETS},
)
    return true
end

function MOI.is_valid(
    edge::EdgeModel,
    ci::MOI.ConstraintIndex{<:_FUNCTIONS,<:_SETS},
)
    return MOI.is_valid(edge.qp_data, ci)
end

### MOI.ListOfConstraintTypesPresent

function MOI.get(edge::EdgeModel, attr::MOI.ListOfConstraintTypesPresent)
    return MOI.get(edge.qp_data, attr)
end


function MOI.add_constraint(
    edge::EdgeModel,
    func::_FUNCTIONS, 
    set::_SETS
)
    # TODO: variable mapping
    index = MOI.add_constraint(edge.qp_data, func, set)
    #model.solver = nothing
    return index
end

function MOI.get(
    edge::EdgeModel,
    attr::Union{MOI.NumberOfConstraints{F,S},MOI.ListOfConstraintIndices{F,S}},
) where {F<:_FUNCTIONS,S<:_SETS}
    return MOI.get(edge.qp_data, attr)
end

function MOI.get(
    edge::EdgeModel,
    attr::Union{
        MOI.ConstraintFunction,
        MOI.ConstraintSet,
        MOI.ConstraintDualStart,
    },
    ci::MOI.ConstraintIndex{F,S},
) where {F<:_FUNCTIONS,S<:_SETS}
    return MOI.get(edge.qp_data, attr, ci)
end

function MOI.set(
    edge::EdgeModel,
    ::MOI.ConstraintSet,
    ci::MOI.ConstraintIndex{F,S},
    set::S,
) where {F<:_FUNCTIONS,S<:_SETS}
    MOI.set(edge.qp_data, MOI.ConstraintSet(), ci, set)
    #model.solver = nothing
    return
end

function MOI.supports(
    ::EdgeModel,
    ::MOI.ConstraintDualStart,
    ::Type{MOI.ConstraintIndex{F,S}},
) where {F<:_FUNCTIONS,S<:_SETS}
    return true
end

function MOI.set(
    edge::EdgeModel,
    attr::MOI.ConstraintDualStart,
    ci::MOI.ConstraintIndex{F,S},
    value::Union{Real,Nothing},
) where {F<:_FUNCTIONS,S<:_SETS}
    MOI.throw_if_not_valid(model, ci)
    MOI.set(edge.qp_data, attr, ci, value)
    # No need to reset model.solver, because this gets handled in optimize!.
    return
end


### MOI.NLPBlockDualStart

MOI.supports(::EdgeModel, ::MOI.NLPBlockDualStart) = true

function MOI.set(
    edge::EdgeModel,
    ::MOI.NLPBlockDualStart,
    values::Union{Nothing,Vector},
)
    edge.nlp_dual_start = values
    # No need to reset model.solver, because this gets handled in optimize!.
    return
end

MOI.get(edge::EdgeModel, ::MOI.NLPBlockDualStart) = edge.nlp_dual_start

### MOI.NLPBlock

MOI.supports(::EdgeModel, ::MOI.NLPBlock) = true

MOI.get(edge::EdgeModel, ::MOI.NLPBlock) = edge.nlp_data

function MOI.set(edge::EdgeModel, ::MOI.NLPBlock, nlp_data::MOI.NLPBlockData)
    edge.nlp_data = nlp_data
    #model.solver = nothing
    return
end

# Objectives/Edges
### ObjectiveSense

MOI.supports(::EdgeModel, ::MOI.ObjectiveSense) = true

function MOI.set(
    edge::EdgeModel,
    ::MOI.ObjectiveSense,
    sense::MOI.OptimizationSense,
)
    edge.sense = sense
    #model.solver = nothing
    return
end

MOI.get(edge::EdgeModel, ::MOI.ObjectiveSense) = edge.sense

### ObjectiveFunction
function MOI.supports(
    ::EdgeModel,
    ::MOI.ObjectiveFunction{<:Union{MOI.VariableIndex,<:_FUNCTIONS}},
)
    return true
end

function MOI.get(
    edge::EdgeModel,
    attr::Union{MOI.ObjectiveFunctionType,MOI.ObjectiveFunction},
)
    return MOI.get(edge.qp_data, attr)
end

function MOI.set(
    edge::EdgeModel,
    attr::MOI.ObjectiveFunction{F},
    func::F,
) where {F<:Union{MOI.VariableIndex,<:_FUNCTIONS}}
    MOI.set(edge.qp_data, attr, func)
    # model.solver = nothing
    return
end

### Eval_F_CB

function MOI.eval_objective(edge::EdgeModel, x)
    if edge.sense == MOI.FEASIBILITY_SENSE
        return 0.0
    elseif edge.nlp_data.has_objective
        return MOI.eval_objective(edge.nlp_data.evaluator, x)
    end
    return MOI.eval_objective(edge.qp_data, x)
end

### Eval_Grad_F_CB

function MOI.eval_objective_gradient(edge::EdgeModel, grad, x)
    if edge.sense == MOI.FEASIBILITY_SENSE
        grad .= zero(eltype(grad))
    elseif edge.nlp_data.has_objective
        MOI.eval_objective_gradient(edge.nlp_data.evaluator, grad, x)
    else
        MOI.eval_objective_gradient(edge.qp_data, grad, x)
    end
    return
end

### Eval_G_CB

function MOI.eval_constraint(edge::EdgeModel, g, x)
    MOI.eval_constraint(edge.qp_data, g, x)
    g_nlp = view(g, (length(edge.qp_data)+1):length(g))
    MOI.eval_constraint(edge.nlp_data.evaluator, g_nlp, x)
    return
end

### Eval_Jac_G_CB

function MOI.jacobian_structure(edge::EdgeModel)
    J = MOI.jacobian_structure(edge.qp_data)
    offset = length(edge.qp_data)
    if length(edge.nlp_data.constraint_bounds) > 0
        for (row, col) in MOI.jacobian_structure(edge.nlp_data.evaluator)
            push!(J, (row + offset, col))
        end
    end
    return J
end

function MOI.eval_constraint_jacobian(edge::EdgeModel, values, x)
    offset = MOI.eval_constraint_jacobian(edge.qp_data, values, x)
    nlp_values = view(values, (offset+1):length(values))
    MOI.eval_constraint_jacobian(edge.nlp_data.evaluator, nlp_values, x)
    return
end

### Eval_H_CB

function MOI.hessian_lagrangian_structure(edge::EdgeModel)
    H = MOI.hessian_lagrangian_structure(edge.qp_data)
    append!(H, MOI.hessian_lagrangian_structure(edge.nlp_data.evaluator))
    return H
end

function MOI.eval_hessian_lagrangian(edge::EdgeModel, H, x, σ, μ)
    offset = MOI.eval_hessian_lagrangian(edge.qp_data, H, x, σ, μ)
    H_nlp = view(H, (offset+1):length(H))
    μ_nlp = view(μ, (length(edge.qp_data)+1):length(μ))
    MOI.eval_hessian_lagrangian(edge.nlp_data.evaluator, H_nlp, x, σ, μ_nlp)
    return
end

### NLP Models Wrapper

struct BlockNLPModel{T} <: AbstractNLPModel{T,Vector{T}}
    meta::NLPModelMeta{T, Vector{T}}
    optimizer::SchurOptimizer
    counters::NLPModels.Counters
end

function _fill_variable_info!(block, block_data, x0, x_lower, x_upper)
    # loop through each node
    for node in block.nodes
        ninds = block_data.node_data[node]
        x0_node = Vector{Float64}(undef, BOI._num_variables(node))
        for i = 1:BOI._num_variables(node)
            if node.model.variable_primal_start[i] !== nothing
                x0_node[i] = node.model.variable_primal_start[i]
            else
                x0_node[i] = clamp(0, node.model.variables.lower[i], node.model.variables.upper[i])
            end
        end
        x0[ninds] .= x0_node
        x_lower[ninds] .= node.model.variables.lower
        x_upper[ninds] .= node.model.variables.upper
    end

    # recursively call sub-blocks
    for sub_block in block.sub_blocks
        sub_block_data = block_data.sub_block_data[sub_block.index]
        _fill_variable_info!(sub_block, sub_block_data, x0, x_lower, x_upper)
    end
    return
end

_dual_start(::EdgeModel, ::Nothing, ::Int=1) = 0.0

_dual_start(model::EdgeModel, value::Real, scale::Int=1) = value*scale

function _fill_constraint_info!(block, block_data, y0, c_lower, c_upper)
    # loop through each edge
    for edge in block.edges
        minds = block_data.edge_data[edge].row_indices
        model = edge.model
        g_L = copy(model.qp_data.g_L)
        g_U = copy(model.qp_data.g_U)
        for bound in model.nlp_data.constraint_bounds
            push!(g_L, bound.lower)
            push!(g_U, bound.upper)
        end
        c_lower[minds] .= g_L
        c_upper[minds] .= g_U

        # dual start
        y0_edge = Vector{Float64}(undef, BOI._num_constraints(edge))
        for (i, start) in enumerate(model.qp_data.mult_g)
            y0_edge[i] = _dual_start(model, start, -1)
        end
        offset = length(model.qp_data.mult_g)
        if model.nlp_dual_start === nothing
            y0_edge[(offset+1):end] .= 0.0
        else
            for (i, start) in enumerate(model.nlp_dual_start::Vector{Float64})
                y0_edge[offset+i] = _dual_start(model, start, -1)
            end
        end
        y0[minds] .= y0_edge
    end

    # recursively call sub-blocks
    for sub_block in block.sub_blocks
        sub_block_data = block_data.sub_block_data[sub_block.index]
        _fill_constraint_info!(sub_block, sub_block_data, y0, c_lower, c_upper)
    end
    return
end

obj(nlp::BlockNLPModel,x::Vector{Float64}) = MOI.eval_objective(nlp.model,x)

function grad!(nlp::BlockNLPModel,x::Vector{Float64},f::Vector{Float64})
    MOI.eval_objective_gradient(nlp.model,f,x)
end

function cons!(nlp::BlockNLPModel,x::Vector{Float64},c::Vector{Float64})
    MOI.eval_constraint(nlp.model,c,x)
end

function jac_coord!(nlp::BlockNLPModel,x::Vector{Float64},jac::Vector{Float64})
    MOI.eval_constraint_jacobian(nlp.model,jac,x)
end

function hess_coord!(nlp::BlockNLPModel,x::Vector{Float64},l::Vector{Float64},hess::Vector{Float64}; obj_weight::Float64=1.)
    MOI.eval_hessian_lagrangian(nlp.model,hess,x,obj_weight,l)
end

function hess_structure!(nlp::BlockNLPModel, I::AbstractVector{T}, J::AbstractVector{T}) where T
    cnt = 1
    for (row, col) in  MOI.hessian_lagrangian_structure(nlp.model)
        I[cnt], J[cnt] = row, col
        cnt += 1
    end
end

function jac_structure!(nlp::BlockNLPModel, I::AbstractVector{T}, J::AbstractVector{T}) where T
    cnt = 1
    for (row, col) in  MOI.jacobian_structure(nlp.model)
        I[cnt], J[cnt] = row, col
        cnt += 1
    end
end

function BlockNLPModel(model::SchurOptimizer)

    # initialize
    block_evaluator = BOI.BlockEvaluator(model.block)
    MOI.initialize(block_evaluator, [:Grad, :Hess, :Jac])
    block = model.block
    block_data = block_evaluator.block_data

    # primals, lower, upper bounds
    nvar = block_data.num_variables
    x0 = Vector{Float64}(undef,nvar) # primal start
    x_lower = Vector{Float64}(undef,nvar)
    x_upper = Vector{Float64}(undef,nvar)
    _fill_variable_info!(block, block_data, x0, x_lower, x_upper)

    # duals, constraints lower & upper bounds
    ncon = block_data.num_constraints
    y0 = Vector{Float64}(undef, ncon) # dual start
    c_lower = Vector{Float64}(undef, ncon)
    c_upper = Vector{Float64}(undef, ncon)
    _fill_constraint_info!(block, block_data, y0, c_lower, c_upper)

    # Sparsity
    nnzh = block_data.nnz_hess
    nnzj = block_data.nnz_jac

    model.options[:jacobian_constant], model.options[:hessian_constant] = false, false
    model.options[:dual_initialized] = !iszero(y0)

    return BlockNLPModel(
        NLPModelMeta(
            nvar,
            x0 = x0,
            lvar = x_lower,
            uvar = x_upper,
            ncon = ncon,
            y0 = y0,
            lcon = c_lower,
            ucon = c_upper,
            nnzj = nnzj,
            nnzh = nnzh,
            minimize = model.sense == MOI.MIN_SENSE
        ),
        model,
        NLPModels.Counters())
end

# # Optimize!
# function MOI.optimize!(model::SchurOptimizer)
#     model.nlp = BlockMOIModel(model)
#     if model.silent
#         model.options[:print_level] = MadNLP.ERROR
#     end

#     # get block partitions from block structure
#     partitions = get_block_partitions(model)

#     # pass schur options
#     schur_options = SchurLinearOptions()

#     model.solver = MadNLP.MadNLPSolver(
#         model.nlp; 
#         linear_solver=SchurLinearSolver,
#         model.options...
#     )
#     model.result = solve!(model.solver)
#     model.solve_time = model.solver.cnt.total_time
#     model.solve_iterations = model.solver.cnt.k
#     return
# end

# # Schur optimizer needs a two-stage partition
# function get_block_partitions(optimizer::SchurOptimizer)

#     # IDEA: use block to get indices
#     # pinds should come from linking edges

#     n = nlp.ext[:n]
#     m = nlp.ext[:m]
#     p = nlp.ext[:p]

#     ninds = nlp.ninds
#     minds = nlp.minds
#     pinds = nlp.pinds



#     # this is an NLP function. try getting this from the block
#     ind_ineq = findall(get_lcon(nlp).!=get_ucon(nlp))
#     l = length(ind_ineq)

#     partition = Vector{Int}(undef,n+m+l+p)

#     for k=1:length(ninds)
#         part[ninds[k]].=k
#     end
#     for k=1:length(minds)
#         part[minds[k].+n.+l].=k
#     end

#     # loop through edges
#     cnt = 0

#     # for linkedge in nlp.ext[:linkedges]
#     #     for (ind,con) in linkedge.linkconstraints
#     #         cnt+=1
#     #         attached_node_idx = graph.node_idx_map[con.attached_node]
#     #         part[n+l+m+cnt] = attached_node_idx != nothing ? attached_node_idx : error("All the link constraints need to be attached to a node")
#     #     end
#     # end

#     cnt = 0
#     for q in ind_ineq
#         cnt+=1
#         part[n+cnt] = part[n+l+q]
#     end

#     return partition
# end