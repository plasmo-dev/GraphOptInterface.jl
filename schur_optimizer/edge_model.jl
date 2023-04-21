# include(joinpath(@__DIR__,"qp_data.jl"))
import MadNLP

const _SETS = Union{MOI.GreaterThan{Float64},MOI.LessThan{Float64},MOI.EqualTo{Float64}}

const _FUNCTIONS = Union{
    MOI.ScalarAffineFunction{Float64},
    MOI.ScalarQuadraticFunction{Float64},
}

mutable struct EdgeModel <: MOI.ModelLike
    qp_data::MadNLP.QPBlockData{Float64}
    nlp_dual_start::Union{Nothing,Vector{Float64}}
    nlp_data::MOI.NLPBlockData
    sense::MOI.OptimizationSense
end
function EdgeModel()
    return EdgeModel(
        QPBlockData{Float64}(),
        nothing,
        MOI.NLPBlockData([], _EmptyNLPEvaluator(), false),
        MOI.FEASIBILITY_SENSE
    )
end

# map each edge in graph to a model we can evaluate
function build_edge_model(edge::GOI.Edge; _differentiation_backend=MOI.Nonlinear.SparseReverseMode())
    edge_model = EdgeModel()
    vars = MOI.get(edge, MOI.ListOfVariableIndices())
    cons = MOI.get(edge, MOI.ListOfConstraintTypesPresent())
    
    # MOIU requires an index map to pass constraints. we assume a 1 to 1 mapping.
    var_map = MOIU.IndexMap()
    for var in vars
        var_map[var] = var
    end

    MOIU.pass_nonvariable_constraints(edge_model, edge.moi_model, var_map, cons)
    MOIU.pass_attributes(edge_model, edge.moi_model, var_map)

    # TODO: we might need to pass constraint attributes
    # MOIU.pass_attributes(edge_model, edge.moi_model, var_map, constraints) 
    
    # create nlp-block if needed
    if edge.nonlinear_model != nothing
        edge_evaluator = MOI.Nonlinear.Evaluator(edge.nonlinear_model, _differentiation_backend, vars)
        MOI.set(edge_model, MOI.NLPBlock(), MOI.NLPBlockData(edge_evaluator))
    end

    return edge_model
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
    return
end

MOI.get(edge::EdgeModel, ::MOI.NLPBlockDualStart) = edge.nlp_dual_start

### MOI.NLPBlock

MOI.supports(::EdgeModel, ::MOI.NLPBlock) = true

MOI.get(edge::EdgeModel, ::MOI.NLPBlock) = edge.nlp_data

function MOI.set(edge::EdgeModel, ::MOI.NLPBlock, nlp_data::MOI.NLPBlockData)
    edge.nlp_data = nlp_data
    return
end

### ObjectiveSense

MOI.supports(::EdgeModel, ::MOI.ObjectiveSense) = true

function MOI.set(
    edge::EdgeModel,
    ::MOI.ObjectiveSense,
    sense::MOI.OptimizationSense,
)
    edge.sense = sense
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
    return
end

### Eval_F_CB

function MOI.eval_objective(edge::EdgeModel, x::AbstractArray{T}) where T
    return MOI.eval_objective(edge.nlp_data.evaluator, x)
end

function MOI.eval_objective(edge::EdgeModel, x::AbstractArray{T}) where T
    if edge.sense == MOI.FEASIBILITY_SENSE
        return 0.0
    elseif edge.nlp_data.has_objective
        return MOI.eval_objective(edge.nlp_data.evaluator, x)
    end
    return MOI.eval_objective(edge.qp_data, x)
end

### Eval_Grad_F_CB

function MOI.eval_objective_gradient(edge::EdgeModel, grad::AbstractArray{T}, x::AbstractArray{T}) where T
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

function MOI.eval_constraint(edge::EdgeModel, g::AbstractArray{T}, x::AbstractArray{T}) where T
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

function MOI.eval_constraint_jacobian(edge::EdgeModel, values::AbstractArray{T}, x::AbstractArray{T}) where T
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

function MOI.eval_hessian_lagrangian(edge::EdgeModel, H::AbstractArray{T}, x::AbstractArray{T}, σ::Float64, μ::AbstractArray{T}) where T
    offset = MOI.eval_hessian_lagrangian(edge.qp_data, H, x, σ, μ)
    H_nlp = view(H, (offset+1):length(H))
    μ_nlp = view(μ, (length(edge.qp_data)+1):length(μ))
    MOI.eval_hessian_lagrangian(edge.nlp_data.evaluator, H_nlp, x, σ, μ_nlp)
    return
end