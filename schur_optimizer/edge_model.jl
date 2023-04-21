mutable struct EdgeModel <: BOI.EdgeModelLike
    qp_data::QPBlockData{Float64}
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