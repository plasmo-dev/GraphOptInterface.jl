# NLP Models interface from MadNLP.jl
struct GraphModel{T} <: AbstractNLPModel{T,Vector{T}}
	blk::GraphBlockData
    ninds::Vector{UnitRange{Int}}
    minds::Vector{UnitRange{Int}}
    pinds::Vector{UnitRange{Int}}
    nnzs_jac_inds::Vector{UnitRange{Int}}
    nnzs_hess_inds::Vector{UnitRange{Int}}
    nnzs_link_jac_inds::Vector{UnitRange{Int}}

    meta::NLPModelMeta{T, Vector{T}}
    counters::MadNLP.NLPModels.Counters
    ext::Dict{Symbol,Any}
end


# Plasmo.jl MOI interface functions
function obj(nlp::GraphModel, x::AbstractVector)
    return eval_objective(nlp.graph, x, nlp.ninds, nlp.x_index_map, nlp.optinodes)
end
function grad!(nlp::GraphModel, x::AbstractVector, f::AbstractVector)
    return eval_objective_gradient(nlp.graph, f, x, nlp.ninds, nlp.optinodes)
end
function cons!(nlp::GraphModel, x::AbstractVector, c::AbstractVector)
    return eval_constraint(
    nlp.graph,
    c,
    x,
    nlp.ninds,
    nlp.minds,
    nlp.pinds,
    nlp.x_index_map,
    nlp.optinodes,
    nlp.linkedges
    )
end
function hess_coord!(nlp::GraphModel, x::AbstractVector, l::AbstractVector, hess::AbstractVector; obj_weight=1.)
    eval_hessian_lagrangian(
        nlp.graph,hess,x,obj_weight,l,nlp.ninds,nlp.minds,nlp.nnzs_hess_inds,nlp.optinodes
    )
end
function jac_coord!(nlp::GraphModel,x::AbstractVector,jac::AbstractVector)
    eval_constraint_jacobian(
        nlp.graph,jac,x,nlp.ninds,nlp.minds,nlp.nnzs_jac_inds,nlp.nnzs_link_jac_inds,
        nlp.optinodes,nlp.linkedges,
    )
end
function hess_structure!(nlp::GraphModel, I::AbstractVector{T}, J::AbstractVector{T}) where T
    hessian_lagrangian_structure(
        nlp.graph,I,J,nlp.ninds,nlp.nnzs_hess_inds,nlp.optinodes,
    )
end
function jac_structure!(nlp::GraphModel, I::AbstractVector{T}, J::AbstractVector{T}) where T
    jacobian_structure(
        nlp.graph, I, J, nlp.ninds,nlp.minds,nlp.pinds,nlp.nnzs_jac_inds,nlp.nnzs_link_jac_inds,
        nlp.x_index_map,nlp.g_index_map,nlp.optinodes,nlp.linkedges,
    )
end