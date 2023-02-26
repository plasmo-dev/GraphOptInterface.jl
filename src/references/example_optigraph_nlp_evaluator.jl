# NOTE: objective function is summation by default
# x is the full variable vector
# g is the full constraint vector

# NOTE: should we collect all the edges up front, or should we dive into the evaluations hierarchically, through sub-blocks?

# TODO: think of providing a fully working block evaluator

mutable struct OptiGraphNLPEvaluator <: MOI.AbstractNLPEvaluator
    graph::OptiGraph
    optinodes::Vector{OptiNode}

    nlps::Union{Nothing,Vector{MOI.Nonlinear.Evaluator}} 
    has_nlobj
    n       #num variables (columns)
    m       #num constraints (rows)
    #p       #num link constraints (rows)
    ninds   #variable indices for each node
    minds   #row indicies for each node
    #pinds   #link constraint indices
    nnzs_hess_inds
    nnzs_jac_inds
    #nnzs_link_jac_inds
    nnz_hess
    nnz_jac


    # timers
    eval_objective_timer::Float64
    eval_constraint_timer::Float64
    eval_objective_gradient_timer::Float64
    eval_constraint_jacobian_timer::Float64
    eval_hessian_lagrangian_timer::Float64
    function OptiGraphNLPEvaluator(graph::OptiGraph)
        d = new(graph)
        d.eval_objective_timer = 0
        d.eval_constraint_timer = 0
        d.eval_objective_gradient_timer = 0
        d.eval_constraint_jacobian_timer = 0
        d.eval_hessian_lagrangian_timer = 0
        return d
    end
end

#Initialize
function MOI.initialize(d::OptiGraphNLPEvaluator, requested_features::Vector{Symbol})
    graph = d.graph
    optinodes = all_nodes(graph)
    linkedges = all_edges(graph)

    d.optinodes = optinodes
    # d.nlps = Vector{JuMP.NLPEvaluator}(undef, length(optinodes))      #Initialize each optinode with the requested features
    d.nlps = Vector{MOI.Nonlinear.Evaluator}(undef, length(optinodes))  #Initialize each optinode with the requested features
    d.has_nlobj = false

    # TODO: multiple threads
    #@blas_safe_threads for k=1:length(optinodes)
    K = length(optinodes)
    for k in 1:K
        model = jump_model(optinodes[k])
        if JuMP.nonlinear_model(optinodes[k]) == nothing
            #if optinodes[k].nlp_data == nothing
            # JuMP._init_NLP(optinodes[k].model)
            JuMP._init_NLP(model)
        end
        nlp = JuMP.nonlinear_model(optinodes[k])
        # d_node = JuMP.NLPEvaluator(optinodes[k].model)     #Initialize each optinode evaluator
        d_node = JuMP.NLPEvaluator(model)
        MOI.initialize(d_node, requested_features)
        d.nlps[k] = d_node
        #if d_node.has_nlobj
        if nlp.objective != nothing
            d.has_nlobj = true
        end
    end

    #num variables in optigraph
    ns = [num_variables(optinode) for optinode in optinodes]
    n = sum(ns)
    ns_cumsum = cumsum(ns)

    #num constraints NOTE: Should this just be NL constraints? Depends on evaluator mode
    ms = [num_nonlinear_constraints(optinode) for optinode in optinodes]
    m = sum(ms)
    ms_cumsum = cumsum(ms)

    #hessian nonzeros: This grabs quadratic terms if we have a nonlinear objective function on any node
    if d.has_nlobj
        #TODO: grab quadratic constraints too
        nnzs_hess = [_get_nnz_hess_quad(d.optinodes[k], d.nlps[k]) for k in 1:K]
    else
        nnzs_hess = [_get_nnz_hess(d.nlps[k]) for k in 1:K]
    end

    #nnzs_hess = [_get_nnz_hess(d.nlps[k]) for k = 1:K]
    nnzs_hess_cumsum = cumsum(nnzs_hess)
    d.nnz_hess = sum(nnzs_hess)

    #jacobian nonzeros
    nnzs_jac = [_get_nnz_jac(d.nlps[k]) for k in 1:K]
    nnzs_jac_cumsum = cumsum(nnzs_jac)
    d.nnz_jac = sum(nnzs_jac)

    # link jacobian nonzeros (These wouldn't be returned in the JuMP NLP Evaluator)
    # nnzs_link_jac = [get_nnz_link_jac(linkedge) for linkedge in linkedges]
    # nnzs_link_jac_cumsum = cumsum(nnzs_link_jac)
    # nnz_link_jac = isempty(nnzs_link_jac) ? 0 : sum(nnzs_link_jac)

    #variable indices and constraint indices
    ninds = [((i == 1 ? 0 : ns_cumsum[i - 1]) + 1):ns_cumsum[i] for i in 1:K]
    minds = [((i == 1 ? 0 : ms_cumsum[i - 1]) + 1):ms_cumsum[i] for i in 1:K]

    #nonzero indices for hessian and jacobian
    nnzs_hess_inds = [
        ((i == 1 ? 0 : nnzs_hess_cumsum[i - 1]) + 1):nnzs_hess_cumsum[i] for i in 1:K
    ]
    nnzs_jac_inds = [
        ((i == 1 ? 0 : nnzs_jac_cumsum[i - 1]) + 1):nnzs_jac_cumsum[i] for i in 1:K
    ]

    # #num linkedges
    # Q = length(linkedges)
    # ps= [num_linkconstraints(optiedge) for optiedge in linkedges]
    # ps_cumsum =  cumsum(ps)
    # p = sum(ps)
    # pinds = [(i==1 ? m : m+ps_cumsum[i-1])+1:m+ps_cumsum[i] for i=1:Q]

    #link jacobian nonzero indices
    #nnzs_link_jac_inds = [(i==1 ? nnz_jac : nnz_jac+nnzs_link_jac_cumsum[i-1])+1: nnz_jac + nnzs_link_jac_cumsum[i] for i=1:Q]

    d.n = n
    d.m = m
    #d.p = p
    d.ninds = ninds
    d.minds = minds
    #d.pinds = pinds
    d.nnzs_hess_inds = nnzs_hess_inds
    return d.nnzs_jac_inds = nnzs_jac_inds
    #d.nnzs_link_jac_inds = d.nnzs_link_jac_inds

end

###########################################################################################
"""
    Evaluator(
        model::Model,
        backend::AbstractAutomaticDifferentiation,
        ordered_variables::Vector{MOI.VariableIndex},
    )
Create `Evaluator`, a subtype of `MOI.AbstractNLPEvaluator`, from `Model`.
"""
mutable struct BlockNLPEvaluator{B} <: MOI.AbstractNLPEvaluator
    # The internal datastructure.
    blk_data:BlockData
    models::Dict{Edge,EdgeModel}
    
    # The abstract-differentiation backend
    backend::B

    # ordered_constraints is needed because `OrderedDict` doesn't support
    # looking up a key by the linear index.
    ordered_constraints::Vector{ConstraintIndex}

    # Storage for the NLPBlockDual, so that we can query the dual of individual
    # constraints without needing to query the full vector each time.
    constraint_dual::Vector{Float64}

    # Timers
    eval_objective_timer::Float64
    eval_constraint_timer::Float64
    eval_objective_gradient_timer::Float64
    eval_constraint_jacobian_timer::Float64
    eval_hessian_lagrangian_timer::Float64

    function BlockNLPEvaluator(
        blk_model,
        backend::B=nothing,
    ) where {B<:Union{Nothing,MOI.AbstractNLPEvaluator}}
        return new{B}(
            model,
            backend,
            ConstraintIndex[],
            Float64[],
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )
    end
end

function BlockNLPEvaluator()
end

### Eval_F_CB

function MOI.eval_objective(evaluator::BlockNLPEvaluator, x)
	# any edge could contribute to the objective function
	edges = get_all_edges(evaluator.blk)
	obj = Threads.Atomic{Float64}(0.)
	for edge in edges
        edge_model = evaluator.models[edge]
        ninds = get_col_inds(blk, edge)
        Threads.atomic_add!(obj, MOI.eval_objective(edge.model, view(x, ninds)))
    end
    return obj.value
end

### Eval_Grad_F_CB

function MOI.eval_objective_gradient(evaluator::BlockNLPEvaluator, f, x)
    blk = evaluator.block_data
    edges = get_all_edges(blk)
    for edge in edges
        edge_model = evaluator.models[edge]
    	ninds = get_col_inds(blk, edge)
        MOI.eval_objective_gradient(edge.model, view(f,ninds), view(x,ninds))
    end
    return nothing
end

### Eval_G_CB

function eval_constraint(evaluator::BlockNLPEvaluator, c, x)
    #@blas_safe_threads 
	for edge in get_all_edges(blk)
		ninds = get_col_inds(blk, edge)
		minds = get_row_inds(blk, edge)
		MOI.eval_constraint(edge, view(c, minds), view(x, ninds))
	end
    return nothing
end

### Eval_H_CB

function MOI.hessian_lagrangian_structure(evaluator::BlockNLPEvaluator)
    
	nnzs_hess_inds = d.nnzs_hess_inds
    nnz_hess = d.nnz_hess
    nodes = d.optinodes

    I = Vector{Int64}(undef, d.nnz_hess)
    J = Vector{Int64}(undef, d.nnz_hess)

    #@blas_safe_threads for k=1:length(optinodes)
	for edge in all_edges(blk)
        
		ninds = get_col_inds(edge)
		nnzs_hess_inds = get_hess_inds(edge)

		isempty(nnzs_hess_inds) && continue

		offset = ninds[1]-1

        II = view(I, nnzs_hess_inds[k])
        JJ = view(J, nnzs_hess_inds[k])


        madnlp_optimizer = inner_optimizer(optinodes[k])
        cnt = 1
        for (row, col) in  MOI.hessian_lagrangian_structure(madnlp_optimizer)
            II[cnt], JJ[cnt] = row, col
            cnt += 1
        end
        II.+= offset
        JJ.+= offset
    end
    return nothing
end



# reference
# function MOI.hessian_lagrangian_structure(d::OptiGraphNLPEvaluator)
#     nnzs_hess_inds = d.nnzs_hess_inds
#     nnz_hess = d.nnz_hess
#     nodes = d.optinodes

#     I = Vector{Int64}(undef, d.nnz_hess)
#     J = Vector{Int64}(undef, d.nnz_hess)

#     # @blas_safe_threads for k=1:length(optinodes)
#     for k = 1:length(d.nlps)
#         isempty(nnzs_hess_inds[k]) && continue
#         offset = d.ninds[k][1] - 1
#         II = view(I, nnzs_hess_inds[k])
#         JJ = view(J, nnzs_hess_inds[k])
#         #if the optigraph has a nonlinear objective on any node, we need to treat quadratic objectives as nonlinear
#         if d.has_nlobj
#             _hessian_lagrangian_structure_quad(nodes[k], d.nlps[k], II, JJ)
#         else #just run the normal JuMP functions
#             _hessian_lagrangian_structure(d.nlps[k], II, JJ)
#         end
#         II.+= offset
#         JJ.+= offset
#     end
#     hessian_sparsity = collect(zip(I, J)) # return Tuple{Int64,Int64}[]
#     return hessian_sparsity
# end

# # jacobian structure
# function jacobian_structure(linkedge::OptiEdge, I, J, ninds, x_index_map, g_index_map)
#     offset=1
#     for linkcon in Plasmo.linkconstraints(linkedge)
#         offset += jacobian_structure(linkcon, I, J, ninds, x_index_map, g_index_map, offset)
#     end
# end
# function jacobian_structure(linkcon, I, J, ninds, x_index_map, g_index_map, offset)
#     cnt = 0
#     for var in get_vars(linkcon)
#         I[offset+cnt] = g_index_map[linkcon]
#         J[offset+cnt] = x_index_map[var]
#         cnt += 1
#     end
#     return cnt
# end
# function jacobian_structure(
#     graph::OptiGraph, I, J, ninds, minds, pinds,
#     nnzs_jac_inds, nnzs_link_jac_inds, x_index_map, g_index_map,
#     optinodes, linkedges
# )
#     # evaluate optinode jacobians
#     @blas_safe_threads for k=1:length(optinodes)
#         isempty(nnzs_jac_inds[k]) && continue
#         offset_i = minds[k][1]-1
#         offset_j = ninds[k][1]-1
#         II = view(I,nnzs_jac_inds[k])
#         JJ = view(J,nnzs_jac_inds[k])
#         madnlp_optimizer = inner_optimizer(optinodes[k])
#         cnt = 1
#         for (row, col) in MOI.jacobian_structure(madnlp_optimizer)
#             II[cnt], JJ[cnt] = row, col
#             cnt += 1
#         end
#         II.+= offset_i
#         JJ.+= offset_j
#     end
#     # evaluate optiedge jacobians
#     @blas_safe_threads for q=1:length(linkedges)
#         isempty(nnzs_link_jac_inds[q]) && continue
#         II = view(I, nnzs_link_jac_inds[q])
#         JJ = view(J, nnzs_link_jac_inds[q])
#         jacobian_structure(
#             linkedges[q], II, JJ, ninds, x_index_map, g_index_map
#         )
#     end
#     return nothing
# end


