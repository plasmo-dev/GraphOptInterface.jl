struct GraphModel{T} <: AbstractNLPModel{T,Vector{T}}
    ninds::Vector{UnitRange{Int}}
    minds::Vector{UnitRange{Int}}
    pinds::Vector{UnitRange{Int}}
    nnzs_jac_inds::Vector{UnitRange{Int}}
    nnzs_hess_inds::Vector{UnitRange{Int}}
    nnzs_link_jac_inds::Vector{UnitRange{Int}}

    x_index_map::Dict
    g_index_map::Dict

    optinodes::Vector{OptiNode}
    linkedges::Vector{OptiEdge}

    meta::NLPModelMeta{T, Vector{T}}
    graph::OptiGraph
    counters::MadNLP.NLPModels.Counters
    ext::Dict{Symbol,Any}
end

# MOI interface functions
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

function GraphModel(graph::OptiGraph)
    optinodes = all_nodes(graph)
    linkedges = all_edges(graph)

    for optinode in optinodes
        num_variables(optinode) == 0 && error("MadNLP does not support empty nodes. You will need to create your optigraph without empty optinodes.")
    end

    #create a MadNLP optimizer on each optinode
    moi_models = Vector{MadNLP.MOIModel}(undef,length(optinodes))
    @blas_safe_threads for k=1:length(optinodes)
        optinode = optinodes[k]
        # set constructor; optimizer not yet attached
        set_optimizer(optinode, MadNLP.Optimizer)
        # Initialize NLP evaluators on each node
        # JuMP does something like this where  the evaluator is created for each `optimize!` call
        jump_model = Plasmo.jump_model(optinode)
        nonlinear_model = JuMP.nonlinear_model(jump_model)
        if nonlinear_model !== nothing
            evaluator = MOI.Nonlinear.Evaluator(
                nonlinear_model,
                MOI.Nonlinear.SparseReverseMode(),
                JuMP.index.(JuMP.all_variables(jump_model)),
            )
            MOI.set(jump_model, MOI.NLPBlock(), MOI.NLPBlockData(evaluator))
            # TODO: check whether we still need to do this
            empty!(optinode.nlp_duals)
        end

        # attach optimizer on node. this populates the madnlp optimizer
        MOIU.attach_optimizer(jump_model)
        madnlp_optimizer = inner_optimizer(optinode)
        madnlp_optimizer.sense = MOI.MIN_SENSE

        # initializes each optinode evaluator
        moi_models[k] = MadNLP.MOIModel(madnlp_optimizer)
    end

    # calculate dimensions
    K = length(optinodes)
    ns= [num_variables(optinode) for optinode in optinodes]
    n = sum(ns)
    ns_cumsum = cumsum(ns)
    ms= [num_constraints(optinode) for optinode in optinodes]
    ms_cumsum = cumsum(ms)
    m = sum(ms)

    nlps = [moi.model for moi in moi_models]

    # hessian nonzeros
    nnzs_hess = [model.meta.nnzh for model in moi_models]
    nnzs_hess_cumsum = cumsum(nnzs_hess)
    nnz_hess = sum(nnzs_hess)

    # jacobian nonzeros
    nnzs_jac = [model.meta.nnzj for model in moi_models]
    nnzs_jac_cumsum = cumsum(nnzs_jac)
    nnz_jac = sum(nnzs_jac)

    # link jacobian nonzeros
    nnzs_link_jac = [get_nnz_link_jac(linkedge) for linkedge in linkedges]
    nnzs_link_jac_cumsum = cumsum(nnzs_link_jac)
    nnz_link_jac = isempty(nnzs_link_jac) ? 0 : sum(nnzs_link_jac)

    # map nodes to variable, constraint, jacobian, hessian indices
    ninds = [(i==1 ? 0 : ns_cumsum[i-1])+1:ns_cumsum[i] for i=1:K]
    minds = [(i==1 ? 0 : ms_cumsum[i-1])+1:ms_cumsum[i] for i=1:K]
    nnzs_hess_inds = [(i==1 ? 0 : nnzs_hess_cumsum[i-1])+1:nnzs_hess_cumsum[i] for i=1:K]
    nnzs_jac_inds = [(i==1 ? 0 : nnzs_jac_cumsum[i-1])+1:nnzs_jac_cumsum[i] for i=1:K]

    # linking information
    Q = length(linkedges)
    ps= [Plasmo.num_linkconstraints(modeledge) for modeledge in linkedges]
    ps_cumsum =  cumsum(ps)
    p = sum(ps)
    pinds = [(i==1 ? m : m+ps_cumsum[i-1])+1:m+ps_cumsum[i] for i=1:Q]
    nnzs_link_jac_inds =
        [(i==1 ? nnz_jac : nnz_jac+nnzs_link_jac_cumsum[i-1])+1: nnz_jac + nnzs_link_jac_cumsum[i] for i=1:Q]

    # primals, lower, upper bounds
    x = Vector{Float64}(undef,n)
    xl = Vector{Float64}(undef,n)
    xu = Vector{Float64}(undef,n)

    # duals, constraints lower & upper bounds
    l = Vector{Float64}(undef,m+p)
    gl = Vector{Float64}(undef,m+p)
    gu = Vector{Float64}(undef,m+p)

    # set start values for variables and constraints
    @blas_safe_threads for k=1:K
        model = moi_models[k]
        x[ninds[k]] .= model.meta.x0
        xl[ninds[k]] .= model.meta.lvar
        xu[ninds[k]] .= model.meta.uvar
        l[minds[k]] .= model.meta.y0
        gl[minds[k]] .= model.meta.lcon
        gu[minds[k]] .= model.meta.ucon
    end

    # set link constraint start values
    for q=1:Q
        set_g_link!(linkedges[q], view(l,pinds[q]), view(gl,pinds[q]), view(gu,pinds[q]))
    end

    # map variables to graph indices. used for link jacobians
    x_index_map = Dict()
    for k = 1:K
        optinode = optinodes[k]
        for var in all_variables(optinode)
            x_index_map[var] = ninds[k][JuMP.index(var).value]
        end
    end

    # map variables to graph indices. used for link jacobians
    g_index_map = Dict()
    cnt = 1
    for linkref in Plasmo.all_linkconstraints(graph)
        con = JuMP.constraint_object(linkref)
        g_index_map[con] = m + cnt
        cnt += 1
    end

    jac_constant = true
    hess_constant = true
    for moi in moi_models
        jac_constant = jac_constant & moi.model.options[:jacobian_constant]
        hess_constant = hess_constant & moi.model.options[:hessian_constant]
    end

    ext = Dict{Symbol,Any}(
        :n=>n, :m=>m,:p=>p,
        :ninds=>ninds, :minds=>minds, :pinds=>pinds,
        :linkedges=>linkedges, :jac_constant=>jac_constant, :hess_constant=>hess_constant
    )

    return GraphModel(
        ninds,
        minds,
        pinds,
        nnzs_jac_inds,
        nnzs_hess_inds,
        nnzs_link_jac_inds,
        x_index_map,
        g_index_map,
        optinodes,
        linkedges,
        NLPModelMeta(
            n,
            ncon = m+p,
            x0 = x,
            lvar = xl,
            uvar = xu,
            y0 = l,
            lcon = gl,
            ucon = gu,
            nnzj = nnz_jac + nnz_link_jac,
            nnzh = nnz_hess,
            minimize = true #graph.objective_sense == MOI.MIN_SENSE
        ),
        graph,
        MadNLP.NLPModels.Counters(),
        ext
    )
end

function get_partition(graph::OptiGraph, nlp::GraphModel)
    n = nlp.ext[:n]
    m = nlp.ext[:m]
    p = nlp.ext[:p]

    ninds = nlp.ninds
    minds = nlp.minds
    pinds = nlp.pinds

    ind_ineq = findall(get_lcon(nlp).!=get_ucon(nlp))
    l = length(ind_ineq)

    part = Vector{Int}(undef,n+m+l+p)

    for k=1:length(ninds)
        part[ninds[k]].=k
    end
    for k=1:length(minds)
        part[minds[k].+n.+l].=k
    end
    cnt = 0

    for linkedge in nlp.ext[:linkedges]
        for (ind,con) in linkedge.linkconstraints
            cnt+=1
            attached_node_idx = graph.node_idx_map[con.attached_node]
            part[n+l+m+cnt] = attached_node_idx != nothing ? attached_node_idx : error("All the link constraints need to be attached to a node")
        end
    end

    cnt = 0
    for q in ind_ineq
        cnt+=1
        part[n+cnt] = part[n+l+q]
    end

    return part
end

function optimize!(graph::OptiGraph; kwargs...)
    options = Dict{Symbol,Any}(kwargs)
    gm = GraphModel(graph)

    # run either schwarz or schur decompositions
    # TODO: think of a more systematic way to specify decomposition-based linear solvers
    K = num_all_nodes(graph)
    if (haskey(kwargs, :schwarz_custom_partition) && kwargs[:schwarz_custom_partition])
        part = get_partition(graph, gm)
        options[:schwarz_part] = part
        options[:schwarz_num_parts] = num_all_nodes(graph)
    elseif (haskey(kwargs, :schur_custom_partition) && kwargs[:schur_custom_partition])
        part = get_partition(graph, gm)
        part[part.>K].=0 # will any partition indices be greater than K?
        options[:schur_part] = part
        options[:schur_num_parts] = K
    end

    options[:jacobian_constant] = gm.ext[:jac_constant]
    options[:hessian_constant] = gm.ext[:hess_constant]

    ips = MadNLP.MadNLPSolver(gm; options...)
    result = MadNLP.solve!(ips)

    # setup a MadNLP optimizer the optigraph can use to reference solution values
    madnlp = MadNLP.Optimizer()
    madnlp.solver = ips
    madnlp.result = result
    madnlp.solve_time = ips.cnt.total_time
    madnlp.solve_iterations = ips.cnt.k
    graph.moi_backend = madnlp

    # populate solution
    nlps = [inner_optimizer(node) for node in gm.optinodes]
    @blas_safe_threads for k=1:K
        nlps[k].result = MadNLP.MadNLPExecutionStats(
            ips.status,
            result.solution[gm.ninds[k]],
            ips.obj_val,
            result.constraints[gm.minds[k]],
            ips.inf_du,
            ips.inf_pr,
            result.multipliers[gm.minds[k]],
            result.multipliers_L[gm.ninds[k]],
            result.multipliers_U[gm.ninds[k]],
            ips.cnt.k,
            ips.nlp.counters,
            ips.cnt.total_time
        )
        # TODO: quick hack to specify to JuMP that the
        # model is not dirty (so we do not run in `OptimizeNotCalled`
        # exception).
        gm.optinodes[k].model.is_model_dirty = false
    end
end