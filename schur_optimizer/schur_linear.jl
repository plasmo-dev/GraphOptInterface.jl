const INPUT_MATRIX_TYPE = :csc

import MadNLP: AbstractOptions, AbstractLinearSolver, MadNLPLogger, SubVector, 
default_linear_solver, default_dense_solver, default_options

Base.@kwdef mutable struct SchurOptions <: AbstractOptions
    partition::Vector{Int}=Vector{Int}()
    subproblem_solver::Type{default_linear_solver()}=default_linear_solver()
    subproblem_solver_options::AbstractOptions=default_options(default_linear_solver())
    dense_solver::Type{default_dense_solver()}=default_dense_solver()
    dense_solver_options::AbstractOptions=default_options(default_dense_solver())
end

mutable struct SolverWorker{T,ST<:AbstractLinearSolver}
    V::Vector{Int}
    V_0_nz::Vector{Int}
    csc::SparseMatrixCSC{T,Int32}
    csc_view::SubVector{T}
    compl::SparseMatrixCSC{T,Int32}
    compl_view::SubVector{T}
    M::ST
    w::Vector{T}
end

mutable struct SchurLinearSolver{T} <: AbstractLinearSolver{T}
    csc::SparseMatrixCSC{T,Int32}
    inds::Vector{Int}
    partitions::Vector{Int}
    num_partitions::Int

    schur::Matrix{Float64}
    colors
    fact # dense solver

    V_0::Vector{Int}
    csc_0::SparseMatrixCSC{T,Int32}
    csc_0_view::SubVector{T}
    w_0::Vector{Float64}

    sws::Vector{SolverWorker}

    opt::SchurOptions
    logger::MadNLPLogger
end

function SchurLinearSolver(
	csc::SparseMatrixCSC{T};
    opt=SchurOptions(),
	logger=MadNLPLogger()
) where T

    if string(opt.schur_subproblem_solver) == "MadNLP.Mumps"
        @warn(logger,"When Mumps is used as a subproblem solver, Schur is run in serial.")
        @warn(logger,"To use parallelized Schur, use Ma27 or Ma57.")
    end

    # non-zeros in KKT
    inds = collect(1:nnz(csc))
    num_partitions = unique(partition)

    # first stage indices
    V_0   = findall(partition.==0)
    colors = get_colors(length(V_0), num_partitions)

    # KKT first-stage
    csc_0, csc_0_view = get_cscsy_view(csc, V_0, inds=inds)
    schur_matrix = Matrix{T}(undef, length(V_0), length(V_0))

    # first-stage primal-dual step
    w_0 = Vector{Float64}(undef, length(V_0))

    # solver-workers
    sws = Vector{SolverWorker{opt.schur_subproblem_solver}}(undef, num_partitions)

    Threads.@threads for k=1:num_partitions
        sws[k] = SolverWorker{opt.schur_subproblem_solver}(
            partition, 
            V_0, 
            csc,
            inds, 
            k,
            opt.subproblem_solver,
            opt.subproblem_solver_options,
            logger
        )
    end

    # dense system solver
    fact = opt.dense_solver{T}(schur_matrix; opt=opt.dense_solver_options)

    return SchurLinearSolver{T}(
        csc, 
        inds,
        partitions,
        num_partitions,
        schur_matrix,
        colors,
        fact,
        V_0,
        csc_0,
        csc_0_view,
        w_0,
        sws,
        opt,
        logger
    )
end

get_colors(n0, K) = [findall((x)->mod(x-1,K)+1==k,1:n0) for k=1:K]

function SolverWorker(
    partition::Vector{Int}, 
    V_0::Vector{Int},
    csc::SparseMatrixCSC{T},
    inds::Vector{Int},
    k::Int,
    subproblem_solver::Type{ST},
    options::AbstractOptions,
    logger::MadNLPLogger
) where T where ST <: AbstractLinearSolver

    # partition indices
    V = findall(partition.==k)

    # TODO: document what these are
    csc_k, csc_k_view = get_cscsy_view(csc, V, inds=inds)
    compl, compl_view = get_csc_view(csc, V, V_0, inds=inds)
    V_0_nz = findnz(compl.colptr)

    # sub-problem linear solver
    solver = subproblem_solver{T}(csc_k; opt=options, logger=logger)
    
    # sub-problem step
    w = Vector{T}(undef,csc_k.n)

    return SolverWorker(V, V_0_nz, csc_k, csc_k_view, compl, compl_view, solver, w)
end

function findnz(colptr)
    nz = Int[]
    for j=1:length(colptr)-1
        colptr[j]==colptr[j+1] || push!(nz,j)
    end
    return nz
end

function factorize!(M::SchurLinearSolver)
    M.schur .= 0.
    M.csc_0.nzval .= M.csc_0_view
    M.schur .= M.csc_0
    Threads.@threads for sw in M.sws
        sw.csc.nzval .= sw.csc_view
        sw.compl.nzval .= sw.compl_view
        factorize!(sw.M)
    end

    # NOTE: asynchronous multithreading doesn't work here
    for q = 1:length(M.colors)
        Threads.@threads for k = 1:length(M.sws)
            for j = M.colors[mod(q+k-1, length(M.sws))+1] # each subproblem works on a different color
                factorize_worker!(j, M.sws[k], M.schur)
            end
        end
    end
    factorize!(M.fact)
    return M
end

function factorize_worker!(j,sw,schur)
    j in sw.V_0_nz || return
    sw.w.= view(sw.compl, :, j)
    solve!(sw.M, sw.w)
    mul!(view(schur, :, j), sw.compl', sw.w, -1., 1.)
end


function solve!(M::SchurLinearSolver, x::AbstractVector{T}) where T
    M.w_0 .= view(x, M.V_0)
    Threads.@threads for sw in M.sws
        sw.w.=view(x, sw.V)
        solve!(sw.M, sw.w)
    end
    for sw in M.sws
        mul!(M.w_0, sw.compl', sw.w, -1., 1.)
    end
    solve!(M.fact, M.w_0)
    view(x, M.V_0) .= M.w_0
    Threads.@threads for sw in M.sws
        x_view = view(x,sw.V)
        sw.w.= x_view
        mul!(sw.w,sw.compl,M.w_0,1.,1.)
        solve!(sw.M,sw.w)
        x_view.=sw.w
    end
    return x
end

is_inertia(M::SchurLinearSolver) = is_inertia(M.fact) && is_inertia(M.sws[1].M)

function inertia(M::SchurLinearSolver)
    numpos,numzero,numneg = inertia(M.fact)
    for k=1:M.opt.schur_num_parts
        _numpos,_numzero,_numneg =  inertia(M.sws[k].M)
        numpos += _numpos
        numzero += _numzero
        numneg += _numneg
    end
    return (numpos,numzero,numneg)
end

function improve!(M::SchurLinearSolver)
    for sw in M.sws
        improve!(sw.M) || return false
    end
    return true
end

function introduce(M::SchurLinearSolver)
    sw = M.sws[1]
    return "schur equipped with "*introduce(sw.M)
end

# end # module