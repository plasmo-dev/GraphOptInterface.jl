# IDEA: Plasmo.jl can take this kind of approach for MOI vs BMOI backend optimizers

"""
    StochasticProgramOptimizer
Wrapper type around both the optimizer_constructor provided to a stochastic program and the resulting optimizer object. Used to conviently distinguish between standard MOI optimizers and structure-exploiting optimizers when instantiating the stochastic program.
"""
mutable struct StochasticProgramOptimizer{Opt <: StochasticProgramOptimizerType}
    optimizer_constructor
    optimizer::Opt

    function StochasticProgramOptimizer(::Nothing)
        universal_fallback = MOIU.UniversalFallback(MOIU.Model{Float64}())
        caching_optimizer = MOIU.CachingOptimizer(universal_fallback, MOIU.AUTOMATIC)
        return new{StochasticProgramOptimizerType}(nothing, caching_optimizer)
    end

    function StochasticProgramOptimizer(optimizer_constructor, optimizer::MOI.AbstractOptimizer)
        universal_fallback = MOIU.UniversalFallback(MOIU.Model{Float64}())
        caching_optimizer = MOIU.CachingOptimizer(universal_fallback, MOIU.AUTOMATIC)
        MOIU.reset_optimizer(caching_optimizer, optimizer)
        Opt = MOI.AbstractOptimizer
        return new{Opt}(optimizer_constructor, caching_optimizer)
    end

    function StochasticProgramOptimizer(optimizer_constructor, optimizer::AbstractStructuredOptimizer)
        Opt = AbstractStructuredOptimizer
        return new{Opt}(optimizer_constructor, optimizer)
    end
end

function StochasticProgramOptimizer(optimizer_constructor)
    optimizer = MOI.instantiate(optimizer_constructor; with_bridge_type = bridge_type(optimizer_constructor))
    return StochasticProgramOptimizer(optimizer_constructor, optimizer)
end

bridge_type(optimizer::Type{<:AbstractStructuredOptimizer}) = nothing
bridge_type(optimizer::Type{<:AbstractSampledOptimizer}) = nothing
bridge_type(optimizer::Type{<:MOI.AbstractOptimizer}) = Float64
bridge_type(optimizer::MOI.OptimizerWithAttributes) = bridge_type(optimizer.optimizer_constructor)
bridge_type(optimizer::Function) = bridge_type(typeof(optimizer()))