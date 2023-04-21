### Utility funcs

function _num_variables(node::Node)
    return MOI.get(node, MOI.NumberOfVariables())
end

function _num_constraints(edge::Edge)
    n_con = 0
    for (F,S) in MOI.get(edge, MOI.ListOfConstraintTypesPresent())
        n_con += MOI.get(edge, MOI.NumberOfConstraints{F,S}())
    end
    nlp_block = MOI.get(edge, MOI.NLPBlock())
    n_con += length(nlp_block.constraint_bounds)
    return n_con
end

function _num_variables(block::Block)
    return MOI.get(block, MOI.NumberOfVariables())
end

function _num_constraints(block::Block)
	return sum(num_constraints(edge) for edge in all_edges(block))
end