### Utility funcs

function _num_variables(node::GOI.Node)
    return MOI.get(node, MOI.NumberOfVariables())
end

function _num_constraints(edge::GOI.Edge)
    n_con = 0
    for (F,S) in MOI.get(edge, MOI.ListOfConstraintTypesPresent())
        n_con += MOI.get(edge, MOI.NumberOfConstraints{F,S}())
    end
    if edge.nonlinear_model != nothing
        n_con += length(edge.nonlinear_model.constraints)
    end
    return n_con
end

function _num_variables(block::GOI.Block)
    return MOI.get(block, MOI.NumberOfVariables())
end

function _num_constraints(block::GOI.Block)
	return sum(num_constraints(edge) for edge in GOI.all_edges(block))
end