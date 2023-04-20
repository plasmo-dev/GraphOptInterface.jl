struct NodeScalarAffineTerm{T}
	coefficient::T
	variable::MOI.VariableIndex
	node::HyperNode
end

function ScalarAffineTerm(coeff)