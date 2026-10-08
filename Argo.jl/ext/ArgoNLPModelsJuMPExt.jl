module ArgoNLPModelsJuMPExt

import Argo
import NLPModelsJuMP

const MOI = NLPModelsJuMP.MOI

# The methods are those of the NLPModels solver set with the `"solver"`
# attribute; see `Argo.implemented_methods` for the contract at that level.
function Argo.implemented_methods(optimizer::NLPModelsJuMP.Optimizer)
    constructor = get(optimizer.options, "solver", nothing)
    constructor === nothing && return Argo.MethodImplementation[]
    return map(Argo.implemented_methods(constructor)) do impl
        configure! = (optimizer, parameters) -> begin
            return _configure!(optimizer, constructor, impl, parameters)
        end
        return Argo.MethodImplementation(impl.method, configure!)
    end
end

function _configure!(optimizer, constructor, impl, parameters)
    # The solver is only created in `MOI.copy_to`, once the model is known.
    configured = nlp -> begin
        solver = constructor(nlp)
        for (key, value) in pairs(impl.configure!(solver, parameters))
            MOI.set(optimizer, MOI.RawOptimizerAttribute(string(key)), value)
        end
        return solver
    end
    MOI.set(optimizer, MOI.RawOptimizerAttribute("solver"), configured)
    return
end

end
