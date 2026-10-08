module ArgoJSOSolversExt

import Argo
import JSOSolvers

const set_value! = JSOSolvers.SolverParameters.set_value!

function Argo.implemented_methods(::Type{<:JSOSolvers.FoSolver})
    return [Argo.MethodImplementation(:gradient_descent, _configure_fo!)]
end

# R2 starts with the step `α0`, accepts a step if the ratio `ρ` between the
# actual and predicted decrease is at least `η1`, and then never makes the step
# larger than `αmax`. For an `L`-smooth objective, a step `h < 2/L` gives
# `ρ ≥ 1 - L h / 2 > 0`, so with `α0 = αmax = h` and a tiny `η1`, every step is
# accepted with `α = h`: R2 is then gradient descent with the fixed step `h`.
function _configure_fo!(solver::JSOSolvers.FoSolver{T}, parameters) where {T}
    h = T(parameters[:step_size])
    set_value!(solver.params.αmax, h)
    set_value!(solver.params.η1, eps(T))
    set_value!(solver.params.step_backend, JSOSolvers.r2_step())
    # Monotone: `solve!` reads `M` from the length of the objective memory
    set_value!(solver.params.M, 1)
    resize!(solver.o, 1)
    return (; α0=h)
end

end
