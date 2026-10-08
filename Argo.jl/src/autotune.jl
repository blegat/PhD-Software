"""
    MethodImplementation(method::Symbol, configure!)

Declare that a solver implements the catalogue method with id `method`.
`configure!(target, parameters)` sets the solver up so that it runs exactly
this method with the method parameters `parameters`, a `Dict{Symbol,Float64}`
such as `Dict(:step_size => 0.1)` computed by Argo from a certificate.
"""
struct MethodImplementation
    method::Symbol
    configure!::Function
end

"""
    implemented_methods(solver)::Vector{MethodImplementation}

Return the catalogue methods that `solver`, in its current configuration, can
be set up to run. Solver packages add methods to this function to let
[`AutoTune`](@ref) select and tune their algorithm. The `target` given to
`configure!` depends on what `solver` is:

* A MathOptInterface optimizer: `target` is the optimizer, and `configure!`
  sets its attributes.
* The type of a solver following the SolverCore API (`S(nlp)` then
  `SolverCore.solve!(solver, nlp, stats; kwargs...)`): `target` is the
  solver created by `S(nlp)`, which `configure!` may modify, and `configure!`
  returns the extra keyword arguments to pass to `SolverCore.solve!`.
"""
implemented_methods(::Any) = MethodImplementation[]

"""
    AutoTune(optimizer_constructor; accuracy=1e-6, initial_bounds, oracle_costs=Dict())

Return an optimizer constructor, to be used as `JuMP.Model(AutoTune(...))`,
for a meta-optimizer that converts the model with [`minimize`](@ref), ranks
its certificates with [`rank`](@ref) and configures the optimizer created by
`optimizer_constructor` with the parameters of the best certificate whose
method is in [`implemented_methods`](@ref). The `initial_bounds` are only
used to compare the plans; they default to `1` for every initial quantity.

Requires `MathOptInterface` to be loaded.
"""
function AutoTune end
