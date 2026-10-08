module ArgoMathOptInterfaceExt

using LinearAlgebra

import Argo
import MathOptInterface as MOI

using Argo:
    Property, Term, Variable, convex, lipschitz, oracle, smooth, strongly_convex

const _Quadratic = MOI.ScalarQuadraticFunction{Float64}

"""
Conversion state: the Argo variable standing for the vector of all MOI
variables, and the coordinate of each MOI variable in that vector.
"""
struct _Context
    variable::Variable
    index::Dict{MOI.VariableIndex,Int}
end

_dimension(context::_Context) = length(context.index)

# Univariate atoms. Each entry gives the facts of `t ↦ op(t)` on the reals;
# only facts that hold globally are declared since Argo certifies with them.
function _atom(op::Symbol)
    if op in (:exp, :exp2, :expm1)
        return Property[convex()], (:value, :gradient)
    elseif op === :abs
        return Property[convex(), lipschitz(1)], (:value, :subgradient, :prox)
    elseif op === :abs2
        return Property[convex(), smooth(2), strongly_convex(2)],
        (:value, :gradient, :prox)
    elseif op in (:sin, :cos)
        return Property[smooth(1), lipschitz(1)], (:value, :gradient)
    elseif op in (:atan, :tanh)
        return Property[lipschitz(1)], (:value, :gradient)
    elseif op in (:log, :log1p, :log2, :log10, :sqrt, :cbrt, :inv, :tan)
        return Property[], (:value, :gradient)
    end
    return nothing
end

function _atom_term(op::Symbol)
    atom = _atom(op)
    atom === nothing &&
        throw(ArgumentError("unsupported nonlinear operator `$op` for Argo"))
    properties, oracles = atom
    return Argo.term(op; properties, oracles=[oracle(name) for name in oracles])
end

# `t ↦ t^p` for a constant exponent `p`.
function _power_term(p::Real)
    if p == 2
        return _atom_term(:abs2)
    elseif isinteger(p) && p > 0 && iseven(p)
        return Argo.term(
            Symbol(:pow, Int(p));
            properties=Property[convex()],
            oracles=[oracle(:value), oracle(:gradient)],
        )
    end
    return Argo.term(:pow; oracles=[oracle(:value), oracle(:gradient)])
end

# Quadratic (or affine, or constant) expressions are kept as MOI functions so
# that they can be merged and turned into a single data-backed Argo atom.

_quadratic(f::Real) = convert(_Quadratic, Float64(f))
_quadratic(f::MOI.VariableIndex) = convert(_Quadratic, f)
_quadratic(f::MOI.ScalarAffineFunction) = convert(_Quadratic, f)
_quadratic(f::MOI.ScalarQuadraticFunction) = convert(_Quadratic, f)

function _quadratic(f::MOI.ScalarNonlinearFunction)
    args = f.args
    if f.head === :+
        parts = map(_quadratic, args)
        any(isnothing, parts) && return nothing
        return reduce(_add, parts; init=_quadratic(0))
    elseif f.head === :- && length(args) == 1
        part = _quadratic(only(args))
        return part === nothing ? nothing : MOI.Utilities.operate(-, Float64, part)
    elseif f.head === :- && length(args) == 2
        left, right = _quadratic(args[1]), _quadratic(args[2])
        (left === nothing || right === nothing) && return nothing
        return MOI.Utilities.operate(-, Float64, left, right)
    elseif f.head === :*
        parts = map(_quadratic, args)
        any(isnothing, parts) && return nothing
        return _product(parts)
    elseif f.head === :/ && length(args) == 2
        denominator = _constant(args[2])
        denominator === nothing && return nothing
        part = _quadratic(args[1])
        part === nothing && return nothing
        return MOI.Utilities.operate(/, Float64, part, denominator)
    elseif f.head === :^ && length(args) == 2
        p = _constant(args[2])
        (p === nothing || !isinteger(p) || !(0 <= p <= 2)) && return nothing
        base = _quadratic(args[1])
        base === nothing && return nothing
        return _product(fill(base, Int(p)))
    end
    return nothing
end

_quadratic(::Any) = nothing

_add(f::_Quadratic, g::_Quadratic) = MOI.Utilities.operate(+, Float64, f, g)

_degree(f::_Quadratic) = isempty(f.quadratic_terms) ? (isempty(f.affine_terms) ? 0 : 1) : 2

function _product(parts::Vector{_Quadratic})
    result = _quadratic(1)
    for part in parts
        _degree(result) + _degree(part) > 2 && return nothing
        result = if _degree(result) == 0
            MOI.Utilities.operate(*, Float64, MOI.constant(result), part)
        elseif _degree(part) == 0
            MOI.Utilities.operate(*, Float64, MOI.constant(part), result)
        else
            MOI.Utilities.operate(*, Float64, _affine(result), _affine(part))
        end
    end
    return result
end

function _affine(f::_Quadratic)
    return MOI.ScalarAffineFunction(f.affine_terms, f.constant)
end

function _constant(f)
    part = _quadratic(f)
    (part === nothing || _degree(part) > 0) && return nothing
    return MOI.constant(part)
end

function _leaf(f::_Quadratic, context::_Context)
    n = _dimension(context)
    q = zeros(n)
    for term in f.affine_terms
        q[context.index[term.variable]] += term.coefficient
    end
    if isempty(f.quadratic_terms)
        atom = Argo.Atoms.affine_operator(reshape(q, 1, n), [f.constant])
        return atom(context.variable)
    end
    # MOI stores `(1/2) x'Qx + q'x + c`, which is exactly Argo's `quadratic_form`.
    Q = zeros(n, n)
    for term in f.quadratic_terms
        i = context.index[term.variable_1]
        j = context.index[term.variable_2]
        Q[i, j] += term.coefficient
        i == j || (Q[j, i] += term.coefficient)
    end
    return Argo.Atoms.quadratic_form(Q, q)(context.variable)
end

# Convert `coefficient * f`. Constant factors are pushed down to the leaves so
# that, e.g., `-(f + g)` keeps the structure `-f + -g`. When `top` is true, `f`
# is a summand of the objective so its additive constant can be dropped.
function _term(f, context::_Context, coefficient::Float64=1.0, top::Bool=false)
    part = _quadratic(f)
    if part !== nothing
        return _leaf(MOI.Utilities.operate(*, Float64, coefficient, part), context)
    end
    return _nonlinear_term(f, context, coefficient, top)
end

function _scale(coefficient::Real, value::Term)
    isone(coefficient) && return value
    return coefficient * value
end

function _nonlinear_term(
    f::MOI.ScalarNonlinearFunction, context::_Context, coefficient::Float64, top::Bool
)
    args = f.args
    if f.head === :+
        return _sum(args, fill(coefficient, length(args)), context, top)
    elseif f.head === :- && length(args) == 1
        return _sum(args, [-coefficient], context, top)
    elseif f.head === :- && length(args) == 2
        return _sum(args, [coefficient, -coefficient], context, top)
    elseif f.head === :*
        rest = Any[]
        for arg in args
            value = _constant(arg)
            value === nothing ? push!(rest, arg) : (coefficient *= value)
        end
        length(rest) == 1 || throw(
            ArgumentError("Argo cannot represent a product of nonconstant factors")
        )
        return _term(only(rest), context, coefficient, top)
    elseif f.head === :/ && length(args) == 2
        denominator = _constant(args[2])
        denominator === nothing &&
            throw(ArgumentError("Argo only supports division by a constant"))
        return _term(args[1], context, coefficient / denominator, top)
    elseif f.head === :^ && length(args) == 2
        p = _constant(args[2])
        p === nothing && throw(ArgumentError("Argo only supports constant exponents"))
        isone(p) && return _term(args[1], context, coefficient, top)
        value = Argo.compose(_power_term(p), _term(args[1], context))
    elseif f.head in (:max, :min)
        values = Term[_term(arg, context) for arg in args]
        value = if length(values) == 1
            only(values)
        elseif f.head === :max
            maximum(values...)
        else
            minimum(values...)
        end
    elseif length(args) == 1
        value = Argo.compose(_atom_term(f.head), _term(only(args), context))
    else
        throw(ArgumentError("unsupported nonlinear operator `$(f.head)` for Argo"))
    end
    return _scale(coefficient, value)
end

function _nonlinear_term(f, ::_Context, ::Float64, ::Bool)
    throw(ArgumentError("unsupported function of type $(typeof(f)) for Argo"))
end

# Merge the quadratic children of a sum into one atom and keep the others as
# separate terms of the Argo sum.
function _sum(args, coefficients::Vector{Float64}, context::_Context, top::Bool)
    quadratic = _quadratic(0)
    values = Term[]
    for (arg, coefficient) in zip(args, coefficients)
        part = _quadratic(arg)
        if part === nothing
            push!(values, _nonlinear_term(arg, context, coefficient, top))
        else
            scaled = MOI.Utilities.operate(*, Float64, coefficient, part)
            quadratic = _add(quadratic, scaled)
        end
    end
    keep = _degree(quadratic) > 0 || isempty(values)
    keep |= !top && !iszero(MOI.constant(quadratic))
    keep && pushfirst!(values, _leaf(quadratic, context))
    return reduce(+, values)
end

function _context(variables::Vector{MOI.VariableIndex}, name::Symbol)
    index = Dict(vi => i for (i, vi) in enumerate(variables))
    return _Context(Argo.variable(name; shape=[length(variables)]), index)
end

"""
    Argo.term(f::MOI.AbstractScalarFunction, variables::Vector{MOI.VariableIndex}; name=:x)

Convert `f` into an Argo objective term over a single vector variable `name`
whose `i`th coordinate is `variables[i]`.
"""
function Argo.term(
    f::Union{
        MOI.VariableIndex,
        MOI.ScalarAffineFunction,
        MOI.ScalarQuadraticFunction,
        MOI.ScalarNonlinearFunction,
    },
    variables::Vector{MOI.VariableIndex};
    name::Symbol=:x,
)
    return _term(f, _context(variables, name), 1.0, true)
end

function _box(model::MOI.ModelLike, variables::Vector{MOI.VariableIndex})
    lower = fill(-Inf, length(variables))
    upper = fill(Inf, length(variables))
    for (i, vi) in enumerate(variables)
        l, u = MOI.Utilities.get_bounds(model, Float64, vi)
        lower[i], upper[i] = l, u
    end
    all(isinf, lower) && all(isinf, upper) && return nothing
    all(isfinite, lower) && all(isfinite, upper) || throw(
        ArgumentError(
            "Argo only supports variable bounds that are finite for every variable"
        ),
    )
    return Argo.Atoms.indicator(:box; lower, upper)
end

const _BOUND_SETS = (
    MOI.GreaterThan{Float64},
    MOI.LessThan{Float64},
    MOI.EqualTo{Float64},
    MOI.Interval{Float64},
)

"""
    Argo.minimize(model::MOI.ModelLike; name=:x)

Convert the objective of `model` into an Argo `Problem` over a vector
variable `name` collecting the variables of `model` in the order of
`MOI.ListOfVariableIndices`. Finite bounds on every variable become an
`indicator(:box)` term; any other constraint is rejected.
"""
function Argo.minimize(model::MOI.ModelLike; name::Symbol=:x)
    for (F, S) in MOI.get(model, MOI.ListOfConstraintTypesPresent())
        F === MOI.VariableIndex && S in _BOUND_SETS && continue
        throw(ArgumentError("Argo does not support `$F`-in-`$S` constraints"))
    end
    sense = MOI.get(model, MOI.ObjectiveSense())
    sense === MOI.FEASIBILITY_SENSE &&
        throw(ArgumentError("the model has no objective to minimize"))
    variables = MOI.get(model, MOI.ListOfVariableIndices())
    F = MOI.get(model, MOI.ObjectiveFunctionType())
    f = MOI.get(model, MOI.ObjectiveFunction{F}())
    coefficient = sense === MOI.MAX_SENSE ? -1.0 : 1.0
    context = _context(variables, name)
    objective = _term(f, context, coefficient, true)
    box = _box(model, variables)
    box === nothing || (objective = objective + box(context.variable))
    return Argo.minimize(objective)
end

# AutoTune meta-optimizer

mutable struct AutoTuneOptimizer <: MOI.AbstractOptimizer
    optimizer_constructor::Any
    accuracy::Float64
    initial_bounds::Dict{Symbol,Float64}
    oracle_costs::Dict
    options::Dict{MOI.AbstractOptimizerAttribute,Any}
    # Only used to answer `supports` queries before `copy_to`.
    probe::Union{Nothing,MOI.ModelLike}
    inner::Union{Nothing,MOI.ModelLike}
    plan::Any
end

function _default_initial_bounds()
    return Dict(q => 1.0 for q in Argo._KNOWN_INITIAL_BOUNDS if q !== :none)
end

function AutoTuneOptimizer(
    optimizer_constructor;
    accuracy::Real=1e-6,
    initial_bounds=_default_initial_bounds(),
    oracle_costs=Dict(),
)
    return AutoTuneOptimizer(
        optimizer_constructor,
        accuracy,
        Dict{Symbol,Float64}(initial_bounds),
        oracle_costs,
        Dict{MOI.AbstractOptimizerAttribute,Any}(),
        nothing,
        nothing,
        nothing,
    )
end

function Argo.AutoTune(optimizer_constructor; kwargs...)
    return () -> AutoTuneOptimizer(optimizer_constructor; kwargs...)
end

function _probe(model::AutoTuneOptimizer)
    if model.probe === nothing
        model.probe = MOI.instantiate(model.optimizer_constructor)
    end
    return model.probe
end

MOI.is_empty(model::AutoTuneOptimizer) = model.inner === nothing

function MOI.empty!(model::AutoTuneOptimizer)
    model.inner = nothing
    model.plan = nothing
    return
end

MOI.supports_incremental_interface(::AutoTuneOptimizer) = false

function MOI.supports_constraint(
    model::AutoTuneOptimizer,
    F::Type{<:MOI.AbstractFunction},
    S::Type{<:MOI.AbstractSet},
)
    return MOI.supports_constraint(_probe(model), F, S)
end

function MOI.supports(
    model::AutoTuneOptimizer,
    attr::Union{MOI.ObjectiveSense,MOI.ObjectiveFunction},
)
    return MOI.supports(_probe(model), attr)
end

function MOI.supports(
    model::AutoTuneOptimizer,
    attr::MOI.VariablePrimalStart,
    ::Type{MOI.VariableIndex},
)
    return MOI.supports(_probe(model), attr, MOI.VariableIndex)
end

const _PLAN = MOI.RawOptimizerAttribute("Argo.plan")

MOI.supports(::AutoTuneOptimizer, ::MOI.RawOptimizerAttribute) = true
function MOI.supports(model::AutoTuneOptimizer, ::MOI.Silent)
    return MOI.supports(_probe(model), MOI.Silent())
end

function MOI.set(
    model::AutoTuneOptimizer,
    attr::Union{MOI.RawOptimizerAttribute,MOI.Silent},
    value,
)
    attr == _PLAN && throw(MOI.SetAttributeNotAllowed(attr))
    model.options[attr] = value
    return
end

function MOI.get(model::AutoTuneOptimizer, attr::MOI.RawOptimizerAttribute)
    attr == _PLAN && return model.plan
    haskey(model.options, attr) && return model.options[attr]
    return MOI.get(something(model.inner, _probe(model)), attr)
end

MOI.get(model::AutoTuneOptimizer, ::MOI.Silent) = get(model.options, MOI.Silent(), false)

function MOI.get(model::AutoTuneOptimizer, ::MOI.SolverName)
    return "Argo.AutoTune(" * MOI.get(_probe(model), MOI.SolverName()) * ")"
end

"""
The method parameters of `certificate` as numbers, or `nothing` if one of them
is symbolic or changes with the iteration (e.g., a momentum schedule).
"""
function _numeric_parameters(certificate, initial_bounds)
    parameters = Dict{Symbol,Float64}()
    for name in certificate.plan.method.required_parameters
        values = map((1, 2)) do k
            value = Argo.parameter_value(certificate, name; k, initial_bounds)
            value isa Real ? value : Argo.try_evaluate_scalar(value)
        end
        any(isnothing, values) && return nothing
        values[1] == values[2] || return nothing
        parameters[name] = values[1]
    end
    return parameters
end

function _select_plan(model::AutoTuneOptimizer, src::MOI.ModelLike, inner)
    implementations = Argo.implemented_methods(inner)
    isempty(implementations) && return nothing
    problem = try
        Argo.minimize(src)
    catch err
        # The model is outside what Argo can analyze: keep the solver defaults.
        err isa ArgumentError || rethrow()
        return nothing
    end
    methods = Set(impl.method for impl in implementations)
    # A reformulation would change the problem given to the solver, so only
    # certificates for the formulation as written can be configured.
    candidates = filter(Argo.certificates(problem)) do certificate
        return isempty(certificate.reformulation_steps) &&
               certificate.plan.method.id in methods
    end
    ranked = Argo.rank(
        candidates;
        accuracy=model.accuracy,
        initial_bounds=model.initial_bounds,
        oracle_costs=model.oracle_costs,
    )
    for assessment in ranked
        certificate = only(assessment.phases).certificate
        parameters = _numeric_parameters(certificate, model.initial_bounds)
        parameters === nothing && continue
        implementation = first(
            impl for impl in implementations if impl.method === certificate.plan.method.id
        )
        return (; assessment, certificate, implementation, parameters)
    end
    return nothing
end

function MOI.copy_to(model::AutoTuneOptimizer, src::MOI.ModelLike)
    inner = MOI.instantiate(model.optimizer_constructor)
    for (attr, value) in model.options
        MOI.set(inner, attr, value)
    end
    model.plan = _select_plan(model, src, inner)
    if model.plan !== nothing
        model.plan.implementation.configure!(inner, model.plan.parameters)
    end
    index_map = MOI.copy_to(inner, src)
    model.inner = inner
    return index_map
end

MOI.optimize!(model::AutoTuneOptimizer) = MOI.optimize!(model.inner)

const _FORWARDED_MODEL_ATTRIBUTES = Union{
    MOI.TerminationStatus,
    MOI.PrimalStatus,
    MOI.DualStatus,
    MOI.ResultCount,
    MOI.RawStatusString,
    MOI.ObjectiveValue,
    MOI.DualObjectiveValue,
    MOI.ObjectiveBound,
    MOI.SolveTimeSec,
    MOI.BarrierIterations,
}

function MOI.get(model::AutoTuneOptimizer, attr::_FORWARDED_MODEL_ATTRIBUTES)
    if model.inner === nothing
        attr isa MOI.TerminationStatus && return MOI.OPTIMIZE_NOT_CALLED
        attr isa MOI.ResultCount && return 0
        attr isa Union{MOI.PrimalStatus,MOI.DualStatus} && return MOI.NO_SOLUTION
    end
    return MOI.get(model.inner, attr)
end

function MOI.get(
    model::AutoTuneOptimizer,
    attr::MOI.VariablePrimal,
    vi::MOI.VariableIndex,
)
    return MOI.get(model.inner, attr, vi)
end

function MOI.get(
    model::AutoTuneOptimizer,
    attr::Union{MOI.ConstraintDual,MOI.ConstraintPrimal},
    ci::MOI.ConstraintIndex,
)
    return MOI.get(model.inner, attr, ci)
end

end
