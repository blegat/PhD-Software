"""
Argo discovers and compares certificate-backed first-order minimization plans
for an initial formulation and finite, guarantee-preserving reformulations.
It analyzes methods; it never executes an optimization algorithm.
"""
module Argo

using PrecompileTools

include("scalars.jl")
include("modeling.jl")
include("inference.jl")

# Named scientific kernels referenced by the TOML catalogue. They are internal
# implementation details rather than public rate-function APIs.
include("rates.jl")
include("catalogue.jl")
include("reformulation_core.jl")
include("certification.jl")
include("ranking.jl")

include("standard_atoms.jl")
include("modeling_dsl.jl")
include("authoring.jl")
include("curvature_transfer.jl")
include("recursive.jl")
include("hybrid.jl")
include("sampling.jl")
include("parameter_optimization.jl")
include("regime_analysis.jl")
include("autotune.jl")
include("precompile_workload.jl")

export AutoTune,
    Certificate,
    ApplicabilityRequest,
    ComparisonContext,
    FixedBudgetRequest,
    Oracle,
    OracleComplexity,
    MetricEstimate,
    PlanAssessment,
    PlanPhase,
    Problem,
    Property,
    RankingRequest,
    ScalarExpr,
    ScalarSymbol,
    MethodImplementation,
    Term,
    Variable

export certificate_bound,
    certificates,
    compose,
    evaluate_scalar,
    get_oracle,
    get_property,
    implemented_methods,
    has_oracle,
    has_property,
    iterations_to_accuracy,
    minimize,
    metric,
    objective_terms,
    oracle,
    oracle_complexity,
    oracles,
    parameter_domain,
    parameter_interval,
    parameter_value,
    properties,
    property,
    rank,
    scalar,
    scalar_symbols,
    symbolic,
    term,
    try_evaluate_scalar,
    variable,
    with

export centered_quadratic,
    compact_domain,
    convex,
    frank_wolfe_curvature,
    gradient_mapping_dominance,
    hypo_convex,
    linear,
    linear_composition,
    lipschitz,
    monotone,
    polyak_lojasiewicz,
    quadratic,
    smooth,
    strongly_convex

export @assume, @minimize, @model, @oracle, @term, @variable

end
