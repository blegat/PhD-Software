using Test

using Argo
import JSOSolvers
import JuMP
import MathOptInterface as MOI
import NLPModelsJuMP

function _fo_model(; autotune=true)
    optimizer = NLPModelsJuMP.Optimizer
    model = JuMP.Model(autotune ? Argo.AutoTune(optimizer) : optimizer)
    JuMP.set_attribute(model, "solver", JSOSolvers.FoSolver)
    JuMP.set_silent(model)
    JuMP.@variable(model, x[1:3], start = 1.0)
    # L = 8 and μ = 2
    JuMP.@objective(model, Min, x[1]^2 + 2x[2]^2 + 4x[3]^2 - x[1])
    return model, x
end

@testset "AutoTune: fo runs gradient descent with Argo's step" begin
    model, x = _fo_model()
    @test occursin("AutoTune", JuMP.solver_name(model))
    JuMP.optimize!(model)
    @test JuMP.is_solved_and_feasible(model)
    @test JuMP.value.(x) ≈ [0.5, 0, 0] atol = 1e-6
    plan = JuMP.get_attribute(model, "Argo.plan")
    @test plan.certificate.plan.method.id === :gradient_descent
    @test plan.certificate.theorem.id === :gd_sc_optimal_step_objective_from_distance
    # 2 / (L + μ)
    @test plan.parameters[:step_size] ≈ 0.2
    # Every step was accepted, so R2 kept the fixed step
    autotune = JuMP.unsafe_backend(model)
    @test autotune.inner.solver.α ≈ 0.2
    @test JuMP.get_attribute(model, "α0") ≈ 0.2
    # Same iterates as gradient descent with step 0.2 and R2's stopping rule
    ∇f(x) = [2x[1] - 1, 4x[2], 8x[3]]
    y, iterations = ones(3), 0
    tol = sqrt(eps()) * (1 + sqrt(sum(abs2, ∇f(y))))
    while sqrt(sum(abs2, ∇f(y))) > tol
        y -= 0.2 * ∇f(y)
        iterations += 1
    end
    @test autotune.inner.stats.iter == iterations
    @test JuMP.value.(x) ≈ y
end

@testset "AutoTune: no implementation leaves the solver untouched" begin
    model, x = _fo_model()
    JuMP.set_attribute(model, "solver", JSOSolvers.LBFGSSolver)
    JuMP.optimize!(model)
    @test JuMP.is_solved_and_feasible(model)
    @test JuMP.get_attribute(model, "Argo.plan") === nothing
    # Outside of what Argo can analyze
    model, x = _fo_model()
    JuMP.@objective(model, Min, exp(x[1]) * exp(x[2]) + x[3]^2)
    JuMP.optimize!(model)
    @test JuMP.get_attribute(model, "Argo.plan") === nothing
end
