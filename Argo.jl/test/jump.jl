using Test

using Argo
import JuMP
import MathOptInterface as MOI

function _jump_model(n)
    model = JuMP.Model()
    JuMP.@variable(model, x[1:n])
    return model, x
end

@testset "JuMP: quadratic objective becomes a quadratic_form" begin
    model, x = _jump_model(2)
    JuMP.@objective(model, Min, x[1]^2 + 2x[2]^2 + x[1] - 3)
    problem = minimize(model)
    @test problem.objective.name === :quadratic_form
    @test problem.objective.variables == [variable(:x; shape=[2])]
    # (1/2) x'Qx with Q = diag(2, 4)
    @test evaluate_scalar(get_property(problem, :smooth).parameters[:L]) ≈ 4
    @test evaluate_scalar(get_property(problem, :strongly_convex).parameters[:mu]) ≈ 2
    @test !isempty(certificates(problem))
end

@testset "JuMP: nonlinear objective" begin
    model, x = _jump_model(2)
    JuMP.@objective(model, Min, abs(x[1] - x[2]) + x[1]^2 + x[2]^2)
    problem = minimize(model)
    objective = problem.objective
    @test objective.operation === Argo.SumOperation
    quadratic, absolute = objective.children
    @test quadratic.name === :quadratic_form
    @test absolute.operation === Argo.CompositionOperation
    outer, inner = absolute.children
    @test outer.name === :abs
    @test inner.name === :affine_operator
    @test has_property(problem, :convex)
    @test has_property(problem, :strongly_convex)
    # |x₁ - x₂| is √2-Lipschitz
    @test evaluate_scalar(get_property(absolute, :lipschitz).parameters[:M]) ≈ sqrt(2)
    @test has_oracle(outer, :subgradient)
    @test has_oracle(absolute, :value)
    @test !isempty(certificates(problem))
end

@testset "JuMP: smooth composition with an affine map" begin
    model, x = _jump_model(2)
    JuMP.@objective(model, Min, sin(3x[1] + 4x[2]))
    problem = minimize(model)
    # sin is 1-smooth and the inner map has norm 5
    @test evaluate_scalar(get_property(problem, :smooth).parameters[:L]) ≈ 25
    @test !has_property(problem, :convex)
end

@testset "JuMP: scale, division, negation, max and powers" begin
    model, x = _jump_model(2)
    f = 2 * exp(x[1]) / 4 - log(x[2]) + max(x[1], x[2]) + (exp(x[2]))^2 + x[1]^4
    t = Argo.term(f, x)
    @test t.operation === Argo.SumOperation
    @test length(t.children) == 5
    scaled, negated, maximum_term, squared, quartic = t.children
    @test scaled.operation === Argo.ScaleOperation
    @test evaluate_scalar(scaled.coefficient) ≈ 0.5
    @test only(scaled.children).children[1].name === :exp
    @test negated.operation === Argo.ScaleOperation
    @test evaluate_scalar(negated.coefficient) ≈ -1
    @test maximum_term.operation === Argo.MaximumOperation
    @test all(c -> c.name === :affine_operator, maximum_term.children)
    @test squared.children[1].name === :abs2
    @test quartic.children[1].name === :pow4
    @test has_property(quartic, :convex)
    @test has_property(scaled, :convex)
    @test !has_property(t, :convex)
end

@testset "JuMP: product of affine factors is quadratic" begin
    model, x = _jump_model(2)
    f = JuMP.@expression(model, (x[1] + 1) * (x[1] - 1) + exp(x[2]))
    t = Argo.term(f, x)
    @test t.children[1].name === :quadratic_form
    @test t.children[2].children[1].name === :exp
end

@testset "JuMP: maximization and box bounds" begin
    model = JuMP.Model()
    JuMP.@variable(model, 0 <= x[1:2] <= 1)
    JuMP.@objective(model, Max, -exp(x[1]) - x[2]^2)
    problem = minimize(model)
    @test problem.objective.operation === Argo.SumOperation
    @test problem.objective.children[end].name === :indicator_box
    @test has_property(problem, :convex)
    @test has_oracle(problem.objective.children[end], :prox)
end

@testset "JuMP: unsupported" begin
    model, x = _jump_model(2)
    JuMP.@objective(model, Min, exp(x[1]) * exp(x[2]))
    @test_throws ArgumentError minimize(model)
    JuMP.@objective(model, Min, x[1]^x[2])
    @test_throws ArgumentError minimize(model)
    JuMP.@objective(model, Min, sinh(x[1]))
    @test_throws ArgumentError minimize(model)
    JuMP.@objective(model, Min, x[1])
    JuMP.@constraint(model, x[1] + x[2] <= 1)
    @test_throws ArgumentError minimize(model)
    model = JuMP.Model()
    JuMP.@variable(model, x >= 0)
    JuMP.@objective(model, Min, x)
    @test_throws ArgumentError minimize(model)
end

@testset "JuMP: constants are kept inside max" begin
    model, x = _jump_model(2)
    t = Argo.term(max(exp(x[1]) + 1, x[2]) - 5, x)
    @test t.operation === Argo.MaximumOperation
    plus_one = t.children[1]
    @test plus_one.operation === Argo.SumOperation
    @test plus_one.children[1].name === :affine_operator
end

@testset "JuMP: other atoms" begin
    model, x = _jump_model(2)
    t = Argo.term(min(x[1]^3, atan(x[2])) + x[1]^1 * 2 + (x[2] + 1)^0, x)
    @test t.operation === Argo.SumOperation
    affine, minimum_term = t.children
    @test affine.name === :affine_operator
    @test minimum_term.operation === Argo.MinimumOperation
    cube, arctangent = minimum_term.children
    @test cube.children[1].name === :pow
    @test !has_property(cube, :convex)
    @test arctangent.children[1].name === :atan
    @test evaluate_scalar(get_property(arctangent, :lipschitz).parameters[:M]) ≈ 1
end

@testset "MOI: direct conversion" begin
    model = MOI.Utilities.Model{Float64}()
    x = MOI.add_variables(model, 2)
    MOI.add_constraint.(model, x, MOI.Interval(-1.0, 1.0))
    f = MOI.ScalarNonlinearFunction(:exp, Any[1.0 * x[2] + 2.0])
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.set(model, MOI.ObjectiveFunction{typeof(f)}(), f)
    problem = minimize(model; name=:y)
    exponential, box = problem.objective.children
    @test exponential.children[1].name === :exp
    @test exponential.children[2].variables == [variable(:y; shape=[2])]
    @test box.name === :indicator_box
    @test evaluate_scalar(get_property(box, :compact_domain).parameters[:diameter]) ≈
          2sqrt(2)
    @test Argo.term(x[1], x).name === :affine_operator
    MOI.set(model, MOI.ObjectiveSense(), MOI.FEASIBILITY_SENSE)
    @test_throws ArgumentError minimize(model)
end
