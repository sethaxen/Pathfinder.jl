using AbstractDifferentiation
using Distributions
using ForwardDiff
using LinearAlgebra
using Optim
using Optimization
using Pathfinder
using Random
using ReverseDiff
using SciMLBase
using Test
using Transducers

@testset "single path pathfinder" begin
    @testset "IsoNormal" begin
        # here pathfinder finds the exact solution after 1 iteration
        logp(x) = -sum(abs2, x) / 2
        ∇logp(x) = -x
        ndraws = 100
        rngs = if VERSION ≥ v"1.7"
            [MersenneTwister(), Random.default_rng()]
        else
            [MersenneTwister()]
        end
        seed = 42
        @testset for dim in [1, 5, 10, 100], rng in rngs
            executor = rng isa MersenneTwister ? SequentialEx() : ThreadedEx()
            dist_optimizer = Pathfinder.MaximumELBO(; rng, executor)

            init = randn(dim)
            Random.seed!(rng, seed)
            result = @inferred pathfinder(logp, ∇logp; init, ndraws, rng, dist_optimizer)
            @test result isa PathfinderResult
            @test result.input === (logp, ∇logp)
            @test result.optim_prob isa SciMLBase.OptimizationProblem
            @test result.logp(init) ≈ logp(init)
            @test result.rng === rng
            @test result.optimizer ===
                Pathfinder.default_optimizer(Pathfinder.DEFAULT_HISTORY_LENGTH)
            fit_distribution = result.fit_distribution
            @test fit_distribution isa MvNormal
            @test fit_distribution.μ ≈ zeros(dim)
            @test fit_distribution.Σ isa Pathfinder.WoodburyPDMat
            @test fit_distribution.Σ ≈ I
            @test size(fit_distribution.Σ.B) == (dim, 2) # history contains only 1 iteration
            @test result.draws isa AbstractMatrix
            @test size(result.draws) == (dim, ndraws)
            @test result.fit_distribution_transformed === result.fit_distribution
            @test result.draws_transformed === result.draws
            @test result.num_tries ≥ 1
            @test result.optim_solution isa SciMLBase.OptimizationSolution
            @test result.optim_trace isa Pathfinder.OptimizationTrace
            @test result.fit_distributions isa Vector{typeof(fit_distribution)}
            @test length(result.fit_distributions) == length(result.optim_trace)
            @test result.fit_distributions[result.fit_iteration + 1] == fit_distribution
            @test result.fit_iteration == argmax(getproperty.(result.fit_stats, :value))

            Random.seed!(rng, seed)
            result2 = pathfinder(logp, ∇logp; init, ndraws, rng, dist_optimizer)
            @test result2.fit_iteration == result.fit_iteration
            @test result2.draws == result.draws
            @test getproperty.(result2.fit_stats, :value) ==
                getproperty.(result.fit_stats, :value)

            ndraws = 2
            result3 = pathfinder(logp, ∇logp; init, ndraws, dist_optimizer)
            @test size(result3.draws) == (dim, ndraws)
        end
    end
    @testset "MvNormal" begin
        #! format: off
        Σ = [
            2.71   0.5    0.19   0.07   1.04
            0.5    1.11  -0.08  -0.17  -0.08
            0.19  -0.08   0.26   0.07  -0.7
            0.07  -0.17   0.07   0.11  -0.21
            1.04  -0.08  -0.7   -0.21   8.65
        ]
        #! format: on
        P = inv(Symmetric(Σ))
        logp(x) = -dot(x, P, x) / 2
        ∇logp(x) = -(P * x)
        dim = 5
        ad_backend = AD.ReverseDiffBackend()
        rngs = if VERSION ≥ v"1.7"
            [MersenneTwister(), Random.default_rng()]
        else
            [MersenneTwister()]
        end
        seed = 38
        optimizer = Optim.LBFGS(; m=6)
        @testset for rng in rngs
            executor = rng isa MersenneTwister ? SequentialEx() : ThreadedEx()
            dist_optimizer = Pathfinder.MaximumELBO(;
                rng, executor, ndraws=100, save_draws=true
            )

            Random.seed!(rng, seed)
            result = @inferred pathfinder(
                logp; rng, dim, optimizer, dist_optimizer, ad_backend
            )
            @test result.input === logp
            @test result.fit_distribution.Σ ≈ Σ rtol = 1e-1
            @test result.optimizer == optimizer
            @test size(result.draws) == (5, 100)
            @test result.draws == result.fit_stats[result.fit_iteration].draws
            Random.seed!(rng, seed)
            result2 = pathfinder(logp; rng, dim, optimizer, dist_optimizer, ad_backend)
            @test result2.fit_distribution == result.fit_distribution
            @test result2.draws == result.draws
            @test getproperty.(result2.fit_stats, :value) ==
                getproperty.(result.fit_stats, :value)
        end
        @testset "kwargs forwarded to solve" begin
            Random.seed!(42)
            i = 0
            callback = (args...,) -> (i += 1; false)
            pathfinder(logp; dim, callback)
            @test i ≠ 6

            Random.seed!(42)
            i = 0
            pathfinder(logp; dim, callback, maxiters=5)
            @test i == 6
        end
    end
    @testset "retries" begin
        @testset "logp returning NaN" begin
            dim = 5
            nfail = 20
            logp(x) = i ≤ nfail ? NaN : -sum(abs2, x) / 2
            callback = (args...,) -> (i += 1; true)
            i = 1
            result = pathfinder(logp; dim, callback)
            @test result.fit_distribution.μ ≈ zeros(dim) atol = 1e-6
            @test result.fit_distribution.Σ ≈ diagm(ones(dim)) atol = 1e-6
            @test result.num_tries == nfail + 1
            @test result.optim_prob.u0 == result.optim_trace.points[1]
            i = 1
            init = randn(dim)
            result2 = pathfinder(logp; init, callback, ntries=nfail)
            @test !isapprox(result2.fit_distribution.μ, zeros(dim); atol=1e-6)
            @test result2.fit_iteration == 0
            @test isempty(result2.fit_stats)
            @test result2.num_tries == nfail
            @test result2.optim_prob.u0 == result2.optim_trace.points[1]
        end
    end
    @testset "UniformSampler" begin
        @test_throws DomainError Pathfinder.UniformSampler(-1.0)
        @test_throws DomainError Pathfinder.UniformSampler(0.0)
        @testset for scale in [1, 2], seed in [42, 38]
            sampler_fun = Pathfinder.UniformSampler(scale)
            rng = Random.seed!(Random.default_rng(), seed)
            x = zeros(100)
            sampler_fun(rng, x)
            @test all(-scale .≤ x .≤ scale)
            Random.seed!(rng, seed)
            x2 = zeros(100)
            x2 .= rand.(rng) .* 2scale .- scale
            @test x2 == x
        end
    end
    @testset "errors if no gradient provided" begin
        logp(x) = -sum(abs2, x) / 2
        init = randn(5)
        prob = SciMLBase.OptimizationProblem(logp, init, nothing)
        @test_throws ArgumentError pathfinder(prob)
        fun = SciMLBase.OptimizationFunction(logp, Optimization.AutoForwardDiff())
        prob = SciMLBase.OptimizationProblem(fun, init, nothing)
        @test_throws ArgumentError pathfinder(prob)
    end
    @testset "errors if neither dim nor init valid" begin
        logp(x) = -sum(abs2, x) / 2
        @test_throws ArgumentError pathfinder(logp)
        @test_throws ArgumentError pathfinder(logp; dim=0)
        pathfinder(logp; dim=3)
        pathfinder(logp; init=randn(5))
    end
end
