using AbstractDifferentiation
using Distributions
using ForwardDiff
using LinearAlgebra
using Optimization
using Pathfinder
using PSIS
using ReverseDiff
using SciMLBase
using Test
using Transducers

@testset "multi path pathfinder" begin
    @testset "MvNormal" begin
        dim = 10
        nruns = 20
        ndraws = 1000_000
        ndraws_per_run = ndraws ÷ nruns
        Σ = rand_pd_mat(Float64, dim)
        μ = randn(dim)
        d = MvNormal(μ, Σ)
        logp(x) = logpdf(d, x)
        ∇logp(x) = ForwardDiff.gradient(logp, x)
        rngs = if VERSION ≥ v"1.7"
            [MersenneTwister(), Random.default_rng()]
        else
            [MersenneTwister()]
        end
        seed = 76
        @testset for rng in rngs
            executor = rng isa MersenneTwister ? SequentialEx() : ThreadedEx()
            dist_optimizer = Pathfinder.MaximumELBO(; rng, ndraws=100)

            Random.seed!(rng, seed)
            result = multipathfinder(
                logp,
                ∇logp,
                ndraws;
                dim,
                nruns,
                dist_optimizer,
                ndraws_per_run,
                rng,
                executor,
            )
            @test result isa MultiPathfinderResult
            @test result.input === (logp, ∇logp)
            @test result.optim_fun isa SciMLBase.OptimizationFunction
            @test result.rng === rng
            @test result.optimizer ===
                Pathfinder.default_optimizer(Pathfinder.DEFAULT_HISTORY_LENGTH)
            @test result.fit_distribution isa MixtureModel
            @test ncomponents(result.fit_distribution) == nruns
            @test Distributions.component_type(result.fit_distribution) <: MvNormal
            @test result.draws isa AbstractMatrix
            @test size(result.draws) == (dim, ndraws)
            @test result.draw_component_ids isa Vector{Int}
            @test length(result.draw_component_ids) == ndraws
            @test extrema(result.draw_component_ids) == (1, nruns)
            @test result.fit_distribution_transformed === result.fit_distribution
            @test result.draws_transformed == result.draws
            @test result.pathfinder_results isa Vector{<:PathfinderResult}
            @test length(result.pathfinder_results) == nruns
            @test result.psis_result isa PSIS.PSISResult
            μ_hat = mean(result.draws; dims=2)
            Σ_hat = cov(result.draws .- μ_hat; dims=2, corrected=false)
            # adapted from the MvNormal tests
            # allow for 15x disagreement in atol, since this method is approximate
            multiplier = 15
            for i in eachindex(μ)
                atol = sqrt(Σ[i, i] / ndraws) * 8 * multiplier
                @test μ_hat[i] ≈ μ[i] atol = atol
            end
            for i in axes(Σ, 1), j in axes(Σ, 2)
                atol = sqrt(Σ[i, i] * Σ[j, j] / ndraws) * 10 * multiplier
                @test isapprox(Σ_hat[i, j], Σ[i, j], atol=atol)
            end

            Random.seed!(rng, seed)
            result2 = multipathfinder(
                logp, ndraws; dim, nruns, dist_optimizer, ndraws_per_run, rng, executor
            )
            @test result2.fit_distribution == result.fit_distribution
            @test result2.draws == result.draws
            @test result2.draw_component_ids == result.draw_component_ids

            Random.seed!(rng, seed)
            ad_backend = AD.ReverseDiffBackend()
            result3 = multipathfinder(
                logp,
                ndraws;
                dim,
                nruns,
                dist_optimizer,
                ndraws_per_run,
                rng,
                executor,
                ad_backend,
            )
            for (c1, c2) in
                zip(result.fit_distribution.components, result3.fit_distribution.components)
                @test c1 ≈ c2 atol = 1e-6
            end
        end

        init = [randn(dim) for _ in 1:nruns]
        result = multipathfinder(logp, ∇logp, ndraws; init)
        @test ncomponents(result.fit_distribution) == nruns
        @test size(result.draws) == (dim, ndraws)
    end
    @testset "errors if no gradient provided" begin
        logp(x) = -sum(abs2, x) / 2
        init = [randn(5) for _ in 1:10]
        fun = SciMLBase.OptimizationFunction(logp, Optimization.AutoForwardDiff())
        @test_throws ArgumentError multipathfinder(fun, 10; init)
    end
    @testset "errors if neither dim nor init valid" begin
        logp(x) = -sum(abs2, x) / 2
        nruns = 2
        @test_throws ArgumentError multipathfinder(logp, 10; nruns)
        @test_throws ArgumentError multipathfinder(logp, 10; nruns, dim=0)
        multipathfinder(logp, 10; nruns, dim=2)
        multipathfinder(logp, 10; init=[randn(2) for _ in 1:nruns])
    end
    @testset "errors if neither init nor nruns valid" begin
        logp(x) = -sum(abs2, x) / 2
        @test_throws ArgumentError multipathfinder(logp, 10; dim=5, nruns=0)
        multipathfinder(logp, 10; dim=5, nruns=2)
    end
end
