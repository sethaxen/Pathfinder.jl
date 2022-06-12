using DynamicHMC,
    LinearAlgebra,
    LogDensityProblems,
    MCMCDiagnosticTools,
    Optim,
    Pathfinder,
    Random,
    Statistics,
    StatsFuns,
    Test,
    TransformVariables

Random.seed!(1)

struct RegressionProblem{X,Y}
    x::X
    y::Y
end

function (prob::RegressionProblem)(θ)
    σ = θ.σ
    α = θ.α
    β = θ.β
    x = prob.x
    y = prob.y
    lp = normlogpdf(σ) + logtwo
    lp += normlogpdf(α)
    lp += sum(normlogpdf, β)
    y_hat = muladd(x, β, α)
    lp += sum(eachindex(y_hat, y)) do i
        return normlogpdf(y_hat[i], σ, y[i])
    end
    return lp
end

function mean_and_mcse(f, θs)
    zs = map(f, θs)
    ms = mean(zs)
    ses = map(mcse, eachrow(reduce(hcat, zs)))
    return ms, ses
end

function compare_estimates(f, xs1, xs2, α=0.05)
    nparams = length(first(xs1))
    α /= nparams  # bonferroni correction
    p = α / 2
    m1, s1 = mean_and_mcse(f, xs1)
    m2, s2 = mean_and_mcse(f, xs2)
    zs = @. (m1 - m2) / sqrt(s1^2 + s2^2)
    @test all(norminvcdf(p) .< zs .< norminvccdf(p))
end

@testset "DynamicHMC integration" begin
    A = Diagonal(rand(5))
    B = randn(5, 2)
    D = exp(Symmetric(randn(2, 2)))
    M⁻¹ = Pathfinder.WoodburyPDMat(A, B, D)

    @testset "Pathfinder.WoodburyLeftInvFactor" begin
        L = Pathfinder.WoodburyLeftInvFactor(M⁻¹)
        v = randn(5)
        V = randn(5, 2)
        @test L.A === M⁻¹
        @test size(L) === size(M⁻¹)
        Lmat = @inferred Matrix(L)
        @test Lmat isa Matrix
        @test Lmat * Lmat' ≈ inv(M⁻¹)
        @test L * v ≈ Lmat * v
        @test L * V ≈ Lmat * V
        @test AbstractMatrix{Float32}(L) isa Pathfinder.WoodburyLeftInvFactor{Float32}
        for i in axes(L, 1), j in axes(L, 2)
            @test L[i, j] ≈ Lmat[i, j]
        end
    end

    @testset "DynamicHMC.GaussianKineticEnergy" begin
        κ = DynamicHMC.GaussianKineticEnergy(M⁻¹)
        @test κ.M⁻¹ === M⁻¹
        @test κ.W === Pathfinder.WoodburyLeftInvFactor(M⁻¹)
        κdense = DynamicHMC.GaussianKineticEnergy(Symmetric(Matrix(M⁻¹)))
        κdense2 = DynamicHMC.GaussianKineticEnergy(Symmetric(Matrix(M⁻¹)), Matrix(κ.W))
        p = randn(5)
        q = randn(5)

        @test size(κ) == size(κdense)
        @test DynamicHMC.kinetic_energy(κ, p, q) ≈ DynamicHMC.kinetic_energy(κdense, p, q)
        @test DynamicHMC.calculate_p♯(κ, p, q) ≈ DynamicHMC.calculate_p♯(κdense, p, q)
        @test DynamicHMC.∇kinetic_energy(κ, p, q) ≈ DynamicHMC.∇kinetic_energy(κdense, p, q)
        @test DynamicHMC.rand_p(MersenneTwister(42), κ, q) ≈
            DynamicHMC.rand_p(MersenneTwister(42), κdense2, q)
    end

    @testset "DynamicHMC.mcmc_with_warmup" begin
        ndraws = 1_000
        x = 0:0.01:1
        y = sin.(x) .+ randn.() .* 0.2 .+ x
        X = [x x .^ 2 x .^ 3]
        prob = RegressionProblem(X, y)
        trans = as((σ=asℝ₊, α=asℝ, β=as(Array, size(X, 2))))
        P = TransformedLogDensity(trans, prob)
        ∇P = ADgradient(:ForwardDiff, P)
        rng = Random.GLOBAL_RNG

        result_hmc1 = mcmc_with_warmup(rng, ∇P, ndraws; reporter=NoProgressReport())

        logp(x) = LogDensityProblems.logdensity(P, x)
        ∇logp(x) = LogDensityProblems.logdensity_and_gradient(∇P, x)[2]
        result_pf = pathfinder(logp, ∇logp; dim=LogDensityProblems.dimension(P), ndraws=1)

        @testset "Initial point" begin
            result_hmc2 = mcmc_with_warmup(
                rng,
                ∇P,
                ndraws;
                initialization=(; q=result_pf.draws[:, 1]),
                reporter=NoProgressReport(),
            )
            @test result_hmc2.κ.M⁻¹ isa Diagonal
            compare_estimates(identity, result_hmc2.chain, result_hmc1.chain)
        end

        @testset "Initial point and metric" begin
            result_hmc3 = mcmc_with_warmup(
                rng,
                ∇P,
                ndraws;
                initialization=(;
                    q=result_pf.draws[:, 1],
                    κ=GaussianKineticEnergy(result_pf.fit_distribution.Σ),
                ),
                warmup_stages=default_warmup_stages(; M=Symmetric),
                reporter=NoProgressReport(),
            )
            @test result_hmc3.κ.M⁻¹ isa Symmetric
            compare_estimates(identity, result_hmc3.chain, result_hmc1.chain)
        end

        @testset "Initial point and final metric" begin
            result_hmc4 = mcmc_with_warmup(
                rng,
                ∇P,
                ndraws;
                initialization=(;
                    q=result_pf.draws[:, 1],
                    κ=GaussianKineticEnergy(result_pf.fit_distribution.Σ),
                ),
                warmup_stages=default_warmup_stages(; middle_steps=0, doubling_stages=0),
                reporter=NoProgressReport(),
            )
            @test result_hmc4.κ.M⁻¹ === result_pf.fit_distribution.Σ
            compare_estimates(identity, result_hmc4.chain, result_hmc1.chain)
        end
    end
end
