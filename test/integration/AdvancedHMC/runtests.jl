using AdvancedHMC,
    ForwardDiff,
    LinearAlgebra,
    MCMCDiagnosticTools,
    Optim,
    Pathfinder,
    Random,
    Statistics,
    StatsFuns,
    Test

Random.seed!(0)

struct RegressionModel{X,Y}
    x::X
    y::Y
end

function (m::RegressionModel)(θ)
    logσ = θ[1]
    σ = exp(logσ)
    α = θ[2]
    β = θ[3:end]
    x = m.x
    y = m.y
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

@testset "AdvancedHMC integration" begin
    A = Diagonal(rand(5))
    B = randn(5, 2)
    D = exp(Symmetric(randn(2, 2)))
    M⁻¹ = Pathfinder.WoodburyPDMat(A, B, D)
    A2 = Diagonal(rand(5))
    B2 = randn(5, 2)
    D2 = exp(Symmetric(randn(2, 2)))
    M⁻¹2 = Pathfinder.WoodburyPDMat(A2, B2, D2)
    ℓπ(x) = -sum(abs2, x) / 2
    ∂ℓπ∂θ(x) = -x

    @testset "RankUpdateEuclideanMetric" begin
        metric = Pathfinder.RankUpdateEuclideanMetric(M⁻¹)
        @test metric.M⁻¹ === M⁻¹
        metric_dense = AdvancedHMC.DenseEuclideanMetric(Symmetric(Matrix(M⁻¹)))
        h = AdvancedHMC.Hamiltonian(metric, ℓπ, ∂ℓπ∂θ)
        h_dense = AdvancedHMC.Hamiltonian(metric_dense, ℓπ, ∂ℓπ∂θ)
        r = randn(5)
        θ = randn(5)

        metric2 = Pathfinder.RankUpdateEuclideanMetric(3)
        @test metric2.M⁻¹ ≈ I
        @test size(metric2) == (3, 3)
        metric2 = Pathfinder.RankUpdateEuclideanMetric((4,))
        @test metric2.M⁻¹ ≈ I
        @test size(metric2) == (4, 4)
        metric2 = Pathfinder.RankUpdateEuclideanMetric(Float32, (4,))
        @test metric2.M⁻¹ ≈ I
        @test size(metric2) == (4, 4)
        @test eltype(metric2.M⁻¹) === Float32

        @test size(metric) == (5, 5)
        @test AdvancedHMC.renew(metric, M⁻¹2).M⁻¹ === M⁻¹2
        @test sprint(show, metric) == "RankUpdateEuclideanMetric(diag=$(diag(metric.M⁻¹)))"
        @test AdvancedHMC.neg_energy(h, r, θ) ≈ AdvancedHMC.neg_energy(h_dense, r, θ)
        @test AdvancedHMC.∂H∂r(h, r) ≈ AdvancedHMC.∂H∂r(h_dense, r)
        compare_estimates(
            identity,
            [rand(metric) for _ in 1:10_000],
            [rand(metric_dense) for _ in 1:10_000],
        )
    end

    @testset "sample" begin
        ndraws = 1_000
        nadapts = 500
        nparams = 5
        x = 0:0.01:1
        y = sin.(x) .+ randn.() .* 0.2 .+ x
        X = [x x .^ 2 x .^ 3]
        θ₀ = randn(nparams)
        ℓπ = RegressionModel(X, y)

        metric = DiagEuclideanMetric(nparams)
        hamiltonian = Hamiltonian(metric, ℓπ, ForwardDiff)
        ϵ = find_good_stepsize(hamiltonian, θ₀)
        integrator = Leapfrog(ϵ)
        proposal = NUTS{MultinomialTS,GeneralisedNoUTurn}(integrator)
        adaptor = StanHMCAdaptor(
            MassMatrixAdaptor(metric), StepSizeAdaptor(0.8, integrator)
        )
        samples1, stats1 = sample(
            hamiltonian,
            proposal,
            θ₀,
            ndraws,
            adaptor,
            nadapts;
            drop_warmup=true,
            progress=false,
        )

        result_pf = pathfinder(ℓπ; dim=5, optimizer=Optim.LBFGS(; m=6), ndraws=1)

        @testset "Initial point" begin
            metric = DiagEuclideanMetric(nparams)
            hamiltonian = Hamiltonian(metric, ℓπ, ForwardDiff)
            ϵ = find_good_stepsize(hamiltonian, θ₀)
            integrator = Leapfrog(ϵ)
            proposal = NUTS{MultinomialTS,GeneralisedNoUTurn}(integrator)
            adaptor = StepSizeAdaptor(0.8, integrator)
            samples2, stats2 = sample(
                hamiltonian,
                proposal,
                result_pf.draws[:, 1],
                ndraws,
                adaptor,
                nadapts;
                drop_warmup=true,
                progress=false,
            )
            compare_estimates(identity, samples2, samples1)
        end

        @testset "Initial point and metric" begin
            metric = DiagEuclideanMetric(diag(result_pf.fit_distribution.Σ))
            hamiltonian = Hamiltonian(metric, ℓπ, ForwardDiff)
            ϵ = find_good_stepsize(hamiltonian, θ₀)
            integrator = Leapfrog(ϵ)
            proposal = NUTS{MultinomialTS,GeneralisedNoUTurn}(integrator)
            adaptor = StepSizeAdaptor(0.8, integrator)
            samples3, stats3 = sample(
                hamiltonian,
                proposal,
                result_pf.draws[:, 1],
                ndraws,
                adaptor,
                nadapts;
                drop_warmup=true,
                progress=false,
            )
            compare_estimates(identity, samples3, samples1)
        end

        @testset "Initial point and final metric" begin
            metric = Pathfinder.RankUpdateEuclideanMetric(result_pf.fit_distribution.Σ)
            hamiltonian = Hamiltonian(metric, ℓπ, ForwardDiff)
            ϵ = find_good_stepsize(hamiltonian, θ₀)
            integrator = Leapfrog(ϵ)
            proposal = NUTS{MultinomialTS,GeneralisedNoUTurn}(integrator)
            adaptor = StepSizeAdaptor(0.8, integrator)
            samples4, stats4 = sample(
                hamiltonian,
                proposal,
                result_pf.draws[:, 1],
                ndraws,
                adaptor,
                nadapts;
                drop_warmup=true,
                progress=false,
            )
            compare_estimates(identity, samples4, samples1)
        end
    end
end
