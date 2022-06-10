"""
    fit_mvnormals(θs, ∇logpθs; history_length=5)

Fit a multivariate-normal distribution to each point on the trajectory `θs`.

Given `θs` with gradients `∇logpθs`, construct LBFGS inverse Hessian approximations with
the provided `history_length`. The inverse Hessians approximate a covariance. The
covariances and corresponding means that define multivariate normal approximations per
point are returned.
"""
function fit_mvnormals(θs, ∇logpθs; kwargs...)
    Σs = lbfgs_inverse_hessians(θs, ∇logpθs; kwargs...)
    trans = Transducers.MapSplat() do Σ, ∇logpθ, θ
        μ = muladd(Σ, ∇logpθ, θ)
        return Distributions.MvNormal(μ, Σ)
    end
    l = length(Σs)
    return @views(zip(Σs, ∇logpθs[1:l], θs[1:l])) |> trans |> collect
end

# faster than computing `logpdf` and `rand` independently
function rand_and_logpdf!(rng, u, dist::Distributions.MvNormal, ndraws)
    μ = dist.μ
    Σ = dist.Σ
    N = length(μ)

    # draw points
    Random.randn!(rng, u)
    unormsq = vec(sum(abs2, u; dims=1))
    x = PDMats.unwhiten!(u, Σ, u)
    x .+= μ

    # compute log density at each point
    logpx = (muladd(N, log2π, logdet(Σ)) .+ unormsq) ./ -2

    return x, logpx
end
