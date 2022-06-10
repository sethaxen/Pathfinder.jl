function maximize_elbo(rng, logp, dists, ndraws, executor)
    μ = dists[1].μ
    N = length(μ)
    u = Random.randn!(rng, similar(μ, N, ndraws))
    EE = Core.Compiler.return_type(
        elbo_and_samples!, Tuple{typeof(rng),typeof(u),typeof(logp),eltype(dists),Int}
    )
    estimates = similar(dists, EE)
    isempty(estimates) && return 0, estimates
    Folds.map!(estimates, dists, executor) do dist
        return elbo_and_samples!(rng, u, logp, dist, ndraws)
    end
    _, iteration_opt = _findmax(estimates |> Transducers.Map(est -> est.value))
    return iteration_opt, estimates
end

function elbo_and_samples!(rng, u, logp, dist, ndraws)
    ϕ, logqϕ = rand_and_logpdf!(rng, u, dist, ndraws)
    logpϕ = similar(logqϕ)
    logpϕ .= logp.(eachcol(ϕ))
    logr = logpϕ - logqϕ
    elbo = Statistics.mean(logr)
    elbo_se = sqrt(Statistics.var(logr) / length(logr))
    return ELBOEstimate(elbo, elbo_se, ϕ, logpϕ, logqϕ, logr)
end

struct ELBOEstimate{T,P,L<:AbstractVector{T}}
    value::T
    std_err::T
    draws::P
    log_densities_target::L
    log_densities_fit::L
    log_density_ratios::L
end

function Base.show(io::IO, ::MIME"text/plain", elbo::ELBOEstimate)
    print(io, "ELBO estimate: ", _to_string(elbo))
    return nothing
end

function _to_string(est::ELBOEstimate; digits=2)
    return "$(round(est.value; digits)) ± $(round(est.std_err; digits))"
end
