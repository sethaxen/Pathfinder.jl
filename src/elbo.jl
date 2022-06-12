"""
    MaximumELBO(; save_draws=false, rng, ndraws, executor)

An optimizer over the trace of fit distributions that returns the ELBO-maximizing distribution.

The ELBO is approximated using Monte Carlo sampling with `ndraws` and the provided `rng`.
This draws can be reused by Pathfinder to avoid extra log-density evaluations. To enable
this, set `save_draws=true`.

`executors` is a Transducers.jl executor that determines if and how to perform ELBO
computation in parallel. The default (`Transducers.SequentialEx()`) performs no
parallelization. If `rng` is known to be thread-safe, and the log-density function is known
to have no internal state, then `Transducers.PreferParallel()` may be used to parallelize
log-density evaluation. This is generally only faster for expensive log density functions.
"""
struct MaximumELBO{save_draws,R,E}
    rng::R
    ndraws::Int
    executor::E
end
function MaximumELBO(;
    rng::Random.AbstractRNG=Random.GLOBAL_RNG,
    ndraws::Int=DEFAULT_NDRAWS_ELBO,
    executor=Transducers.SequentialEx(),
    save_draws::Bool=false,
)
    return MaximumELBO{save_draws,typeof(rng),typeof(executor)}(rng, ndraws, executor)
end

function (optimizer::MaximumELBO{save_draws})(logp, _, _, dists) where {save_draws}
    rng = optimizer.rng
    ndraws = optimizer.ndraws
    executor = optimizer.executor
    EE = Core.Compiler.return_type(
        _compute_elbo, Tuple{typeof(rng),typeof(logp),eltype(dists),Int,Val{save_draws}}
    )
    dists_not_init = @views dists[(begin + 1):end]
    estimates = similar(dists_not_init, EE)
    isempty(estimates) && return false, first(dists), 0, estimates
    Folds.map!(estimates, dists_not_init, executor) do dist
        return _compute_elbo(rng, logp, dist, ndraws, Val(save_draws))
    end
    _, iteration_opt = _findmax(estimates |> Transducers.Map(est -> est.value))
    elbo_opt = estimates[iteration_opt].value
    success = !isnan(elbo_opt) & (elbo_opt != -Inf)
    return success, dists[iteration_opt + 1], iteration_opt, estimates
end

function _compute_elbo(rng, logp, dist, ndraws, ::Val{save_draws}) where {save_draws}
    ϕ, logqϕ = rand_and_logpdf(rng, dist, ndraws)
    logpϕ = similar(logqϕ)
    logpϕ .= logp.(eachcol(ϕ))
    logr = logpϕ - logqϕ
    elbo = Statistics.mean(logr)
    elbo_se = sqrt(Statistics.var(logr) / length(logr))
    return ELBOEstimate(elbo, elbo_se, save_draws ? ϕ : nothing, logpϕ, logqϕ, logr)
end

"""
    ELBOEstimate

Container of results of ELBO estimation via Monte Carlo.

# Fields
- `value`: value of estimate
- `std_err`: Monte Carlo standard error of estimate
- `draws`: Draws used to compute estimate
- `log_densities_actual`: log density of actual distribution evaluated on `draws`.
- `log_densities_fit`: log density of fit distribution evaluated on `draws`.
- `log_density_ratios`: log of ratio of actual to fit density. `value` is the mean of this
    array.
"""
struct ELBOEstimate{T,P,L<:AbstractVector{T}}
    value::T
    std_err::T
    draws::P
    log_densities_actual::L
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

"""
    draws_from_fit_stats(fit_stats, fit_iteration) -> draws

If applicable, return draws the fit distribution from `fit_iteration` stored in `fit_stats`.

The draws must be the same type and layout as one would get by calling
`rand(fit_distribution)`.
"""
function draws_from_fit_stats end

draws_from_fit_stats(fit_stats, fit_iteration) = nothing
draws_from_fit_stats(estimates::AbstractVector{<:ELBOEstimate}, i::Int) = estimates[i].draws
