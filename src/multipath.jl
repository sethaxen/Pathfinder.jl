"""
    MultiPathfinderResult

Container for results of multi-path Pathfinder.

# Fields
- `input`: User-provided input object, e.g. either `logp`, `(logp, ∇logp)`, `optim_fun`,
    `optim_prob`, or another object.
- `optimizer`: Optimizer used for maximizing the log-density
- `rng`: Pseudorandom number generator that was used for sampling
- `optim_prob::SciMLBase.OptimizationProblem`: Otimization problem used for
    optimization
- `logp`: Log-density function
- `fit_distribution::Distributions.MixtureModel`: uniformly-weighted mixture of
    multivariate normal distributions returned by each run.
- `draws::AbstractMatrix{<:Real}`: draws from `fit_distribution` with size `(dim, ndraws)`,
    potentially resampled using importance resampling to be closer to the target
    distribution.
- `draw_component_ids::Vector{Int}`: component id of each draw in `draws`.
- `fit_distribution_transformed`: `fit_distribution` transformed to the same space as the
    user-supplied target distribution. This is only different from `fit_distribution` when
    integrating with other packages, and its type depends on the type of `input`.
- `draws_transformed`: `draws` transformed to be draws from `fit_distribution_transformed`.
- `pathfinder_results::Vector{<:PathfinderResult}`: results of each single-path Pathfinder
    run.
- `psis_result::Union{Nothing,<:PSIS.PSISResult}`: If importance resampling was used, the
    result of Pareto-smoothed importance resampling. `psis_result.pareto_shape` also
    diagnoses whether `draws` can be used to compute estimates from the target distribution.
    See [`PSIS.PSISResult`](https://psis.julia.arviz.org/stable/api/#PSIS.PSISResult) for
    details
"""
struct MultiPathfinderResult{I,O,R,OF,LP,FD,D,C,FDT,DT,PFR,PR}
    input::I
    optimizer::O
    rng::R
    optim_fun::OF
    logp::LP
    fit_distribution::FD
    draws::D
    draw_component_ids::C
    fit_distribution_transformed::FDT
    draws_transformed::DT
    pathfinder_results::PFR
    psis_result::PR
end

function Base.show(io::IO, ::MIME"text/plain", result::MultiPathfinderResult)
    println(io, "Multi-path Pathfinder result")
    println(io, "  runs: $(length(result.pathfinder_results))")
    print(io, "  draws: $(size(result.draws, 2))")
    psis_result = result.psis_result
    if psis_result !== nothing
        println(io)
        k = psis_result.pareto_shape
        assessment = if k > 1
            "very bad"
        elseif k > 0.7
            "bad"
        elseif k > 0.5
            "ok"
        else
            "good"
        end
        print(io, "  Pareto shape diagnostic: ", round(k; digits=2), " ($assessment)")
    end
    return nothing
end

"""
    multipathfinder(logp, ndraws; kwargs...)
    multipathfinder(logp, ∇logp, ndraws; kwargs...)
    multipathfinder(fun::SciMLBase.OptimizationFunction, ndraws; kwargs...)

Run [`pathfinder`](@ref) multiple times to fit a multivariate normal mixture model.

For `nruns=length(init)`, `nruns` parallel runs of pathfinder produce `nruns` multivariate
normal approximations ``q_k = q(\\phi | \\mu_k, \\Sigma_k)`` of the posterior. These are
combined to a mixture model ``q`` with uniform weights.

``q`` is augmented with the component index to generate random samples, that is, elements
``(k, \\phi)`` are drawn from the augmented mixture model
```math
\\tilde{q}(\\phi, k | \\mu, \\Sigma) = K^{-1} q(\\phi | \\mu_k, \\Sigma_k),
```
where ``k`` is a component index, and ``K=`` `nruns`. These draws are then resampled with
replacement. Discarding ``k`` from the samples would reproduce draws from ``q``.

If `importance=true`, then Pareto smoothed importance resampling is used, so that the
resulting draws better approximate draws from the target distribution ``p`` instead of
``q``. This also prints a warning message if the importance weighted draws are unsuitable
for approximating expectations with respect to ``p``.

# Arguments
- `logp`: a callable that computes the log-density of the target distribution.
- `∇logp`: a callable that computes the gradient of `logp`. If not provided, `logp` is
    automatically differentiated using the backend specified in `ad_backend`.
- `fun::SciMLBase.OptimizationFunction`: an optimization function that represents
    `-logp(x)` with its gradient. It must have the necessary features (e.g. a Hessian
    function) for the chosen optimization algorithm. For details, see
    [Optimization.jl: OptimizationFunction](https://optimization.sciml.ai/stable/API/optimization_function/).
- `ndraws::Int`: number of approximate draws to return

# Keywords
- `init`: iterator of length `nruns` of initial points of length `dim` from which each
    single-path Pathfinder run will begin. `length(init)` must be implemented. If `init` is
    not provided, `dim` and `nruns` must be.
- `nruns::Int`: number of runs of Pathfinder to perform. Ignored if `init` is provided.
- `ad_backend=AD.ForwardDiffBackend()`: AbstractDifferentiation.jl AD backend used to
    differentiate `logp` if `∇logp` is not provided.
- `ndraws_per_run::Int`: The number of draws to take for each component before resampling.
    Defaults to a number such that `ndraws_per_run * nruns > ndraws`.
- `importance::Bool=true`: Perform Pareto smoothed importance resampling of draws.
- `rng::AbstractRNG=Random.GLOBAL_RNG`: Pseudorandom number generator. It is recommended to
    use a parallelization-friendly PRNG like the default PRNG on Julia 1.7 and up.
- `executor::Transducers.Executor`: Transducers.jl executor that determines if and how
    to run the single-path runs in parallel. If `rng` is known to be thread-safe, the
    default is `Transducers.PreferParallel()` (parallel executation, defaulting to
    multi-threading). Otherwise, it is `Transducers.SequentialEx()` (no parallelization).
- `kwargs...` : Remaining keywords are forwarded to [`pathfinder`](@ref).

# Returns
- [`MultiPathfinderResult`](@ref)
"""
function multipathfinder end

function multipathfinder(
    logp, ndraws::Int; ad_backend=AD.ForwardDiffBackend(), input=logp, kwargs...
)
    return multipathfinder(build_optim_function(logp; ad_backend), ndraws; input, kwargs...)
end
function multipathfinder(
    logp,
    ∇logp,
    ndraws::Int;
    ad_backend=AD.ForwardDiffBackend(),
    input=(logp, ∇logp),
    kwargs...,
)
    return multipathfinder(
        build_optim_function(logp, ∇logp; ad_backend), ndraws; input, kwargs...
    )
end
function multipathfinder(
    optim_fun::SciMLBase.OptimizationFunction,
    ndraws::Int;
    init=nothing,
    input=optim_fun,
    nruns::Int=init === nothing ? -1 : length(init),
    ndraws_per_run::Int=cld(ndraws, max(nruns, 1)),
    rng::Random.AbstractRNG=Random.GLOBAL_RNG,
    history_length::Int=DEFAULT_HISTORY_LENGTH,
    optimizer=default_optimizer(history_length),
    executor::Transducers.Executor=_default_executor(rng),
    importance::Bool=true,
    kwargs...,
)
    if optim_fun.grad === nothing || optim_fun.grad isa Bool
        throw(ArgumentError("optimization function must define a gradient function."))
    end
    if init === nothing
        nruns > 0 || throw(
            ArgumentError("A positive `nruns` must be set or `init` must be provided.")
        )
        _init = fill(init, nruns)
    else
        _init = init
    end
    if ndraws > ndraws_per_run * nruns
        @warn "More draws requested than total number of draws across replicas. Draws will not be unique."
    end
    logp(x) = -optim_fun.f(x, nothing)

    # run pathfinder independently from each starting point
    trans = Transducers.Map() do (init_i)
        return pathfinder(
            optim_fun;
            rng,
            history_length,
            optimizer,
            ndraws=ndraws_per_run,
            init=init_i,
            kwargs...,
        )
    end
    iter_sp = Transducers.withprogress(_init; interval=1e-3) |> trans
    pathfinder_results = Folds.collect(iter_sp, executor)
    fit_distributions =
        pathfinder_results |> Transducers.Map(x -> x.fit_distribution) |> collect
    draws_all = reduce(hcat, pathfinder_results |> Transducers.Map(x -> x.draws))

    # draw samples from augmented mixture model
    inds = axes(draws_all, 2)
    sample_inds, psis_result = if importance
        log_densities_fit =
            pathfinder_results |>
            Transducers.MapCat() do x
                return Distributions.logpdf(x.fit_distribution, x.draws)
            end |>
            collect
        iter_logp = eachcol(draws_all) |> Transducers.Map(logp)
        log_densities_target = Folds.collect(iter_logp, executor)
        log_densities_ratios = log_densities_target - log_densities_fit
        resample(rng, inds, log_densities_ratios, ndraws)
    else
        resample(rng, inds, ndraws), nothing
    end

    fit_distribution = Distributions.MixtureModel(fit_distributions)
    draws = draws_all[:, sample_inds]

    # get component ids (k) of draws in ϕ
    draw_component_ids = cld.(sample_inds, ndraws_per_run)

    # placeholders
    fit_distribution_transformed = fit_distribution
    draws_transformed = draws

    return MultiPathfinderResult(
        input,
        optimizer,
        rng,
        optim_fun,
        logp,
        fit_distribution,
        draws,
        draw_component_ids,
        fit_distribution_transformed,
        draws_transformed,
        pathfinder_results,
        psis_result,
    )
end
