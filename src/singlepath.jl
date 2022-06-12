
"""
    PathfinderResult

Container for results of single-path Pathfinder.

# Fields
- `input`: User-provided input object, e.g. either `logp`, `(logp, ∇logp)`, `optim_fun`,
    `optim_prob`, or another object.
- `optimizer`: Optimizer used for maximizing the log-density
- `rng`: Pseudorandom number generator that was used for sampling
- `optim_prob::SciMLBase.OptimizationProblem`: Otimization problem used for
    optimization
- `logp`: Log-density function
- `dist_optimizer`: Optimizer used to compute `fit_iteration` and `fit_distribution` from
    `logp`, `optim_solution`, `optim_trace`, and `fit_distributions`.
- `fit_distribution::Distributions.MvNormal`: multivariate normal distribution returned by
    `dist_optimizer`
- `draws::AbstractMatrix{<:Real}`: draws from multivariate normal with size `(dim, ndraws)`
- `fit_distribution_transformed`: `fit_distribution` transformed to the same space as the
    user-supplied target distribution. This is only different from `fit_distribution` when
    integrating with other packages, and its type depends on the type of `input`.
- `draws_transformed`: `draws` transformed to be draws from `fit_distribution_transformed`.
- `fit_iteration::Int`: Iteration at which `fit_distribution` occurred
- `num_tries::Int`: Number of tries until Pathfinder succeeded
- `optim_solution::SciMLBase.OptimizationSolution`: Solution object of optimization.
- `optim_trace::Pathfinder.OptimizationTrace`: container for optimization trace of points,
    log-density, and gradient. The first point is the initial point.
- `fit_distributions::AbstractVector{Distributions.MvNormal}`: Multivariate normal
    distributions for each point in `optim_trace`, where
    `fit_distributions[fit_iteration + 1] == fit_distribution`
- `fit_stats`: Additional statistics computed by `dist_optimizer`.

# Returns
- [`PathfinderResult`](@ref)
"""
struct PathfinderResult{I,O,R,OP,LP,DO,FD,D,FDT,DT,FI,OS,OT,FS}
    input::I
    optimizer::O
    rng::R
    optim_prob::OP
    logp::LP
    dist_optimizer::DO
    fit_distribution::FD
    draws::D
    fit_distribution_transformed::FDT
    draws_transformed::DT
    fit_iteration::FI
    num_tries::Int
    optim_solution::OS
    optim_trace::OT
    fit_distributions::Vector{FD}
    fit_stats::FS
end

function Base.show(io::IO, ::MIME"text/plain", result::PathfinderResult)
    println(io, "Single-path Pathfinder result")
    println(io, "  tries: $(result.num_tries)")
    println(io, "  draws: $(size(result.draws, 2))")
    println(
        io,
        "  fit iteration: $(string(result.fit_iteration)) (total: $(length(result.optim_trace) - 1))",
    )
    if result.fit_stats isa AbstractVector{<:ELBOEstimate}
        println(io, "  fit ELBO: $(_to_string(result.fit_stats[result.fit_iteration]))")
    end
    print(io, "  fit distribution: ", result.fit_distribution)
    return nothing
end

"""
    pathfinder(logp; kwargs...)
    pathfinder(logp, ∇logp; kwargs...)
    pathfinder(fun::SciMLBase::OptimizationFunction; kwargs...)
    pathfinder(prob::SciMLBase::OptimizationProblem; kwargs...)

Find the best multivariate normal approximation encountered while maximizing `logp`.

From an optimization trajectory, Pathfinder constructs a sequence of (multivariate normal)
approximations to the distribution specified by `logp`.

By default, the approximation that maximizes the evidence lower bound (ELBO), or
equivalently, minimizes the KL divergence between the approximation and the true
distribution, is returned. This can be changed by providing a different `dist_optimizer`.

The covariance of the multivariate normal distribution is an inverse Hessian approximation
constructed using at most the previous `history_length` steps.

# Arguments
- `logp`: a callable that computes the log-density of the target distribution.
- `∇logp`: a callable that computes the gradient of `logp`. If not provided, `logp` is
    automatically differentiated using the backend specified in `ad_backend`.
- `fun::SciMLBase.OptimizationFunction`: an optimization function that represents
    `-logp(x)` with its gradient. It must have the necessary features (e.g. a Hessian
    function) for the chosen optimization algorithm. For details, see
    [Optimization.jl: OptimizationFunction](https://optimization.sciml.ai/stable/API/optimization_function/).
- `prob::SciMLBase.OptimizationProblem`: an optimization problem containing a function with
    the same properties as `fun`, as well as an initial point, in which case `init` and
    `dim` are ignored.

# Keywords
- `dim::Int`: dimension of the target distribution. If not provided, `init` or must be.
    Ignored if `init` is provided.
- `init::AbstractVector{<:Real}`: initial point of length `dim` from which to begin
    optimization. If not provided, an initial point of type `Vector{Float64}` and length
    `dim` is created and filled using `init_sampler`.
- `init_scale::Real`: scale factor ``s`` such that the default `init_sampler` samples
    entries uniformly in the range ``[-s, s]``
- `init_sampler`: function with the signature `(rng, x) -> x` that modifies a vector of
    length `dims` in-place to generate an initial point
- `ndraws::Int`: number of approximate draws to return. Defaults to 0 unless draws from
    `dist_optimizer` can be resued.
- `ad_backend=AD.ForwardDiffBackend()`: AbstractDifferentiation.jl AD backend.
- `rng::Random.AbstractRNG`: The random number generator to be used for drawing samples
- `history_length::Int=$DEFAULT_HISTORY_LENGTH`: Size of the history used to approximate the
    inverse Hessian.
- `optimizer`: Optimizer to be used for constructing trajectory. Can be any optimizer
    compatible with Optimization.jl, so long as it supports callbacks. Defaults to
    `Optim.LBFGS(; m=history_length, linesearch=LineSearches.MoreThuente())`. See
    the [Optimization.jl documentation](https://optimization.sciml.ai/stable) for details.
- `dist_optimizer`: Callable that selects the returned distribution. Its signature must be
    `(logp, optim_solution, optim_trace, fit_distributions) -> (success, fit_distribution, fit_iteration, fit_stats)`, where
    `optim_solution` is a `SciMLBase.OptimizationSolution`,
    `optim_trace` is a `Pathfinder.OptimizationTrace`,
    `success` indicates whether the optimization succeeded,
    `fit_iteration` is an integer such that
    `fit_distributions[iteration + 1] === fit_distribution` or `nothing`, and
    `fit_stats` is an arbitrary container of statistics computed during optimization. If
    the optimizer draws from `fit_distribution`, these draws can be reused by storing them
    in `fit_stats` and overloading [`Pathfinder.draws_from_fit_stats`](@ref).
    Defaults to [`Pathfinder.MaximumELBO`](@ref).
- `ntries::Int=1_000`: Number of times to try the optimization, restarting if it fails. Before
    every restart, a new initial point is drawn using `init_sampler`.
- `kwargs...` : Remaining keywords are forwarded to
    [`Optimization.solve`](https://optimization.sciml.ai/stable/API/solve).

# Returns
- [`PathfinderResult`](@ref)
"""
function pathfinder end

function pathfinder(logp; ad_backend=AD.ForwardDiffBackend(), kwargs...)
    return pathfinder(build_optim_function(logp; ad_backend); input=logp, kwargs...)
end
function pathfinder(logp, ∇logp; ad_backend=AD.ForwardDiffBackend(), kwargs...)
    return pathfinder(
        build_optim_function(logp, ∇logp; ad_backend); input=(logp, ∇logp), kwargs...
    )
end
function pathfinder(
    optim_fun::SciMLBase.OptimizationFunction;
    rng=Random.GLOBAL_RNG,
    init=nothing,
    dim::Int=-1,
    init_scale=2,
    init_sampler=UniformSampler(init_scale),
    input=optim_fun,
    kwargs...,
)
    if init !== nothing
        _init = init
        allow_mutating_init = false
    elseif init === nothing && dim > 0
        _init = Vector{Float64}(undef, dim)
        init_sampler(rng, _init)
        allow_mutating_init = true
    else
        throw(ArgumentError("An initial point `init` or dimension `dim` must be provided."))
    end
    prob = build_optim_problem(optim_fun, _init)
    return pathfinder(prob; rng, input, init_sampler, allow_mutating_init, kwargs...)
end
function pathfinder(
    prob::SciMLBase.OptimizationProblem;
    rng::Random.AbstractRNG=Random.GLOBAL_RNG,
    history_length::Int=DEFAULT_HISTORY_LENGTH,
    optimizer=default_optimizer(history_length),
    dist_optimizer=MaximumELBO(; rng),
    ndraws::Int=dist_optimizer isa MaximumELBO{true} ? dist_optimizer.ndraws : 0,
    input=prob,
    kwargs...,
)
    if prob.f.grad === nothing || prob.f.grad isa Bool
        throw(ArgumentError("optimization function must define a gradient function."))
    end
    logp(x) = -prob.f.f(x, nothing)
    path_result = ProgressLogging.progress(; name="Optimizing") do progress_id
        return _pathfinder_try_until_succeed(
            rng,
            prob,
            logp;
            history_length,
            optimizer,
            progress_id,
            dist_optimizer,
            kwargs...,
        )
    end
    @unpack (
        itry,
        success,
        optim_solution,
        optim_trace,
        fit_distributions,
        fit_distribution,
        fit_iteration,
        fit_stats,
    ) = path_result

    success ||
        @warn "Pathfinder failed after $itry tries. Increase `ntries`, inspect the model for numerical instability, or provide a more suitable `init_sampler`."

    draws = if success
        stats_draws = draws_from_fit_stats(fit_stats, fit_iteration)
        if stats_draws !== nothing
            # reuse existing draws if available; draw additional ones if necessary
            ndraws_stats = size(stats_draws, 2)
            if ndraws_stats < ndraws
                hcat(stats_draws, rand(rng, fit_distribution, ndraws - ndraws_stats))
            else
                stats_draws[:, 1:ndraws]
            end
        else
            rand(rng, fit_distribution, ndraws)
        end
    else
        rand(rng, fit_distribution, ndraws)
    end

    # placeholders
    fit_distribution_transformed = fit_distribution
    draws_transformed = draws

    return PathfinderResult(
        input,
        optimizer,
        rng,
        optim_solution.prob,
        logp,
        dist_optimizer,
        fit_distribution,
        draws,
        fit_distribution_transformed,
        draws_transformed,
        fit_iteration,
        itry,
        optim_solution,
        optim_trace,
        fit_distributions,
        fit_stats,
    )
end

function _pathfinder_try_until_succeed(
    rng,
    prob,
    logp;
    ntries::Int=1_000,
    init_scale=2,
    init_sampler=UniformSampler(init_scale),
    allow_mutating_init::Bool=false,
    kwargs...,
)
    itry = 1
    progress_name = "Optimizing (try 1)"
    result = _pathfinder(prob, logp; progress_name, kwargs...)
    _prob = prob
    while !result.success && itry < ntries
        if itry == 1 && !allow_mutating_init
            _prob = deepcopy(prob)
        end
        itry += 1
        init_sampler(rng, _prob.u0)
        progress_name = "Optimizing (try $itry)"
        result = _pathfinder(_prob, logp; progress_name, kwargs...)
    end
    return (; itry, result...)
end

function _pathfinder(
    prob,
    logp;
    history_length::Int=DEFAULT_HISTORY_LENGTH,
    optimizer=default_optimizer(history_length),
    dist_optimizer=MaximumELBO(; rng),
    kwargs...,
)
    # compute trajectory
    optim_solution, optim_trace = optimize_with_trace(prob, optimizer; kwargs...)

    # fit mv-normal distributions to trajectory
    fit_distributions = fit_mvnormals(
        optim_trace.points, optim_trace.gradients; history_length
    )

    success, fit_distribution, fit_iteration, fit_stats = dist_optimizer(
        logp, optim_solution, optim_trace, fit_distributions
    )
    success &= length(optim_trace) > 1

    return (;
        success,
        optim_solution,
        optim_trace,
        fit_distributions,
        fit_distribution,
        fit_iteration,
        fit_stats,
    )
end

"""
    UniformSampler(scale::Real)

Sampler that in-place modifies an array to be IID uniformly distributed on `[-scale, scale]`
"""
struct UniformSampler{T<:Real}
    scale::T
    function UniformSampler(scale::T) where {T<:Real}
        scale > 0 || throw(DomainError(scale, "scale of uniform sampler must be positive."))
        return new{T}(scale)
    end
end

function (s::UniformSampler)(rng::Random.AbstractRNG, point)
    scale = s.scale
    @. point = rand(rng) * 2scale - scale
    return point
end
