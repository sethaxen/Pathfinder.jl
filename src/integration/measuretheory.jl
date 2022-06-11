using .MeasureTheory: MeasureTheory
using .TransformVariables: TransformVariables
using Accessors: Accessors

function pathfinder(measure::MeasureTheory.AbstractMeasure; kwargs...)
    logp = MeasureTheoryLogDensity(measure)
    tr = logp.transform
    dim = TransformVariables.dimension(tr)
    result = pathfinder(logp; dim, input=measure, kwargs...)
    fit_dist = result.fit_distribution
    fit_measure = MeasureTheory.MvNormal(fit_dist.μ, fit_dist.Σ)
    fit_measure_transformed, draws_transformed = _transform_measure_and_draws(
        tr, fit_measure, result.draws
    )
    result_new = Accessors.@set result.fit_distribution_transformed =
        fit_measure_transformed
    result_new2 = Accessors.@set result_new.draws_transformed = draws_transformed
    return result_new2
end

function multipathfinder(measure::MeasureTheory.AbstractMeasure, ndraws::Int; kwargs...)
    logp = MeasureTheoryLogDensity(measure)
    tr = logp.transform
    dim = TransformVariables.dimension(tr)
    result = multipathfinder(logp, ndraws; dim, kwargs...)
    fit_dist = result.fit_distribution
    fit_measure_components = map(fit_dist.components) do c
        return MeasureTheory.MvNormal(c.μ, c.Σ)
    end
    fit_measure = MeasureTheory.SuperpositionMeasure(fit_measure_components)
    fit_measure_transformed, draws_transformed = _transform_measure_and_draws(
        tr, fit_measure, result.draws
    )
    result_new = Accessors.@set result.fit_distribution_transformed =
        fit_measure_transformed
    result_new2 = Accessors.@set result_new.draws_transformed = draws_transformed
    return result_new2
end

function _transform_measure_and_draws(transform, measure, draws)
    measure_transformed = MeasureTheory.Pushforward(transform, measure)
    draws_transformed = TransformVariables.transform.(Ref(transform), eachcol(draws))
    return measure_transformed, draws_transformed
end

struct MeasureTheoryLogDensity{M,T}
    measure::M
    transform::T
end
function MeasureTheoryLogDensity(m::MeasureTheory.AbstractMeasure)
    return MeasureTheoryLogDensity(m, MeasureTheory.as(m))
end

function (logp::MeasureTheoryLogDensity)(z)
    x, logdetJ = TransformVariables.transform_and_logjac(logp.transform, z)
    return MeasureTheory.logdensityof(logp.measure, x) + logdetJ
end
function (logp::MeasureTheoryLogDensity{M,<:TransformVariables.ScalarTransform})(
    z
) where {M}
    x, logdetJ = TransformVariables.transform_and_logjac(logp.transform, z[1])
    return MeasureTheory.logdensityof(logp.measure, x) + logdetJ
end
