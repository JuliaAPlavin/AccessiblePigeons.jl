module AccessiblePigeons

using Reexport
using Test: @inferred
@reexport using Pigeons
using Random: AbstractRNG
using DataManipulation
using Distributions
@reexport using AccessorsExtra

export AccessiblePotential, samples


struct AccessiblePotential{F,PD,OB,OP}
    func::F
    # priors::PS
    prior::PD
    obj₀::OB
    optics::OP
end

function AccessiblePotential(func, specs, objtype::Type=Any)
    priors = map(last, specs)
    obj₀ = construct(objtype, map(s -> first(s) => mean(last(s)), specs)...)
    optics = concat(first.(specs)...)
	@inferred func(obj₀)
    # @assert isconcretetype(eltype(collect(getall(obj₀, optics))))
    @assert all(specs) do s
        length(getall(obj₀, first(s))) == 1
    end
    AccessiblePotential(
        func,
        # priors,
        product_distribution(priors...),
        obj₀,
        optics,
    )
end

function AccessiblePotential(func, specs, obj₀)
    optics = concat(first.(specs)...)
	@inferred func(obj₀)
    # @assert isconcretetype(eltype(collect(getall(obj₀, optics))))
    priors = flatmap(specs) do (optic, dist)
        n = length(getall(obj₀, optic))
        fill(dist, n)
    end
    AccessiblePotential(
        func,
        # priors,
        product_distribution(priors...),
        obj₀,
        optics,
    )
end

obj(p::AccessiblePotential, x) = setall(p.obj₀, p.optics, x)
function (p::AccessiblePotential)(x)
    prior = logpdf(p.prior, x)
    prior == -Inf && return prior
    like = p.func(obj(p, x))
    return prior + like
end

Pigeons.default_reference(p::AccessiblePotential) = Pigeons.DistributionLogPotential(p.prior)
Pigeons.initialization(p::AccessiblePotential, ::AbstractRNG, ::Int) = collect(getall(p.obj₀, p.optics))

Pigeons.LogDensityProblems.dimension(p::AccessiblePotential) = length(getall(p.obj₀, p.optics))
Pigeons.LogDensityProblems.logdensity(p::AccessiblePotential, x) = p(x)

samples(pt) = map(x -> obj(pt.inputs.target, x), get_sample(pt))


Pigeons.sample_names(_::Array, p::AccessiblePotential) = flatmap(p.optics.optics) do o
    # n = length(getall(p.obj₀, o))
    basename = @p let
		AccessorsExtra.flat_concatoptic(p.obj₀, o)
		AccessorsExtra._optics
		map(AccessorsExtra.barebones_string)
    end
end |> collect

Pigeons.sample_names(x::Array, p::Pigeons.InterpolatedLogPotential) = [sample_names(x, p.path.target); :log_density]

end
