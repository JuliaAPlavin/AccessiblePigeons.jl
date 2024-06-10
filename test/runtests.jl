using TestItems
using TestItemRunner
@run_package_tests

@testitem "basic" begin
    using Distributions
    using StructArrays
    using ForwardDiff
    using MCMCChains: Chains

    struct ExpModel{A,B}
        scale::A
        shift::B
    end
    
    struct SumModel{T <: Tuple}
        comps::T
    end
    
    (m::ExpModel)(x) = m.scale * exp(-(x - m.shift)^2)
    (m::SumModel)(x) = sum(c -> c(x), m.comps)

    loglike(m::SumModel, data) = sum(r -> pdf(Normal(m(r.x), 1), r.y), data)

    truemod = SumModel((
        ExpModel(2, 5),
        ExpModel(0.5, 2),
        ExpModel(0.5, 8),
    ))

    data = let x = 0:0.2:10
        StructArray(; x, y=truemod.(x) .+ range(-0.01, 0.01, length=length(x)))
    end
    
    mod0 = SumModel((
        ExpModel(1., 1.),
        ExpModel(1., 2.),
        ExpModel(1., 3.),
    ))
    target = AccessiblePotential(Base.Fix2(loglike, data), (
        (@o _.comps[∗].shift) => Uniform(0, 10),
        (@o _.comps[∗].scale) => Uniform(0.3, 10),
    ), mod0)
    record=[traces; round_trip; record_default()]
    @testset for kwargs in [
            (; n_rounds=8, multithreaded=false, record),
            (; n_rounds=8, multithreaded=true, record),
            (; n_rounds=8, explorer=AutoMALA(default_autodiff_backend=:ForwardDiff), record),
            # (; n_rounds=8, explorer=AutoMALA(default_autodiff_backend=:Enzyme), record),
        ]
        pt = pigeons(; target, kwargs...)

        chs = Chains(pt)
        @test size(chs) == (2^8, 7, 1)
        @test names(chs) == [Symbol("comps[1].shift"), Symbol("comps[2].shift"), Symbol("comps[3].shift"), Symbol("comps[1].scale"), Symbol("comps[2].scale"), Symbol("comps[3].scale"), :log_density]

        ss = samples(pt)
        @test length(ss) == 2^8
        @test ss[1] isa SumModel
        @test 0 ≤ ss[1].comps[1].shift ≤ 10
    end
end

@testitem "_" begin
    import Aqua
    Aqua.test_all(AccessiblePigeons; ambiguities=false, piracies=false)
    Aqua.test_ambiguities(AccessiblePigeons)

    import CompatHelperLocal as CHL
    CHL.@check()
end
