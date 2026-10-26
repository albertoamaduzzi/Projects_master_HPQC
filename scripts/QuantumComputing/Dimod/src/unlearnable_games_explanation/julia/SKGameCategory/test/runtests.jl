using Test
using SKGameCategory
using Catlab

# literal, auto-generated presentations (must parse with Catlab)
include(joinpath(@__DIR__, "..", "..", "..", "category", "SKGamePaperLiteral.jl"))

const JSONPATH = joinpath(@__DIR__, "..", "..", "..", "category", "category.json")

@testset "SKGameCategory" begin
    C = load_category(JSONPATH)
    rep = check_category(C)
    @test rep.n_objects == length(C.json[:objects])
    @test isempty(rep.orphans)
    order = topo_order(C)
    pos = Dict(n => i for (i, n) in enumerate(order))
    for h in C.json[:generators], d in h[:dom]
        @test pos[String(d)] < pos[String(h[:codom])]
    end
    # Fig. 17 needs the simulations, the static RS solution and the finite-beta DMFT
    @test upstream(C, "Fig17OrderParameter") == sort(["QvsTemperature", "StaticDMFTSolution", "DMFTFiniteBeta"])
    @test "Fig14Eps1Decorrelation" in downstream(C, "AnalyticCurves")
    f = derivation(C, "Fig03Autocorrelation")
    @test dom(f) == munit(FreeSymmetricMonoidalCategory.Ob)
    @test codom(f) == generator(C.outputs, :Fig03Autocorrelation)
    # impact analysis through the Uses span
    imp = impacted_by(C, "dmft_solve")
    @test "Fig11C2vsEps" in imp && "Fig17OrderParameter" in imp && !("Fig05RewardTrajectories" in imp)
    @test "optimise" in morphisms_of(C, "PlannerBenchmark")
    # the literal presentations agree with the ones built from JSON
    @test length(generators(SKGamePaperLiteral, :Hom)) == length(generators(C.outputs, :Hom))
    @test length(generators(SKGameMathLiteral, :Hom)) == length(generators(C.math, :Hom))
    write_graphviz(C, joinpath(mktempdir(), "g.dot"))
end
