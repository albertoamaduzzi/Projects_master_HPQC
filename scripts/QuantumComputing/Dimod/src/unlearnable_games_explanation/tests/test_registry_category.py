import json
from pathlib import Path

import numpy as np
import pytest

import skgame.outputs  # noqa: F401
from skgame.registry import (REGISTRY, topological_order, upstream, downstream, derivation_expr, category_dict,
                             export_category, Pipeline, RunContext, Output, FigureOutput)
from skgame.mathcat import OBJECTS, MORPHISMS, OUTPUT_USES, check_well_typed

PAPER_FIGURES = [f"Fig{k:02d}" for k in range(1, 20)]


def test_every_paper_figure_has_a_class():
    names = list(REGISTRY)
    for f in PAPER_FIGURES:
        assert any(n.startswith(f) for n in names), f


def test_category_is_acyclic_and_typed():
    order = topological_order()
    pos = {n: i for i, n in enumerate(order)}
    for n, c in REGISTRY.items():
        for d in c.depends_on:
            assert pos[d.__name__] < pos[n]
    # every non-figure, non-table object feeds at least one figure
    for n, c in REGISTRY.items():
        if c.kind not in ("figure", "table"):
            assert any(REGISTRY[m].kind == "figure" for m in downstream(n)), n
    # figures only consume, never produce for others
    for n, c in REGISTRY.items():
        if c.kind == "figure":
            assert not downstream(n)


def test_derivation_and_queries():
    assert upstream("Fig17OrderParameter") == {"QvsTemperature", "StaticDMFTSolution", "DMFTFiniteBeta"}
    assert "Fig14Eps1Decorrelation" in downstream("AnalyticCurves")
    e = derivation_expr("Fig03Autocorrelation")
    assert e.startswith("make_Fig03Autocorrelation ∘") and "make_DMFTAlpha1" in e


def test_math_category_and_uses():
    assert check_well_typed()
    non_fig = {n for n, c in REGISTRY.items() if c.kind != "figure"}
    assert non_fig == set(OUTPUT_USES)


def test_export(tmp_path):
    paths = export_category(tmp_path)
    d = json.loads(paths["json"].read_text())
    assert len(d["objects"]) == len(REGISTRY) and len(d["generators"]) == len(REGISTRY)
    assert len(d["math"]["objects"]) == len(OBJECTS) and len(d["math"]["generators"]) == len(MORPHISMS)
    jl = paths["catlab"].read_text()
    assert "@present SKGamePaperLiteral(FreeSymmetricMonoidalCategory)" in jl and "munit()" in jl


def test_pipeline_cache_and_routing(tmp_path):
    ctx = RunContext(out_dir=tmp_path, preset="quick", verbose=False)
    res = Pipeline(ctx).run(["Fig10CycleComplexity"])
    assert (tmp_path / "theory" / "CycleComplexity" / "result.npz").exists()
    assert list((tmp_path / "figure" / "Fig10CycleComplexity").glob("*.png"))
    meta = json.loads((tmp_path / "theory" / "CycleComplexity" / "meta.json").read_text())
    assert meta["status"] == "ok" and meta["kind"] == "theory"
    # second run is fully cached
    pl = Pipeline(ctx)
    pl.run(["Fig10CycleComplexity"])
    assert pl.timings == {}


def test_failure_isolation(tmp_path):
    class _Boom(Output):
        kind = "data"

        def compute(self, deps):
            raise RuntimeError("boom")

    class _AfterBoom(FigureOutput):
        depends_on = (_Boom,)

        def compute(self, deps):
            return []
    try:
        ctx = RunContext(out_dir=tmp_path, verbose=False)
        pl = Pipeline(ctx)
        pl.run(["_AfterBoom"])
        assert "_Boom" in pl.failed and "_AfterBoom" in pl.failed
    finally:
        REGISTRY.pop("_Boom"); REGISTRY.pop("_AfterBoom")
