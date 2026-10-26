"""The staged campaign of main_simulation.py and the annealer's full-statistics call."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

dimod = pytest.importorskip("dimod")
pytest.importorskip("dwave.samplers")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import main_simulation as ms  # noqa: E402
from skgame.optim.problem import PlannerProblem  # noqa: E402
from skgame.optim.quantum import QuantumAnnealingSolver  # noqa: E402
from skgame.optim.regimes import REGIMES, problem_classes, regimes_by_name  # noqa: E402
from skgame.optim.solvers import ExactEnumeration  # noqa: E402


def test_regimes_cover_easy_and_hard_phases():
    phases = {(r.planner_phase, r.game_phase) for r in REGIMES}
    assert ("ferromagnetic", "fixed_points") in phases
    assert ("spin_glass", "fixed_points") in phases
    assert {r.game_phase for r in REGIMES} == {"fixed_points", "cycles", "chaotic", "paramagnet"}
    # regimes differing only in (alpha, beta) share instances: 6 regimes, 4 problem classes
    assert len(problem_classes(REGIMES)) == 4
    with pytest.raises(KeyError):
        regimes_by_name(["nope"])


def test_sample_problem_keeps_every_read_and_skips_seed_when_unsupported():
    P = PlannerProblem.random(8, rng=1)
    qa = QuantumAnnealingSolver(sampler_factory=dimod.ExactSolver, use_simulator=True, num_reads=1)
    qa.sample_kwargs.pop("num_reads")          # ExactSolver takes no kwargs at all, not even a seed
    r = qa.sample_problem(P, rng=0)
    assert sum(r["counts"]) == 2 ** 8
    assert r["best_reward"] == pytest.approx(ExactEnumeration().solve(P).reward)
    assert r["rewards"] == sorted(r["rewards"], reverse=True)
    S = np.array([1.0 if c == "+" else -1.0 for c in r["best_S"]])
    assert P.reward(S) == pytest.approx(r["best_reward"])


def test_campaign_stages_end_to_end(tmp_path):
    out = tmp_path / "c"
    ms.main(["init", "--preset", "smoke", "--out", str(out), "--Ns", "8,12", "--n-instances", "2"])
    ms.main(["classical", "--out", str(out), "--task", "0"])
    ms.main(["classical", "--out", str(out)])
    first = ms.classical_path(out, "eps0_J03", 8).stat().st_mtime_ns
    ms.main(["classical", "--out", str(out)])                       # resume: nothing recomputed
    assert ms.classical_path(out, "eps0_J03", 8).stat().st_mtime_ns == first
    ms.main(["sim", "--out", str(out), "--max-calls", "3"])
    assert len(list((out / "anneal" / "sim_logical").rglob("*.json"))) == 3
    ms.main(["sim", "--out", str(out)])
    assert len(list((out / "anneal" / "sim_logical").rglob("*.json"))) == 2 * 4 * 2
    ms.main(["analyze", "--out", str(out)])
    rows = ms.collect(out, ms.load_config(out))
    kinds = {r["kind"] for r in rows}
    assert kinds == {"classical", "anneal", "game"}
    for r in rows:
        assert 0.0 <= r["p_success_median"] <= 1.0
        assert r["residual_median"] >= -1e-9          # nobody beats the reference
    exact = [r for r in rows if r["kind"] == "anneal" and r["N"] == 8]
    assert all(r["ref_exact"] and r["ref_from_annealer"] == 0 for r in exact)
    assert (out / "summary" / "summary.csv").exists()
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["Ns"] == [8, 12]


def test_zephyr_and_pegasus_qpu_backends_compared(tmp_path, monkeypatch):
    """The two QPU back-ends end to end against D-Wave's mock QPU (no token)."""
    import dwave.system
    from dwave.system.testing import MockDWaveSampler

    def fake(solver=None, **kw):
        topo = solver["topology__type"]
        return MockDWaveSampler(topology_type=topo, topology_shape=[3, 4] if topo == "zephyr" else [4])

    monkeypatch.setattr(dwave.system, "DWaveSampler", fake)
    out = tmp_path / "q"
    ms.main(["init", "--out", str(out), "--Ns", "8,12", "--n-instances", "1", "--regimes", "sg_satisficing",
             "--chain-break-method", "minimize_energy"])
    ms.main(["classical", "--out", str(out)])
    ms.main(["qpu", "--out", str(out)])
    for label in ("qpu_zephyr_clique", "qpu_pegasus_clique"):
        files = sorted((out / "anneal" / label).rglob("*.json"))
        assert len(files) == 2
        r = json.loads(files[-1].read_text())
        assert r["info"]["embedding"]["n_qubits"] >= 12
        assert r["info"]["settings"]["chain_break_method"] == "minimize_energy"
        assert r["polished_best_reward"] >= r["best_reward"] - 1e-12
    rows = ms.stage_analyze(out, ms.load_config(out), log=lambda *a: None)
    comp = ms.compare_backends(rows)
    assert {c["N"] for c in comp} == {8, 12}
    assert all(c["device"] == "qpu" for c in comp)
    assert (out / "summary" / "backend_comparison.csv").exists()
