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


def test_qpu_budget_guard_logic():
    b = ms.QPUBudget(1.0, safety=1.5)
    assert b.allows(0.6) and not b.allows(0.7)            # 1.5 x 0.7 > 1
    b.charge(0.4, 0.2)                                     # charged twice the estimate
    assert b.spent == 0.4 and b.factor() == pytest.approx(2.2)
    assert not b.allows(0.3)                               # 0.4 + 2.2 x 0.3 > 1
    assert ms.QPUBudget(None).allows(1e9)


def test_qpu_budget_stops_before_overspending(tmp_path, monkeypatch, capsys):
    import dwave.system
    from dwave.system.testing import MockDWaveSampler
    monkeypatch.setattr(dwave.system, "DWaveSampler",
                        lambda solver=None, **kw: MockDWaveSampler(topology_type="zephyr", topology_shape=[3, 4]))
    out = tmp_path / "b"
    ms.main(["init", "--out", str(out), "--Ns", "8,12", "--n-instances", "1", "--regimes", "sg_satisficing",
             "--num-reads", "100", "--backends", "qpu:zephyr:clique"])
    ms.main(["qpu", "--out", str(out), "--budget-seconds", "0.01"])      # less than one call
    assert not list((out / "anneal").rglob("*.json")) if (out / "anneal").exists() else True
    ms.main(["qpu", "--out", str(out), "--budget-seconds", "1"])
    files = list((out / "anneal" / "qpu_zephyr_clique").rglob("*.json"))
    assert len(files) == 2
    assert all("estimated_access_s" in json.loads(f.read_text())["info"] for f in files)
    rows = ms.qpu_spent(out, log=lambda *a: None)
    assert rows["qpu_zephyr_clique"]["calls"] == 2 and rows["qpu_zephyr_clique"]["spent_s"] < 1


def test_ground_states_report(tmp_path):
    out = tmp_path / "g"
    ms.main(["init", "--out", str(out), "--Ns", "10,14", "--n-instances", "2", "--regimes", "sg_satisficing",
             "--backends", "sim:zephyr:clique"])
    ms.main(["classical", "--out", str(out)])
    ms.main(["sim", "--out", str(out), "--backend", "sim:zephyr:clique"])
    rows = ms.ground_states(out, ms.load_config(out), log=lambda *a: None)
    assert len(rows) == 4 and all(r["certified"] and r["found_by"] == "exact" for r in rows)
    for r in rows:
        P = PlannerProblem(ms.ensure_instances(out, ms.load_config(out), "eps0_J00", 0.0, 0.0, r["N"])[r["instance"]])
        S = np.array([1.0 if c == "+" else -1.0 for c in r["ground_state"]])
        assert r["ground_state"][0] == "+"
        assert P.reward(S) == pytest.approx(ExactEnumeration().solve(P).reward)
        assert r["energy"] == pytest.approx(P.energy(S)) and r["classical:hamming"] == 0
    assert (out / "summary" / "ground_states.csv").exists()


def test_records_are_complete_and_validate(tmp_path, monkeypatch):
    """Everything needed to reconstruct a QPU call a posteriori is saved, and validate re-derives it."""
    import dwave.system
    from dwave.system.testing import MockDWaveSampler
    monkeypatch.setattr(dwave.system, "DWaveSampler",
                        lambda solver=None, **kw: MockDWaveSampler(topology_type="zephyr", topology_shape=[3, 4]))
    runs = []
    for pf in ("0.5", "1.414"):
        out = tmp_path / f"p{pf}"
        ms.main(["init", "--out", str(out), "--Ns", "8,12", "--n-instances", "1", "--regimes", "sg_satisficing",
                 "--num-reads", "50", "--backends", "qpu:zephyr:clique", "--chain-strength-prefactor", pf])
        ms.main(["qpu", "--out", str(out), "--budget-seconds", "1"])
        runs.append(out)
    r = json.loads(next((runs[0] / "anneal" / "qpu_zephyr_clique").rglob("N0012_i0.json")).read_text())
    info = r["info"]
    for k in ("problem_id", "embedding_used", "solver_properties", "programmed", "chain_strength", "chain_lengths"):
        assert k in info, k
    assert info["sample_kwargs"]["label"].startswith("skgame")
    assert len(r["reads"]["S"]) == len(r["reads"]["reward"]) and len(r["reads"]["S"][0]) == 12
    prog = info["programmed"]                          # range [-1, 1]: divide by the largest |J|, chain or logical
    assert prog["auto_scale_divisor"] == pytest.approx(prog["max_abs_J_before_scaling"])
    assert prog["auto_scale_divisor"] >= info["chain_strength"] - 1e-12
    tally = ms.validate(runs[0], ms.load_config(runs[0]), log=lambda *a: None)
    assert tally and all(bad == 0 for ok, bad in tally.values())
    rows = ms.scan_summary(runs, log=lambda *a: None)
    assert {row["prefactor"] for row in rows} == {0.5, 1.414} and (tmp_path / "scan_summary.csv").exists()


def test_choose_coupling_rule_and_interpolation():
    from dwave_utils import coupling_scan as cs
    rows = [{"p": 0.5, "broken_fraction": 0.2}, {"p": 1.0, "broken_fraction": 0.0},
            {"p": 1.414, "broken_fraction": 0.0}, {"p": 2.0, "broken_fraction": 0.0}]
    assert cs.choose_coupling(rows)[0] == 1.0                      # smallest p with no broken chain
    assert cs.choose_coupling(rows, threshold=0.25)[0] == 0.5      # looser threshold -> smaller p
    p, why = cs.choose_coupling([{"p": 0.5, "broken_fraction": 0.3}, {"p": 1.0, "broken_fraction": 0.1}])
    assert p == 1.0 and "LARGEST" in why                           # never reaches 0: fewest, flagged
    table = {"qpu_zephyr_clique": {"8": 0.5, "48": 1.5, "104": 2.0}}
    assert cs.coupling_for(table, "qpu_zephyr_clique", 28, 9.9) == pytest.approx(1.0)   # interpolated
    assert cs.coupling_for(table, "qpu_zephyr_clique", 160, 9.9) == 2.0                 # beyond: last value
    assert cs.coupling_for(table, "qpu_pegasus_clique", 64, 1.414) == 1.414             # no table: default


def test_coupling_scan_end_to_end_and_main_run_uses_it(tmp_path, monkeypatch):
    import dwave.system
    from dwave.system.testing import MockDWaveSampler
    monkeypatch.setattr(dwave.system, "DWaveSampler",
                        lambda solver=None, **kw: MockDWaveSampler(topology_type="zephyr", topology_shape=[3, 4]))
    out = tmp_path / "c"
    ms.main(["init", "--out", str(out), "--Ns", "8:24:8", "--n-instances", "1", "--regimes", "sg_satisficing",
             "--num-reads", "50", "--backends", "qpu:zephyr:clique"])
    ms.main(["coupling-scan", "--out", str(out), "--Ns", "8", "24", "--prefactors", "0.5", "1.414",
             "--num-reads", "50", "--budget-seconds", "2"])
    cfg = ms.load_config(out)
    t = cfg.chain_strength_by_N["qpu_zephyr_clique"]
    assert set(t) == {"8", "24"} and (out / "coupling_scan" / "coupling_scan.png").exists()
    ms.main(["qpu", "--out", str(out), "--Ns", "16", "--budget-seconds", "1"])        # only N = 16
    files = list((out / "anneal" / "qpu_zephyr_clique").rglob("*.json"))
    assert len(files) == 1 and "N0016" in files[0].name
    used = json.loads(files[0].read_text())["info"]["settings"]["chain_strength_prefactor"]
    assert used == pytest.approx((t["8"] + t["24"]) / 2)                             # interpolated at N = 16
    tally = ms.validate(out, cfg, log=lambda *a: None)
    assert all(bad == 0 for ok, bad in tally.values())


def test_extended_j_range_and_shimming_on_mock(tmp_path, monkeypatch):
    """Extended J range halves the compression; shimming learns offsets and the main run uses them."""
    import dwave.system
    from dwave.system.testing import MockDWaveSampler
    monkeypatch.setattr(dwave.system, "DWaveSampler",
                        lambda solver=None, **kw: MockDWaveSampler(topology_type="zephyr", topology_shape=[3, 4]))
    out = tmp_path / "s"
    ms.main(["init", "--out", str(out), "--Ns", "16", "--n-instances", "1", "--regimes", "sg_satisficing",
             "--num-reads", "50", "--backends", "qpu:zephyr:clique", "--extended-j-range",
             "--chain-strength-prefactor", "3"])
    ms.main(["shim", "--out", str(out), "--backend", "qpu:zephyr:clique", "--N", "16", "--iterations", "3",
             "--num-reads", "40", "--budget-seconds", "1"])
    from dwave_utils import shimming
    fb = shimming.load_fbos(out, "qpu_zephyr_clique")
    hist = json.loads((out / "shims" / "qpu_zephyr_clique" / "history.json").read_text())
    assert len(hist) == 3 and fb and any(abs(v) > 0 for v in fb.values())
    # offsets follow the rule Phi <- Phi - alpha m, starting from 0, alpha = 1e-5: |Phi| <= 3e-5
    assert max(abs(v) for v in fb.values()) <= 3e-5 + 1e-12
    cfg = ms.load_config(out)
    assert "qpu_zephyr_clique" in cfg.flux_biases_files
    ms.main(["qpu", "--out", str(out), "--budget-seconds", "1"])
    r = json.loads(next((out / "anneal" / "qpu_zephyr_clique").rglob("N0016_i0.json")).read_text())
    info = r["info"]
    assert info["flux_biases"]["n_nonzero"] == sum(1 for v in fb.values() if v != 0)
    sc = info["programmed"]["scaling"]
    assert sc["mode"] == "extended_j_range" and info["sample_kwargs"]["auto_scale"] is False
    # chains (c ~ 3) use the extended range: divisor = c / 2, not c
    c, wmax = info["chain_strength"], info["max_abs_logical_coupling"]
    assert c / 2 - 1e-9 <= sc["divisor"] <= max(c / 2, wmax) + 1e-9
    tally = ms.validate(out, cfg, log=lambda *a: None)
    assert all(bad == 0 for ok, bad in tally.values())


def test_linear_pipeline_on_mock(tmp_path, monkeypatch):
    """budget -> calibrate (every N, adaptive) -> run (exact p_opt per N, machine) -> report."""
    import csv as _csv
    import dwave.system
    from dwave.system.testing import MockDWaveSampler
    monkeypatch.setattr(dwave.system, "DWaveSampler",
                        lambda solver=None, **kw: MockDWaveSampler(topology_type="zephyr", topology_shape=[3, 4]))
    out = tmp_path / "p"
    ms.main(["init", "--out", str(out), "--Ns", "8,12", "--n-instances", "1", "--regimes", "sg_satisficing",
             "--num-reads", "30", "--backends", "qpu:zephyr:clique"])
    ms.main(["budget", "--out", str(out)])
    with pytest.raises(SystemExit):                      # no calibration yet: run refuses
        ms.main(["run", "--out", str(out)])
    ms.main(["calibrate", "--out", str(out), "--budget-seconds", "20"])
    cfg = ms.load_config(out)
    t = cfg.chain_strength_by_N["qpu_zephyr_clique"]
    assert set(t) == {"8", "12"}
    rows = list(_csv.DictReader(open(out / "calibration" / "p_opt.csv")))
    assert {int(r["N"]) for r in rows} == {8, 12}
    # adaptive: no p above the chosen one was tested
    for N, p in t.items():
        tested = [float(f.name.split("_p")[1].split("_i")[0])
                  for f in (out / "coupling_scan" / "qpu_zephyr_clique").rglob(f"N{int(N):04d}_p*_i0.json")]
        assert max(tested) == p
    ms.main(["run", "--out", str(out), "--budget-seconds", "5"])
    res = list(_csv.DictReader(open(out / "results" / "sk_qpu_results.csv")))
    assert len(res) == 2
    for r in res:
        assert float(r["p_used"]) == float(t[r["N"]])
        assert r["reference_source"] == "exact" and r["reached_reference"] in ("True", "False")
        assert len(r["best_configuration"]) == int(r["N"])


def test_pipeline_all_classes_and_fill(tmp_path, monkeypatch):
    """One p_opt per (machine, N) shared by every class; budget --fill sizes the reads to use what is left."""
    import csv as _csv
    import dwave.system
    from dwave_utils import pipeline
    from dwave.system.testing import MockDWaveSampler
    monkeypatch.setattr(dwave.system, "DWaveSampler",
                        lambda solver=None, **kw: MockDWaveSampler(topology_type="zephyr", topology_shape=[3, 4]))
    out = tmp_path / "f"
    ms.main(["init", "--out", str(out), "--Ns", "8,12", "--n-instances", "2", "--regimes", "ferro_easy", "sg_satisficing",
             "--num-reads", "20", "--backends", "qpu:zephyr:clique"])
    ms.main(["calibrate", "--out", str(out)])
    cfg = ms.load_config(out)
    assert cfg.chain_strength_by_class == {} and cfg.calibration_class == "eps0_J00"
    assert set(cfg.chain_strength_by_N["qpu_zephyr_clique"]) == {"8", "12"}
    assert not (out / "coupling_scan" / "qpu_zephyr_clique" / "eps0_J03").exists()   # measured on one class only
    rows = list(_csv.DictReader(open(out / "calibration" / "p_opt.csv")))
    assert len(rows) == 2
    assert pipeline.missing_calibration(out, cfg) == []
    spent_cal = pipeline.spent_in_campaign(out)["calibrate"]
    total = spent_cal + 4.0
    res = pipeline.budget(out, cfg, total=total, fill=True, margin=0.5, log=lambda *a: None)
    assert res["calibrate_left_s"] == 0
    cfg = ms.load_config(out)
    assert cfg.num_reads == res["num_reads"] > 20
    ms.main(["run", "--out", str(out), "--total", str(total)])
    res_rows = list(_csv.DictReader(open(out / "results" / "sk_qpu_results.csv")))
    assert len(res_rows) == 2 * 2 * 2                          # classes x N x instances
    for r in res_rows:                                         # the same p for both classes at each N
        assert float(r["p_used"]) == float(cfg.chain_strength_by_N["qpu_zephyr_clique"][r["N"]])
    assert pipeline.spent_in_campaign(out)["run"] <= total - spent_cal


def test_compare_step_on_mock(tmp_path, monkeypatch):
    """QPU (mock) vs classical vs game, easy -> hard, with exact and best-known references."""
    import csv as _csv
    import dwave.system
    from dwave.system.testing import MockDWaveSampler
    monkeypatch.setattr(dwave.system, "DWaveSampler",
                        lambda solver=None, **kw: MockDWaveSampler(topology_type="zephyr", topology_shape=[3, 4]))
    out = tmp_path / "c"
    ms.main(["init", "--out", str(out), "--Ns", "8,12", "--n-instances", "2", "--regimes", "ferro_easy", "sg_satisficing",
             "--num-reads", "30", "--backends", "qpu:zephyr:clique"])
    cfg = ms.load_config(out)
    cfg.n_exact = 8                                    # N = 12 gets a best-known reference
    ms.save_config(out, cfg)
    ms.main(["classical", "--out", str(out)])
    ms.main(["calibrate", "--out", str(out)])
    ms.main(["run", "--out", str(out)])
    ms.main(["compare", "--out", str(out)])
    rows = list(_csv.DictReader(open(out / "compare" / "compare.csv")))
    assert [r["problem"] for r in rows][0] == "eps0_J03"          # easy first
    kinds = {r["kind"] for r in rows}
    assert {"qpu", "classical", "game"} <= kinds
    assert {r["reference"] for r in rows if r["N"] == "8"} == {"exact"}
    assert {r["reference"] for r in rows if r["N"] == "12"} == {"best_known"}
    for r in rows:
        assert 0.0 <= float(r["P_rho_le_0"]) <= float(r["P_rho_le_0.01"]) <= float(r["P_rho_le_0.05"]) <= 1.0
        assert float(r["rho_best_median"]) >= -1e-9          # nothing beats the reference
    assert (out / "compare" / "compare.png").exists() and (out / "compare" / "reference_quality.png").exists()


def test_feasibility_model():
    """Verdicts follow the formulas: eps = sigma sqrt|E| p sqrt(N-1) / r, T/T_c = p / (r beta T_c)."""
    from dwave_utils import feasibility as fz
    rows = fz.run("zephyr", [16, 200], sigma=0.02, beta=5.0, n=3, log=lambda *a: None)
    r16, r200 = rows
    assert r200["verdict"] == "NO" and "largest clique" in r200["reason"]
    p, N = r16["p_low"], 16
    assert r16["eps"] == pytest.approx(0.02 * np.sqrt(r16["E_mean"]) / r16["programmed_rms"], rel=1e-6)
    assert r16["T_ratio"] == pytest.approx(p / 5.0, rel=0.05)            # D = c = p u when c > max|W|
    good = fz.run("zephyr", [16], sigma=1e-4, beta=1e3, n=3, log=lambda *a: None)[0]
    bad = fz.run("zephyr", [16], sigma=0.5, beta=5.0, n=3, log=lambda *a: None)[0]
    assert good["verdict"] == "YES" and bad["verdict"] == "NO"
    ext = fz.run("zephyr", [16], sigma=0.02, beta=5.0, extended=True, n=3, log=lambda *a: None)[0]
    assert ext["eps"] == pytest.approx(r16["eps"] / 2, rel=0.05)         # extended range halves the error


def test_dwave_utils_api_and_cli(tmp_path, monkeypatch, capsys):
    """The package exposes the functions and `python -m dwave_utils` forwards to main_simulation."""
    import dwave_utils as du
    from dwave_utils.__main__ import main as du_main
    for name in du.__all__:
        assert hasattr(du, name)
    rows = du.check_feasibility("zephyr", [16], n=2, log=lambda *a: None)
    assert du.feasibility.run is du.check_feasibility
    assert rows[0]["verdict"] in ("YES", "MARGINAL", "NO")
    du_main(["feasibility", "--topology", "zephyr", "--N", "16", "--instances", "2"])
    assert "verdict" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        du_main(["no-such-command"])
    out = tmp_path / "api"
    ms.main(["init", "--out", str(out), "--Ns", "8", "--n-instances", "1", "--regimes", "sg_satisficing"])
    o, cfg = du.open_campaign(out)
    assert o == out and cfg.Ns == [8]
    assert du.left(out, 10.0) == pytest.approx(10.0)


def test_dwave_utils_python_sequence_keeps_config(tmp_path, monkeypatch):
    """Calling the steps from Python with ONE cfg object: no step overwrites what another stored."""
    import dwave.system
    import dwave_utils as du
    from dwave.system.testing import MockDWaveSampler
    monkeypatch.setattr(dwave.system, "DWaveSampler",
                        lambda solver=None, **kw: MockDWaveSampler(topology_type="zephyr", topology_shape=[3, 4]))
    out = tmp_path / "py"
    ms.main(["init", "--out", str(out), "--Ns", "8", "--n-instances", "1", "--regimes", "sg_satisficing",
             "--num-reads", "20", "--backends", "qpu:zephyr:clique"])
    out, cfg = du.open_campaign(out)
    du.calibrate(out, cfg, log=lambda *a: None)
    assert cfg.chain_strength_by_N                                   # mirrored into the caller's cfg
    total = du.spent_in_campaign(out)["calibrate"] + 2.0
    du.budget(out, cfg, total=total, fill=True, margin=0.2, log=lambda *a: None)
    disk = ms.load_config(out)
    assert disk.chain_strength_by_N == cfg.chain_strength_by_N and disk.num_reads == cfg.num_reads > 20
    du.shim(out, cfg, "qpu:zephyr:clique", 8, iterations=2, num_reads=20, log=lambda *a: None)
    disk = ms.load_config(out)
    assert disk.flux_biases_files and disk.chain_strength_by_N and disk.num_reads == cfg.num_reads
    du.run(out, cfg, budget_s=du.left(out, total), log=lambda *a: None)
    assert du.report(out, cfg, log=lambda *a: None)
