"""Planner-optimisation benchmark outputs (the 'when does each optimiser break' study)."""
from __future__ import annotations

import json

import numpy as np

from ..registry import Output
from ..optim.benchmark import BenchmarkConfig, run_benchmark
from ..optim.solvers import default_solvers


class PlannerBenchmark(Output):
    kind = "benchmark"
    paper_ref = "Eq. 13, footnote [58]-[59]; Sec. IX A (omniscient planner)"
    description = "Ground-state search for the planner problem with many optimisers vs N: success, residual, TTS."
    presets = {"quick": {"Ns": [8, 12, 16, 20, 24, 32, 48, 64, 96, 128], "eps": 0.0, "n_instances": 3,
                         "n_runs": 3, "n_exact": 24, "budget": "quick", "ref_sweeps": 2000, "ref_restarts": 2},
               "paper": {"Ns": [8, 12, 16, 20, 24, 28, 32, 48, 64, 96, 128, 192, 256], "eps": 0.0, "n_instances": 20,
                         "n_runs": 10, "n_exact": 30, "budget": "paper", "ref_sweeps": 50000, "ref_restarts": 4}}

    def compute(self, deps):
        p = self.params
        cfg = BenchmarkConfig(Ns=tuple(p["Ns"]), eps=p["eps"], n_instances=p["n_instances"], n_runs=p["n_runs"],
                              n_exact=p["n_exact"], seed=self.ctx.seed,
                              ref_sweeps=p["ref_sweeps"], ref_restarts=p["ref_restarts"])
        solvers = default_solvers(p["budget"])
        res = run_benchmark(solvers, cfg, log=self.ctx.log)
        m = res.metrics()
        (self.folder() / "solvers.json").write_text(json.dumps(
            [{"name": s.name, **{k: str(v) for k, v in s.describe().items()}} for s in solvers], indent=2))
        return {"solvers": np.array(res.solvers), "Ns": res.Ns, "reward": res.reward, "seconds": res.seconds,
                "evals": res.evals, "ref": res.ref, "ref_exact": res.ref_exact, "upper_bound": res.upper_bound,
                "p_median": m["p_median"], "residual_median": m["residual_median"], "t_run": m["t_run"],
                "tts99": m["tts99"], "N_break": m["N_break"], "certification_gap": m["certification_gap"],
                "ref_source": res.extra["ref_source"], "eps": p["eps"]}

    def save(self, result, folder):
        r = dict(result)
        r["solvers"] = np.array([str(x) for x in r["solvers"]])
        np.savez_compressed(folder / "result.npz", **{k: v for k, v in r.items() if isinstance(v, np.ndarray)})
        (folder / "scalars.json").write_text(json.dumps({"eps": r["eps"]}))
