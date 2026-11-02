"""D-Wave utilities of the SK-game campaign: calibration, feasibility, the QPU pipeline, shimming.

Modules
    (the submodules stay importable: from dwave_utils import pipeline, coupling_scan, feasibility, shimming)
    pipeline       budget -> calibrate -> run -> report -> compare (the linear QPU pipeline)
    coupling_scan  chain-strength prefactor scan and choice (used by calibrate)
    feasibility    yes / marginal / no: can a quantum annealer represent a problem? (no QPU)
    shimming       flux-bias shimming of a QPU on the SK problem

Python (from the folder unlearnable_games_explanation, e.g. in the notebook):

    import dwave_utils as du
    du.check_feasibility("zephyr", [16, 64, 104])                  # no campaign, no QPU
    out, cfg = du.open_campaign("results/sk")                # a campaign made with `init`
    du.budget(out, cfg, total=178)
    du.calibrate(out, cfg, budget_s=15)
    du.budget(out, cfg, total=178, fill=True)
    du.run(out, cfg, budget_s=du.left(out, total=178))
    rows = du.report(out, cfg)
    rows = du.compare(out, cfg)
    du.shim(out, cfg, "qpu:zephyr:clique", N=104, iterations=30)

Command line (same arguments as the main_simulation.py subcommands):

    python -m dwave_utils feasibility --topology zephyr --N 16 64 104
    python -m dwave_utils calibrate --out results/sk --budget-seconds 15
"""
from __future__ import annotations

from pathlib import Path

from .coupling_scan import (DEFAULT_PREFACTORS, choose_coupling, coupling_for, find_best_coupling_per_problem,
                            prefactor_for, summarize_coupling_scan)
from .feasibility import assess, embedding_stats, field_bound
from .feasibility import run as check_feasibility
from .pipeline import (GRID, budget, calibrate, calibration_class, compare, missing_calibration, report, run,
                       spent_in_campaign)
from .shimming import load_fbos, run_shimming

shim = run_shimming

COMMANDS = ("feasibility", "budget", "calibrate", "run", "report", "compare", "shim", "coupling-scan")


def open_campaign(out) -> tuple:
    """(Path, CampaignConfig) of a campaign directory created with `main_simulation.py init`."""
    import main_simulation as ms
    out = Path(out)
    return out, ms.load_config(out)


def left(out, total: float) -> float:
    """QPU seconds still available: `total` (what you had when the campaign started) minus what it spent."""
    return float(total) - sum(spent_in_campaign(Path(out)).values())


__all__ = ["COMMANDS", "DEFAULT_PREFACTORS", "GRID", "assess", "budget", "calibrate", "calibration_class",
           "choose_coupling", "compare", "coupling_for", "check_feasibility", "embedding_stats", "field_bound",
           "find_best_coupling_per_problem", "left", "load_fbos", "missing_calibration", "open_campaign",
           "prefactor_for", "report", "run", "run_shimming", "shim", "spent_in_campaign",
           "summarize_coupling_scan"]
