"""Command line:  python -m skgame [run|list|category|deps] ...

  python -m skgame list
  python -m skgame run --preset quick                 # everything
  python -m skgame run Fig03Autocorrelation Fig11C2vsEps --preset paper --out output
  python -m skgame category                           # export category.json / .dot / Catlab file
  python -m skgame deps Fig17OrderParameter           # upstream / downstream classes
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from . import outputs  # noqa: F401  (registers classes)
from .registry import REGISTRY, Pipeline, RunContext, export_category, topological_order, upstream, downstream, derivation_expr


def main(argv=None):
    ap = argparse.ArgumentParser(prog="skgame")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("targets", nargs="*")
    r.add_argument("--preset", default="quick", choices=["quick", "paper"])
    r.add_argument("--out", default="output")
    r.add_argument("--force", action="store_true")
    r.add_argument("--seed", type=int, default=12345)
    r.add_argument("--kind", default=None, help="only classes of this kind (theory/data/figure/table/benchmark)")
    sub.add_parser("list")
    c = sub.add_parser("category")
    c.add_argument("--out", default="category")
    d = sub.add_parser("deps")
    d.add_argument("name")
    a = ap.parse_args(argv)

    if a.cmd == "list":
        for n in topological_order():
            cls = REGISTRY[n]
            print(f"{cls.kind:9s} {n:30s} {cls.paper_ref:28s} <- {', '.join(x.__name__ for x in cls.depends_on) or 'I'}")
    elif a.cmd == "category":
        for k, p in export_category(Path(a.out)).items():
            print(f"{k:8s} {p}")
    elif a.cmd == "deps":
        print("upstream  :", sorted(upstream(a.name)))
        print("downstream:", sorted(downstream(a.name)))
        print("derivation:", derivation_expr(a.name))
    elif a.cmd == "run":
        targets = a.targets or None
        if a.kind:
            targets = [n for n in (targets or REGISTRY) if REGISTRY[n].kind == a.kind]
        ctx = RunContext(out_dir=Path(a.out), preset=a.preset, force=a.force, seed=a.seed)
        t0 = time.time()
        pl = Pipeline(ctx)
        pl.run(targets)
        Path(a.out).mkdir(exist_ok=True)
        log = Path(a.out) / f"timings_{a.preset}.json"
        old = json.loads(log.read_text()) if log.exists() else {}
        old.update(pl.timings)
        log.write_text(json.dumps(old, indent=2))
        if pl.failed:
            print("FAILED:", ", ".join(pl.failed))
        print(f"done in {time.time() - t0:.1f}s")
        if pl.failed:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
