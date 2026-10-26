"""Output classes, their dependency category, caching and the pipeline runner.

Category-theoretic reading
--------------------------
* Objects   : output classes (subclasses of `Output`), plus the monoidal unit I ("no input").
* Generators: one morphism per class,  make_C : D_1 (x) ... (x) D_k -> C,  where
              (D_1,...,D_k) = C.depends_on  (I if empty).  Parameters are not objects: a
              morphism is parametrised by C.presets[preset] (a "Para" construction).
* Composites: the unique derivation of C from I is the operadic composite
              make_C o (deriv(D_1) (x) ... (x) deriv(D_k))  (+ copies of shared inputs via
              the diagonal of the cartesian structure). `derivation_expr` returns it.
The free symmetric monoidal category on these generators is exported to JSON and consumed
by the Julia/Catlab module in julia/SKGameCategory.
"""
from __future__ import annotations

import hashlib
import json
import platform
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

REGISTRY: dict[str, type["Output"]] = {}
KINDS = ("theory", "data", "figure", "table", "benchmark")


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, float)):
        xf = float(x)
        return "inf" if np.isinf(xf) else xf
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    return x


def source_hash(cls) -> str:
    """Hash of the class source: editing compute() invalidates the cache of C and its downstream."""
    import inspect
    try:
        return hashlib.sha1(inspect.getsource(cls).encode()).hexdigest()[:12]
    except (OSError, TypeError):
        return "nosrc"


def stable_hash(obj) -> str:
    return hashlib.sha1(json.dumps(_jsonable(obj), sort_keys=True).encode()).hexdigest()[:12]


@dataclass
class RunContext:
    out_dir: Path
    preset: str = "quick"
    force: bool = False
    seed: int = 12345
    verbose: bool = True
    keep_going: bool = True
    cache: dict = field(default_factory=dict)  # class name -> (hash, result)

    def log(self, msg: str):
        if self.verbose:
            print(msg, flush=True)


class Output(ABC):
    """An output class = an object of the dependency category."""

    kind: ClassVar[str] = "data"
    depends_on: ClassVar[tuple] = ()
    paper_ref: ClassVar[str] = ""
    description: ClassVar[str] = ""
    presets: ClassVar[dict] = {"quick": {}, "paper": {}}

    def __init_subclass__(cls, abstract: bool = False, **kw):
        super().__init_subclass__(**kw)
        if abstract:
            return
        if cls.kind not in KINDS:
            raise ValueError(f"{cls.__name__}: unknown kind {cls.kind}")
        if cls.__name__ in REGISTRY:
            raise ValueError(f"duplicate output class {cls.__name__}")
        REGISTRY[cls.__name__] = cls

    # -------------------------------------------------------------------------------
    def __init__(self, ctx: RunContext):
        self.ctx = ctx
        base = dict(self.presets.get("quick", {}))
        base.update(self.presets.get(ctx.preset, {}))
        self.params = base

    @classmethod
    def name(cls) -> str:
        return cls.__name__

    def folder(self) -> Path:
        return self.ctx.out_dir / self.kind / self.name()

    def rng(self, salt: int = 0) -> np.random.Generator:
        return np.random.default_rng([self.ctx.seed, int(stable_hash(self.name())[:8], 16), salt])

    # -------------------------------------------------------------------------------
    @abstractmethod
    def compute(self, deps: dict[str, Any]) -> Any:
        """Morphism make_C : (x)_k deps -> C. Must return a dict (data) or a list of files."""

    def save(self, result: dict, folder: Path) -> None:
        arrays = {k: np.asarray(v) for k, v in result.items() if isinstance(v, np.ndarray)}
        scalars = {k: v for k, v in result.items() if not isinstance(v, np.ndarray)}
        np.savez_compressed(folder / "result.npz", **arrays)
        (folder / "scalars.json").write_text(json.dumps(_jsonable(scalars), indent=2))

    def load(self, folder: Path) -> dict:
        out = {}
        if (folder / "result.npz").exists():
            with np.load(folder / "result.npz", allow_pickle=False) as z:
                out.update({k: z[k] for k in z.files})
        if (folder / "scalars.json").exists():
            out.update(json.loads((folder / "scalars.json").read_text()))
        return out


class FigureOutput(Output, abstract=True):
    kind: ClassVar[str] = "figure"

    def save(self, result, folder):  # figures are written inside compute
        pass

    def load(self, folder):
        return {"files": sorted(str(p) for p in folder.glob("*.png"))}


# ----------------------------------------------------------------------------- graph --
def topological_order(targets=None) -> list[str]:
    targets = list(targets or REGISTRY)
    order, state = [], {}

    def visit(n, stack):
        if state.get(n) == 2:
            return
        if state.get(n) == 1:
            raise ValueError("dependency cycle: " + " -> ".join(stack + [n]))
        state[n] = 1
        for d in REGISTRY[n].depends_on:
            visit(d.__name__, stack + [n])
        state[n] = 2
        order.append(n)

    for t in targets:
        visit(t, [])
    return order


def upstream(name: str) -> set[str]:
    return set(topological_order([name])) - {name}


def downstream(name: str) -> set[str]:
    return {n for n in REGISTRY if name in upstream(n)}


def derivation_expr(name: str) -> str:
    """Composite morphism I -> C as an expression in the free SMC (copies written explicitly)."""
    deps = REGISTRY[name].depends_on
    gen = f"make_{name}"
    if not deps:
        return gen
    inner = " ⊗ ".join(f"({derivation_expr(d.__name__)})" for d in deps)
    return f"{gen} ∘ [{inner}]"


def category_dict() -> dict:
    objs = []
    for n in topological_order():
        c = REGISTRY[n]
        objs.append({"name": n, "kind": c.kind, "paper_ref": c.paper_ref,
                     "module": c.__module__, "description": c.description})
    homs = [{"name": f"make_{n}", "dom": [d.__name__ for d in REGISTRY[n].depends_on],
             "codom": n} for n in topological_order()]
    from .mathcat import math_category_dict
    return {"name": "SKGamePaper", "monoidal_unit": "I", "objects": objs, "generators": homs,
            "math": math_category_dict()}


def export_category(folder: Path) -> dict[str, Path]:
    folder.mkdir(parents=True, exist_ok=True)
    d = category_dict()
    paths = {}
    paths["json"] = folder / "category.json"
    paths["json"].write_text(json.dumps(d, indent=2, ensure_ascii=False))
    # Graphviz
    colors = {"theory": "#cfe2ff", "data": "#d1e7dd", "figure": "#fff3cd", "table": "#e2e3e5",
              "benchmark": "#f8d7da"}
    lines = ["digraph SKGamePaper {", "  rankdir=LR; node [shape=box, style=filled, fontname=Helvetica];"]
    for o in d["objects"]:
        lab = o["name"] + (f"\\n{o['paper_ref']}" if o["paper_ref"] else "")
        lines.append(f'  {o["name"]} [label="{lab}", fillcolor="{colors[o["kind"]]}"];')
    for h in d["generators"]:
        for x in h["dom"]:
            lines.append(f"  {x} -> {h['codom']};")
    lines.append("}")
    paths["dot"] = folder / "category.dot"
    paths["dot"].write_text("\n".join(lines))
    # Mermaid
    mm = ["flowchart LR"]
    for h in d["generators"]:
        for x in h["dom"]:
            mm.append(f"  {x} --> {h['codom']}")
    paths["mermaid"] = folder / "category.mmd"
    paths["mermaid"].write_text("\n".join(mm))
    # Literal Catlab presentation (for inspection; the module builds it from JSON as well)
    jl = ["# AUTO-GENERATED by skgame.registry.export_category — do not edit by hand",
          "using Catlab", "", "@present SKGamePaperLiteral(FreeSymmetricMonoidalCategory) begin"]
    jl.append("  (" + ", ".join(o["name"] for o in d["objects"]) + ")::Ob")
    for h in d["generators"]:
        dom = "munit()" if not h["dom"] else " ⊗ ".join(h["dom"])
        jl.append(f"  {h['name']}::Hom({dom}, {h['codom']})")
    jl.append("end")
    m = d["math"]
    jl += ["", "@present SKGameMathLiteral(FreeSymmetricMonoidalCategory) begin",
           "  (" + ", ".join(o["name"] for o in m["objects"]) + ")::Ob"]
    for h in m["generators"]:
        jl.append(f"  {h['name']}::Hom({' ⊗ '.join(h['dom'])}, {h['codom']})")
    jl.append("end")
    paths["catlab"] = folder / "SKGamePaperLiteral.jl"
    paths["catlab"].write_text("\n".join(jl) + "\n")
    return paths


# ---------------------------------------------------------------------------- runner --
class Pipeline:
    def __init__(self, ctx: RunContext):
        self.ctx = ctx

    def _meta_path(self, obj: Output) -> Path:
        return obj.folder() / "meta.json"

    def run(self, targets=None) -> dict[str, Any]:
        order = topological_order(targets)
        hashes: dict[str, str] = {}
        results: dict[str, Any] = {}
        timings = {}
        self.failed: dict[str, str] = {}
        self.timings = timings
        for n in order:
            cls = REGISTRY[n]
            obj = cls(self.ctx)
            dep_hashes = {d.__name__: hashes[d.__name__] for d in cls.depends_on}
            h = stable_hash({"cls": n, "params": obj.params, "deps": dep_hashes, "seed": self.ctx.seed,
                             "code": source_hash(cls)})
            hashes[n] = h
            folder = obj.folder()
            meta_p = self._meta_path(obj)
            cached = False
            if not self.ctx.force and meta_p.exists():
                meta = json.loads(meta_p.read_text())
                if meta.get("hash") == h and meta.get("status") == "ok":
                    results[n] = obj.load(folder)
                    cached = True
            if not cached:
                folder.mkdir(parents=True, exist_ok=True)
                missing = [d.__name__ for d in cls.depends_on if d.__name__ in self.failed]
                if missing:
                    self.ctx.log(f"[skipped  ] {n} (failed inputs: {missing})")
                    self.failed[n] = f"inputs failed: {missing}"
                    continue
                deps = {d.__name__: results[d.__name__] for d in cls.depends_on}
                self.ctx.log(f"[{cls.kind:9s}] {n:32s} {cls.paper_ref}")
                t0 = time.time()
                try:
                    res = obj.compute(deps)
                except Exception as exc:
                    if not self.ctx.keep_going:
                        raise
                    import traceback
                    self.failed[n] = traceback.format_exc()
                    self.ctx.log(f"[FAILED   ] {n}: {exc!r}")
                    meta_p.write_text(json.dumps({"class": n, "hash": h, "status": "failed",
                                                  "error": self.failed[n]}, indent=2))
                    continue
                dt = time.time() - t0
                timings[n] = dt
                if isinstance(res, dict):
                    obj.save(res, folder)
                results[n] = obj.load(folder) if cls.kind == "figure" else res
                meta = {"class": n, "kind": cls.kind, "paper_ref": cls.paper_ref, "hash": h,
                        "params": _jsonable(obj.params), "deps": dep_hashes, "preset": self.ctx.preset,
                        "seed": self.ctx.seed, "seconds": dt, "status": "ok",
                        "python": platform.python_version(), "time": time.strftime("%Y-%m-%d %H:%M:%S")}
                meta_p.write_text(json.dumps(meta, indent=2))
            else:
                self.ctx.log(f"[cached   ] {n}")
        self.timings = timings
        return results
