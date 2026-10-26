"""Integration check of a full pipeline run (python -m skgame run --preset quick --out output).
Skipped when the output folder is absent."""
import json
from pathlib import Path

import pytest

import skgame.outputs  # noqa: F401
from skgame.registry import REGISTRY

OUT = Path(__file__).resolve().parents[1] / "output"
pytestmark = pytest.mark.skipif(not OUT.exists(), reason="run the pipeline first")


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_output_class_folder(name):
    cls = REGISTRY[name]
    folder = OUT / cls.kind / name
    meta = json.loads((folder / "meta.json").read_text())
    assert meta["status"] == "ok", meta.get("error")
    assert meta["class"] == name and meta["kind"] == cls.kind
    if cls.kind == "figure":
        assert list(folder.glob("*.png")) and list(folder.glob("*.pdf"))
    elif cls.kind == "table":
        assert list(folder.glob("*.md"))
    else:
        assert (folder / "result.npz").exists()
