"""The Python engine against fixtures/golden.json.

web/test/golden.test.js checks the JavaScript engine against the same file,
so together the two tests say the engines agree to the last bit. If this one
fails after a deliberate change to the simulation, regenerate the fixture with
``python scripts/make_golden.py`` and commit it with the change.
"""

import importlib.util
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent


def _make_golden() -> dict[str, Any]:
    script = ROOT / "scripts" / "make_golden.py"
    spec = importlib.util.spec_from_file_location("make_golden", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result: dict[str, Any] = module.golden()
    return result


def test_engine_reproduces_the_golden_fixture_exactly() -> None:
    expected = json.loads((ROOT / "fixtures" / "golden.json").read_text())
    # Round-trip through JSON so tuples compare equal to lists; floats survive
    # exactly because json writes them with repr.
    actual = json.loads(json.dumps(_make_golden()))
    for section in ("rng", "brain", "run"):
        assert actual[section] == expected[section], f"{section} section differs"
