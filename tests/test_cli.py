import csv
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from neuronet.cli import main

FAST = ["--set", "population=8", "--set", "steps=60", "--set", "food_count=12"]


def test_train_writes_history_and_a_resumable_snapshot(tmp_path: Path) -> None:
    out = tmp_path / "run"
    assert main(["train", "--generations", "3", "--out", str(out), "--quiet", *FAST]) == 0
    rows = list(csv.DictReader((out / "history.csv").open()))
    assert [int(r["generation"]) for r in rows] == [0, 1, 2]

    snap = json.loads((out / "snapshot.json").read_text())
    assert snap["generation"] == 3
    assert snap["config"]["population"] == 8
    assert snap["expect"]["generation"] == 3

    more = tmp_path / "more"
    assert main(["train", "--resume", str(out / "snapshot.json"), "--generations", "1",
                 "--out", str(more), "--quiet"]) == 0  # fmt: skip
    resumed = list(csv.DictReader((more / "history.csv").open()))
    assert int(resumed[-1]["best"]) == snap["expect"]["best"]


def test_experiment_writes_results_and_charts(tmp_path: Path) -> None:
    args = ["experiment", "--seeds", "2", "--generations", "2", "--workers", "1"]
    assert main([*args, "--out", str(tmp_path), *FAST]) == 0
    result = json.loads((tmp_path / "experiment.json").read_text())
    assert set(result["runs"]) == {"tournament", "random"}
    assert set(result["runs"]["random"]) == {"1", "2"}
    for theme in ("light", "dark"):
        root = ET.parse(tmp_path / f"fitness-{theme}.svg").getroot()
        assert root.tag.endswith("svg")


def test_bad_config_override_is_reported(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit, match="bad config"):
        main(["train", "--set", "population=0", "--generations", "1"])


def test_seed_must_fit_in_32_bits() -> None:
    with pytest.raises(SystemExit):
        main(["train", "--seed", str(2**32)])
