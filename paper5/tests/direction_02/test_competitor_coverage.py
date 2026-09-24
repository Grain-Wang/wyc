"""Scientific scope, policy information firewall and historical regression checks."""

import copy
import json
from pathlib import Path

import pytest

from paper5.experiments.canary.direction_02 import run_competitor_coverage as coverage
from paper5.experiments.canary.direction_02 import run_stage0 as runner
from paper5.experiments.canary.direction_02 import stage0_analysis as analysis
from paper5.experiments.canary.direction_02 import stage0_data as data


def test_actual_historical_ranking_and_sampling() -> None:
    manifest = coverage.verify_coverage()
    assert manifest["competitive_pool"][:3] == [[12, 15], [10, 15], [10, 12]]
    assert manifest["new_candidates"] == coverage.NEW
    old = data.read_json(coverage.SOURCE / "candidate_manifest.json")
    assert manifest["candidates"][:8] == old["candidates"]
    assert len(manifest["candidates"]) == 10


def test_ten_candidate_replay_budget_and_queries() -> None:
    scores = {
        step: {f"a{i:02d}": float(i) for i in range(10)} for step in (0, 20, 50, 100)
    }
    scores[0]["a08"] = -1.0
    scores[100]["a09"] = -2.0
    choices = analysis.replay(scores, policy_sizes=(10, 5, 2))
    assert choices["B0"]["selected"] == "a08"
    assert choices["Ref100"]["selected"] == "a09"
    assert choices["BSH"]["selected"] == "a00"
    assert {k: v["training_updates"] for k, v in choices.items()} == {
        "B0": 100,
        "B20": 280,
        "BSH": 450,
        "Ref100": 1000,
    }
    assert [
        sum(q["checkpoint"] == step for q in choices["BSH"]["queries"])
        for step in (20, 50, 100)
    ] == [10, 5, 2]
    assert {
        q["candidate"] for q in choices["BSH"]["queries"] if q["checkpoint"] == 100
    } == {"a00", "a01"}
    altered = copy.deepcopy(scores)
    altered[50]["a09"] = -999.0
    altered[100]["a09"] = -999.0
    again = analysis.replay(altered, policy_sizes=(10, 5, 2))
    for policy in ("B0", "B20", "BSH"):
        assert again[policy] == choices[policy]


def test_original_eight_metrics_unchanged() -> None:
    source = coverage.SOURCE
    manifest = data.read_json(source / "data_manifest.json")
    windows = manifest["splits"]["E"]["windows"]
    actual = analysis.analyze(
        data.read_json(source / "selection.json")["choices"],
        data.read_json(source / "s_scores.json"),
        data.read_json(source / "e_losses.json"),
        [w["doc_id"] for w in windows],
        [w["labels"] for w in windows],
        data.read_json(source / "config_used.json"),
        data.read_json(source / "pilot.json")["unit_costs"],
    )
    archived = data.read_json(source / "metrics.json")
    assert actual == {k: archived[k] for k in actual}


def test_missing_trajectory_cannot_open_e(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(runner, "OUT", tmp_path)
    data.write_json(tmp_path / "selection.json", {})
    data.write_json(
        tmp_path / "trajectories.json",
        data.read_json(coverage.SOURCE / "trajectories.json"),
    )
    with pytest.raises(RuntimeError, match="Twenty complete"):
        coverage.open_e({})


def test_final_quality_failure_never_problem_signal() -> None:
    base = data.read_json(coverage.SOURCE / "metrics.json")["seeds"]
    values = copy.deepcopy(base)
    for row in values.values():
        row["quality_usable"] = False
        row["G0"] = {"point": 0.1, "lower95": 0.05, "upper95": 0.15}
    assert analysis.classify(values, 0.02)["verdict"] == "QUALITY_INFEASIBLE"


def test_projection_inherits_time_and_enforces_one_hour(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = data.read_json(coverage.CONFIG)
    monkeypatch.setattr(runner, "CFG", cfg)
    value = coverage.projection()
    assert value["prior_gpu_seconds"] == 2375.137250719592
    assert value["counts"]["updates"] == 400
    assert value["projected_additional_gpu_seconds"] < 3600
    cfg["budget_safety_factor"] = 10
    with pytest.raises(RuntimeError, match="exceed authorized budget"):
        coverage.projection()


def test_scientific_recipe_mutation_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = data.read_json(coverage.CONFIG)
    cfg["learning_rate"] = 0.002
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg))
    monkeypatch.setattr(coverage, "CONFIG", path)
    with pytest.raises(RuntimeError, match="Scientific recipe"):
        coverage.activate()


def test_incomplete_ten_scores_rejected() -> None:
    scores = {
        step: {f"a{i:02d}": float(i) for i in range(10)} for step in (0, 20, 50, 100)
    }
    del scores[100]["a09"]
    with pytest.raises(ValueError, match="Missing"):
        analysis.replay(scores, policy_sizes=(10, 5, 2))
