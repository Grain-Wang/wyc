"""CPU tests of the authorized admission-timing change; no scientific forwards."""

import copy

import pytest

from paper5.experiments.canary.direction_02 import stage0_analysis as analysis
from paper5.experiments.canary.direction_02 import stage0_data as data
from paper5.experiments.canary.direction_02 import stage0_quality_v2 as v2


def test_bad_initial_and_pilot_quality_do_not_veto_capacity_backed_v2() -> None:
    pilot = data.read_json(v2.SOURCE / "pilot.json")
    capacity = data.read_json(v2.CAPACITY / "summary.json")
    assert not pilot["quality_proposal"]["formal_entry_quality_passed"]
    assert not pilot["quality_proposal"]["initial_eligible"]
    v2.admission_evidence(pilot, capacity)
    broken = copy.deepcopy(pilot)
    broken["engineering_passed"] = False
    with pytest.raises(RuntimeError, match="Engineering"):
        v2.admission_evidence(broken, capacity)
    capacity["metrics"][-1]["ppl_ratio"] = 1.16
    with pytest.raises(RuntimeError, match="capacity"):
        v2.admission_evidence(pilot, capacity)


def test_final_quality_still_blocks_positive_selection_signal() -> None:
    cfg = data.read_json(v2.CONFIG)
    initial = {f"a{i:02d}": 3 + i / 100 for i in range(8)}
    terminal = {f"a{i:02d}": 3 + (7 - i) / 100 for i in range(8)}
    scores = {
        str(seed): {
            str(step): initial if step == 0 else terminal for step in cfg["checkpoints"]
        }
        for seed in cfg["training_seeds"]
    }
    choices = {
        seed: analysis.replay({int(k): v for k, v in points.items()})
        for seed, points in scores.items()
    }
    endpoints = {
        str(seed): {name: [loss] * 64 for name, loss in terminal.items()}
        for seed in cfg["training_seeds"]
    }
    endpoints["parent"] = [2.0] * 64
    costs = data.read_json(v2.SOURCE / "pilot.json")["unit_costs"]
    result = analysis.analyze(
        choices, scores, endpoints, [f"d{i}" for i in range(64)], [511] * 64, cfg, costs
    )
    assert all(row["G0"]["lower95"] > 0.02 for row in result["seeds"].values())
    assert result["decision"]["verdict"] == "QUALITY_INFEASIBLE"
    assert result["decision"]["problem"] == "NOT_ASSESSED_QUALITY"
    assert result["decision"]["simple_baseline"] == "NOT_ESTABLISHED"
    endpoints["parent"] = [3.0] * 64
    valid = analysis.analyze(
        choices, scores, endpoints, [f"d{i}" for i in range(64)], [511] * 64, cfg, costs
    )
    assert valid["decision"]["problem"] == "PROBLEM_SIGNAL"


def test_budget_includes_pilot_and_capacity_and_rejects_overrun() -> None:
    pilot = data.read_json(v2.SOURCE / "pilot.json")
    capacity = data.read_json(v2.CAPACITY / "summary.json")
    ledger = data.read_json(v2.CAPACITY / "gpu_budget.json")
    cfg = data.read_json(v2.CONFIG)
    projected = v2.budget_projection(pilot, capacity, ledger, cfg)
    assert projected["prior_gpu_seconds"] == pytest.approx(418.11011994164437)
    assert projected["counts"]["updates"] == 1600
    assert projected["projected_total_gpu_seconds"] < 14400
    ledger["spent_seconds"] = 14390
    with pytest.raises(RuntimeError, match="budget"):
        v2.budget_projection(pilot, capacity, ledger, cfg)


@pytest.mark.parametrize(
    "key,value",
    [
        ("learning_rate", 0.0002),
        ("updates", 20),
        ("training_seeds", [17]),
        ("proposed_quality_max_ppl_ratio", 1.2),
    ],
)
def test_scientific_changes_are_rejected(key: str, value: object) -> None:
    cfg = data.read_json(v2.CONFIG)
    v2.validate_recipe(cfg)
    cfg[key] = value
    with pytest.raises(RuntimeError, match="recipe"):
        v2.validate_recipe(cfg)
