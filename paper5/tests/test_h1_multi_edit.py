"""CPU-only contracts for the once-authorized multi-edit diagnostic."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from paper5.experiments.canary.direction_01 import run_h1_multi_edit as multi


@pytest.fixture(scope="module")
def plan() -> dict:
    """Rebuild the exact candidates from immutable calibration labels."""
    cal, _ = multi.archive()
    return multi.candidate_plan(cal)


def test_candidate_manifest_and_scores(plan: dict) -> None:
    """Lock all 90 tuples, disjoint strata, budgets and frozen prediction ranks."""
    assert (
        multi.digest(multi.canonical(plan["candidate_manifest"])) == multi.MANIFEST_HASH
    )
    assert sum(len(p["architectures"]) for p in plan["panels"].values()) == 90
    for k in (4, 6, 8):
        arches = [
            a
            for s in ("global", "low")
            for a in plan["panels"][f"{k}_{s}"]["architectures"]
        ]
        assert len(set(arches)) == 30
        assert all(len(a) == k for a in arches)
    for p in plan["panels"].values():
        assert p["s1_order"] == multi.history.order(np.array(p["s1"]))
        assert p["s2_order"] == multi.history.order(np.array(p["s2"]))
        assert sorted(p["direct_prefix_order"]) == list(range(15))


def test_second_order_counts_every_pair_once() -> None:
    """Pair identity and explicit k=4 residual sum detect double counting."""
    effects = {(): 0, (1,): 1, (2,): 2, (3,): 3, (4,): 4}
    for i, j in multi.itertools.combinations(range(1, 5), 2):
        effects[(i, j)] = i + j + 0.25
    s1, s2 = multi.scores([(1, 2), (1, 2, 3, 4)], effects)
    np.testing.assert_allclose(s1, [3, 10])
    np.testing.assert_allclose(s2, [effects[(1, 2)], 11.5])


def test_holdout_excludes_old_and_neighbors_and_is_deterministic() -> None:
    """Exactly 64 complete chunks; no changing seed or filling from old windows."""
    old = list(range(0, 256, 4))
    first = multi.holdout_indices(528, old)
    assert first == multi.holdout_indices(528, old[::-1])
    assert len(set(first)) == 64
    assert not set(first) & {j + d for j in old for d in (-1, 0, 1)}
    with pytest.raises(ValueError, match="Fewer"):
        multi.holdout_indices(64, list(range(64)))
    with pytest.raises(ValueError, match="Invalid"):
        multi.holdout_indices(528, [0] * 64)


def test_policy_query_isolation(plan: dict) -> None:
    """Unqueried C labels cannot change policy choices; V_new is not an input."""
    panel = plan["panels"]["4_low"]
    cal = np.arange(15, dtype=float)
    chosen = multi.choices(panel, cal)
    for method in ("s1", "s2"):
        modified = cal.copy()
        allowed = panel[f"{method}_order"][:5]
        modified[[i for i in range(15) if i not in allowed]] = -1e9
        assert (
            multi.choices(panel, modified)[f"{method}_top5"] == chosen[f"{method}_top5"]
        )
        assert multi.choices(panel, modified)[method] == chosen[method]
    for b in (1, 3, 5, 10, 11, 15):
        modified = cal.copy()
        modified[panel["direct_prefix_order"][b:]] = -1e9
        assert multi.choices(panel, modified)[f"direct_{b}"] == chosen[f"direct_{b}"]
    assert multi.choices(panel, np.zeros(15))["direct_15"] == 0
    with pytest.raises(ValueError):
        multi.choices(panel, np.ones(14))


def test_g_is_fixed_reference_and_r_is_hindsight() -> None:
    """Bootstrap may reselect the R minimum, never the calibration reference."""
    g, r = multi.losses(np.array([[3, 2, 1], [1, 2, 3]]), 0, 1)
    np.testing.assert_array_equal(g, [1, -1])
    np.testing.assert_array_equal(r, [2, 0])


def test_paired_bootstrap_all_architectures() -> None:
    """Joint draws preserve all architecture offsets, including cross-k dependence."""
    matrix = np.arange(64, dtype=float)[None, :] + np.arange(91)[:, None] * 0.02
    sampled = multi.history.paired_resamples(matrix, 100, 20260923)
    np.testing.assert_allclose(sampled[:, 90] - sampled[:, 0], 1.8)
    assert np.std(sampled[:, 0]) > 0
    np.testing.assert_allclose(
        multi.interval(np.arange(20000), True),
        np.percentile(np.arange(20000), [0.833333, 99.166667]),
    )


def test_rank_metrics_match_scipy_with_ties() -> None:
    """Vectorized bootstrap metrics agree with scalar reference definitions."""
    predicted = np.array([0, 1, 1, 3, 4, 7, 6, 8, 8, 10, 11, 12, 13, 14, 15])
    actual = predicted[::-1].copy()
    result = multi.prediction_metrics(predicted, actual)
    assert result["kendall_tau_b"][0] == pytest.approx(
        multi.stats.kendalltau(predicted, actual).statistic
    )
    assert result["spearman"][0] == pytest.approx(
        multi.stats.spearmanr(predicted, actual).statistic
    )
    assert result["top_15_overlap_fraction"][0] == 1


def test_budget_includes_safety_and_rejects_expansion() -> None:
    """Measurements and checks have independent hard ceilings."""
    budget = multi.WindowBudget()
    budget.charge(90 * (96 + 64) + 64)
    budget.charge(16, True)
    assert (budget.measurement + budget.safety) * 512 == 7413760
    with pytest.raises(RuntimeError):
        budget.charge(1)
    with pytest.raises(RuntimeError):
        multi.WindowBudget().charge(17, True)


def decision_fixture() -> dict:
    """A synthetic fully qualifying case, not model evidence."""
    panels = {}
    for k in (4, 6, 8):
        row = {
            "G_interval": [0.03, 0.07],
            "R_interval": [0.03, 0.07],
            "G_adjusted_interval": [0.025, 0.075],
        }
        strategies = {
            name: copy.deepcopy(row)
            for name in ("s1", "s2", "s1_top5", "direct_5", "direct_11", "direct_15")
        }
        strategies["direct_15"] = {"G_interval": [0, 0], "R_interval": [0, 0.01]}
        panels[f"{k}_low"] = {
            "practical": True,
            "strategies": strategies,
            "confirmation_slice_G": [0.04, 0.05],
            "calibration_shards": [{"s1": {"G": 0.03}}] * 3,
        }
    return panels


def test_continuation_requires_same_two_stable_practical_unsolved_k() -> None:
    """No near-pass interpretation, no nonsignificance-as-failure or automatic H2."""
    panels = decision_fixture()
    result = multi.decision(panels)
    assert result["verdict"] == "REQUEST_H2_FEASIBILITY_REVIEW_ONLY"
    assert result["h2"] is False
    panels["4_low"]["strategies"]["s2"]["G_interval"][0] = 0.02
    panels["6_low"]["confirmation_slice_G"][0] = 0.02
    assert multi.decision(panels)["verdict"] == "INCONCLUSIVE"
    assert multi.decision(panels, False)["verdict"] == "ENGINEERING_INCOMPLETE"


def test_stop_and_drift_conditions() -> None:
    """Damage-only and solved selections stop; calibration drift prevents support."""
    panels = decision_fixture()
    for p in panels.values():
        p["strategies"]["direct_15"]["R_interval"][1] = 0.03
    assert multi.decision(panels)["verdict"] == "INCONCLUSIVE"
    for p in panels.values():
        p["strategies"]["direct_11"] = {"G_interval": [0, 0], "R_interval": [0, 0.01]}
    assert multi.decision(panels)["verdict"] == "STOP_COMPLEX_METHOD_INVESTMENT"
    for p in panels.values():
        p["practical"] = False
    assert multi.decision(panels)["verdict"] == "STOP_COMPLEX_METHOD_INVESTMENT"


def test_manifest_round_trip(plan: dict) -> None:
    """JSON's tuple/list conversion must not change any frozen digest."""
    assert multi.canonical(json.loads(multi.canonical(plan))) == multi.canonical(plan)


def test_configuration_cannot_change_threshold_budget_or_seed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """No runtime options can silently override the frozen scientific contract."""
    assert multi.configuration()["bootstrap_replicates"] == 20000
    corrupted = tmp_path / "config.yaml"
    corrupted.write_text(
        multi.CONFIG.read_text().replace("epsilon_nll: 0.02", "epsilon_nll: 0.019")
    )
    monkeypatch.setattr(multi, "CONFIG", corrupted)
    with pytest.raises(ValueError, match="configuration changed"):
        multi.configuration()


@pytest.mark.parametrize(
    "corruption", [None, "duplicate", "missing", "nonfinite", "misaligned"]
)
def test_full_measurement_alignment(
    plan: dict, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, corruption: str | None
) -> None:
    """Synthetic complete 14,464-row evidence exercises exact identity checks."""
    monkeypatch.setattr(multi, "OUTPUT", tmp_path)
    plan = json.loads(multi.canonical(plan))
    manifest = {
        "windows": {
            split: {"source_window_indices": list(range(size))}
            for split, size in (("calibration", 96), ("V_new", 64))
        }
    }
    rows = []
    for split, size in (("calibration", 96), ("V_new", 64)):
        for key, p in plan["panels"].items():
            for arch in p["architectures"]:
                for i in range(size):
                    rows.append(
                        {
                            "split": split,
                            "k": len(arch),
                            "stratum": key.split("_")[1],
                            "architecture": ",".join(map(str, arch)),
                            "sample_index": i,
                            "source_window_index": i,
                            "nll": 2.0 + i / 100,
                        }
                    )
    for i in range(64):
        rows.append(
            {
                "split": "V_new",
                "k": 0,
                "stratum": "parent",
                "architecture": "",
                "sample_index": i,
                "source_window_index": i,
                "nll": 2.0,
            }
        )
    assert len(rows) == 14464
    if corruption == "duplicate":
        rows.append(rows[-1].copy())
    elif corruption == "missing":
        rows.pop()
    elif corruption == "nonfinite":
        rows[0]["nll"] = float("nan")
    elif corruption == "misaligned":
        rows[0]["source_window_index"] = 999
    multi.h1.write_csv(tmp_path / "per_window_nll.csv", rows)
    if corruption:
        with pytest.raises(ValueError):
            multi.load_measurements(plan, manifest)
    else:
        c, v, parent = multi.load_measurements(plan, manifest)
        assert all(a.shape == (15, 96) for a in c.values())
        assert all(a.shape == (15, 64) for a in v.values())
        assert parent.shape == (64,)
        # Exercise the entire 20,000-replicate CPU pipeline on synthetic values.
        # Nothing generated here is a scientific result or leaves pytest tmp_path.
        multi.save("predictions.json", plan)
        multi.save("manifest.json", manifest)
        multi.save(
            "calibration_choices.json",
            {
                "choices": {
                    key: multi.choices(p, c[key].mean(axis=1))
                    for key, p in plan["panels"].items()
                }
            },
        )
        multi.save(
            "state.json",
            {
                "phase": "EVALUATED",
                "per_window_sha256": multi.file_digest(tmp_path / "per_window_nll.csv"),
                "safety_windows": 8,
                "executed_code_commit": "synthetic_cpu_test",
                "selected_gpu": "NONE",
                "gpu_name": "NONE",
                "gpu_runtime_seconds": 0,
                "peak_allocated_memory_mib": 0,
            },
        )
        multi.analyze()
        metrics = multi.read("metrics.json")
        assert metrics["bootstrap"]["replicates"] == 20000
        assert metrics["decision"]["h2"] is False
        assert multi.read("state.json")["phase"] == "COMPLETE"
        assert all(
            strategy["G"] == 0 and strategy["R"] == 0
            for panel in metrics["panels"].values()
            for strategy in panel["strategies"].values()
        )


def test_v2_cross_length_overlap_and_boundaries() -> None:
    """Real 256/512 intersections and touching boundaries use token coordinates."""
    from paper5.experiments.canary.direction_01 import h1_isolation as iso

    assert iso.overlap_length((324 * 256, 325 * 256), (162 * 512, 163 * 512)) == 256
    assert iso.overlap_length((784 * 256, 785 * 256), (392 * 512, 393 * 512)) == 256
    assert iso.overlap_length((0, 256), (256, 768)) == 0
    assert iso.overlap_length((500, 756), (512, 1024)) == 244


def test_v2_frozen_selection_neighbors_and_candidates(plan: dict) -> None:
    """Replay the actual amendment, including indirect exclusions and exact order."""
    from paper5.experiments.canary.direction_01 import h1_isolation as iso

    audit = json.loads(multi.ISOLATION_RECORD.read_text())
    original_plan = multi.canonical(plan)
    selected = iso.select_windows(
        528, audit["V_old"], audit["historical_exposures"], audit["coordinate"]
    )
    assert selected == audit["selection"]
    assert selected == iso.select_windows(
        528, audit["V_old"][::-1], audit["historical_exposures"], audit["coordinate"]
    )
    assert len(set(selected["selected"])) == 64
    assert selected["eligible_count"] == 325
    assert sorted(set(audit["original_V_new"]) - set(selected["selected"])) == [
        37,
        162,
        163,
        313,
        392,
        393,
    ]
    assert selected["selected"][-6:] == [436, 73, 364, 527, 476, 218]
    for exposure in audit["historical_exposures"]:
        for j in selected["selected"]:
            assert (
                iso.overlap_length(
                    tuple(exposure["source_token_range"]), (512 * j, 512 * (j + 1))
                )
                == 0
            )
    for item in selected["overlaps"]:
        assert not set(range(item["window"] - 1, item["window"] + 2)) & set(
            selected["selected"]
        )
    cal, _ = multi.archive()
    assert multi.canonical(multi.candidate_plan(cal)) == original_plan
    assert audit["candidate_manifest_sha256"] == multi.MANIFEST_HASH


def test_v2_rejects_incompatible_coordinates_and_insufficient_windows() -> None:
    """Do not silently compare distinct token streams or reduce the 64-window count."""
    from paper5.experiments.canary.direction_01 import h1_isolation as iso

    audit = json.loads(multi.ISOLATION_RECORD.read_text())
    exposures = copy.deepcopy(audit["historical_exposures"])
    exposures[0]["coordinate"]["full_token_stream_sha256"] = "different"
    with pytest.raises(ValueError, match="coordinate"):
        iso.select_windows(528, audit["V_old"], exposures, audit["coordinate"])
    exposures = [
        {
            "id": "all",
            "source_token_range": [0, 270674],
            "coordinate": audit["coordinate"],
        }
    ]
    with pytest.raises(ValueError, match="Fewer than 64"):
        iso.select_windows(528, audit["V_old"], exposures, audit["coordinate"])
