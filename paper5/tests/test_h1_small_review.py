"""CPU-only checks for evidence alignment, selection leakage and resampling."""

from __future__ import annotations

import builtins
import hashlib
import json
import re

import numpy as np
import pytest

from paper5.analysis import review_h1_small as review


@pytest.fixture(scope="module")
def evidence() -> tuple[str, dict[str, list[int]]]:
    """Load only immutable archived observations and their source manifest."""
    metadata = json.loads(review.archive_bytes(review.RESULTS + "metrics.json"))[
        "metadata"
    ]
    return (
        review.archive_bytes(review.RESULTS + "per_window_nll.csv").decode(),
        metadata["selected_source_window_indices"],
    )


@pytest.mark.parametrize(
    "corruption", ["duplicate", "missing", "misaligned", "nonfinite"]
)
def test_reject_corrupt_window_evidence(
    evidence: tuple[str, dict[str, list[int]]], corruption: str
) -> None:
    """A scalar row count cannot hide broken shared-window pairing."""
    text, manifest = evidence
    lines = text.splitlines()
    if corruption == "duplicate":
        lines.append(lines[1])
    elif corruption == "missing":
        lines.pop()
    else:
        row = lines[1].split(",")
        row[4 if corruption == "misaligned" else 5] = (
            "999999" if corruption == "misaligned" else "nan"
        )
        lines[1] = ",".join(row)
    with pytest.raises(ValueError):
        review.load_windows("\n".join(lines), manifest)


def test_validation_pair_labels_cannot_change_predictions(
    evidence: tuple[str, dict[str, list[int]]],
) -> None:
    """Neither A nor B learns parameters or candidate scores from validation pairs."""
    windows = review.load_windows(*evidence)
    cal = review.effects(windows["calibration"])
    val = review.effects(windows["validation"])
    changed = dict(val)
    for index, pair in enumerate(review.PAIRS):
        changed[pair] = 1000 - index
    for name, prediction in review.predict(cal, val).items():
        np.testing.assert_array_equal(prediction, review.predict(cal, changed)[name])
    b_prediction = review.predict(cal, cal)["additive"]
    actual = np.array([val[pair] for pair in review.PAIRS])
    cal_actual = np.array([cal[pair] for pair in review.PAIRS])
    first = review.selection(b_prediction, actual, cal_actual)
    second = review.selection(b_prediction, actual[::-1], cal_actual)
    assert first["selected_pair"] == second["selected_pair"]
    for count in review.TOP_K:
        assert (
            first["shortlists"][str(count)]["calibration_rerank_pair"]
            == second["shortlists"][str(count)]["calibration_rerank_pair"]
        )


def test_shortlist_oracle_is_distinct_from_deployable_calibration_choice() -> None:
    """Good shortlist recall must not silently use validation to select a model."""
    predicted = np.arange(66, dtype=float)
    validation = np.arange(66, dtype=float) + 10
    validation[1] = 1
    result = review.selection(predicted, validation, predicted)
    assert result["selected_pair"] == review.PAIRS[0]
    assert result["reference_best_pair"] == review.PAIRS[1]
    assert result["simple_regret"] == 9
    shortlist = result["shortlists"]["3"]
    assert shortlist["hindsight_regret"] == 0
    assert shortlist["calibration_rerank_regret"] == 9


def test_bootstrap_preserves_window_level_linear_relations() -> None:
    """Joint sampling preserves dependence that independent architecture draws destroy."""
    parent = np.arange(64, dtype=float) ** 2
    single_a = parent + np.sin(parent)
    single_b = parent + np.cos(parent)
    pair = single_a + single_b - parent + 0.3
    means = review.paired_resamples(
        np.stack([parent, single_a, single_b, pair]), 100, 7
    )
    np.testing.assert_allclose(
        means[:, 3] - means[:, 1] - means[:, 2] + means[:, 0], 0.3, atol=1e-10
    )
    assert np.std(means[:, 0]) > 0


def test_candidate_manifest_is_frozen_and_calibration_only(
    evidence: tuple[str, dict[str, list[int]]],
) -> None:
    """Lock the design's 90 unique candidates and exact calibration-only generation."""
    cal = review.effects(review.load_windows(*evidence)["calibration"])
    manifest = review.candidate_manifest(cal)
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    assert (
        hashlib.sha256(canonical).hexdigest()
        == "7c40b1a1963d76e99144d7b27608fbfc6e68f1831e244f332e217a9c704021b9"
    )
    for count, strata in manifest.items():
        assert len(set(strata["global"] + strata["low"])) == 30
        assert all(
            len(arch) == int(count) and tuple(sorted(set(arch))) == arch
            for arch in strata["global"] + strata["low"]
        )


def test_complete_archive_replay_without_torch_or_transformers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reproduce original metrics and decisions while forbidding model-library imports."""
    original_import = builtins.__import__

    def cpu_only_import(name: str, *args: object, **kwargs: object) -> object:
        if name.split(".")[0] in {"torch", "transformers"}:
            raise AssertionError("Model libraries are outside the review scope")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", cpu_only_import)
    result = review.review()
    assert result["discrepancies"] == []
    assert result["original_analysis_replayed"]["h1_verdict"] == "INCONCLUSIVE"
    assert all(result["provenance"].values())
    assert result["window_contract"]["rows"] == 12640
    for baselines in result["selection"].values():
        for baseline in baselines.values():
            assert baseline["simple_regret"] == 0
            assert baseline["predicted_top3"] == baseline["reference_top3"]


def test_protocol_lists_exact_calibration_generated_candidates(
    evidence: tuple[str, dict[str, list[int]]],
) -> None:
    """The human-reviewable proposed manifest must match all 90 frozen tuples."""
    cal = review.effects(review.load_windows(*evidence)["calibration"])
    manifest = review.candidate_manifest(cal)
    protocol = (
        review.ROOT / "paper5/experiments/canary/direction_01/H1_multi_edit_protocol.md"
    ).read_text()
    rows = re.findall(
        r"^\| (4|6|8) \| (global|low) \| (\d+) \| ([0-9,]+) \|$",
        protocol,
        re.MULTILINE,
    )
    assert len(rows) == 90
    assert len({(k, stratum, index) for k, stratum, index, _ in rows}) == 90
    for k, stratum, index, architecture in rows:
        assert (
            tuple(map(int, architecture.split(",")))
            == manifest[k][stratum][int(index) - 1]
        )


def test_output_cannot_overwrite_frozen_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject an original result path before any analysis or write occurs."""
    monkeypatch.setattr(
        review.sys,
        "argv",
        [
            "review_h1_small",
            "--output",
            str(review.ROOT / review.RESULTS / "metrics.json"),
        ],
    )
    with pytest.raises(SystemExit) as error:
        review.main()
    assert error.value.code == 2
