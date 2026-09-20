"""CPU-only audit of the immutable H1-small archive; never load a model."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import io
import itertools
import json
import math
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
import scipy
import yaml
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = "02bdcd523ed2c2715a396fd624b647ef356a1c87"
RESULTS = "paper5/results/canary/direction_01/H1_small/"
RUNNER = "paper5/experiments/canary/direction_01/run_h1_small.py"
LAYERS = (1, 3, 6, 8, 10, 12, 15, 17, 19, 21, 24, 26)
PAIRS = tuple(itertools.combinations(LAYERS, 2))
ARCHITECTURES = ((), *((layer,) for layer in LAYERS), *PAIRS)
TOP_K = (1, 3, 5, 10, 17)
SEED = 20260918
CANDIDATE_SEED = 20260920
FILES = (
    "results.csv",
    "per_window_nll.csv",
    "metrics.json",
    "summary.md",
    "config_used.yaml",
    "interaction_heatmap.svg",
)


def archive_bytes(path: str) -> bytes:
    """Read a Git blob from the fixed evidence commit, never the worktree."""
    return subprocess.check_output(["git", "show", f"{ARCHIVE}:{path}"], cwd=ROOT)


def load_windows(
    text: str, source_indices: dict[str, list[int]]
) -> dict[str, dict[tuple[int, ...], np.ndarray]]:
    """Reject missing, duplicate, nonfinite or incorrectly aligned observations."""
    sizes = {"calibration": 96, "validation": 64}
    windows = {
        split: {arch: np.full(size, np.nan) for arch in ARCHITECTURES}
        for split, size in sizes.items()
    }
    for split, size in sizes.items():
        if (
            len(source_indices[split]) != size
            or len(set(source_indices[split])) != size
        ):
            raise ValueError("Invalid source-window manifest")
    for row in csv.DictReader(io.StringIO(text)):
        split = row["split"]
        arch = tuple(
            int(row[key]) for key in ("first_layer", "second_layer") if row[key]
        )
        sample = int(row["sample_index"])
        if (
            split not in sizes
            or arch not in windows[split]
            or not 0 <= sample < sizes[split]
        ):
            raise ValueError("Unexpected split, architecture or sample")
        if int(row["source_window_index"]) != source_indices[split][sample]:
            raise ValueError("Architecture windows are not paired")
        if not np.isnan(windows[split][arch][sample]):
            raise ValueError("Duplicate architecture/window observation")
        value = float(row["nll"])
        if not np.isfinite(value):
            raise ValueError("Nonfinite NLL")
        windows[split][arch][sample] = value
    if any(
        not np.isfinite(v).all() for split in windows.values() for v in split.values()
    ):
        raise ValueError("Missing architecture/window observation")
    return windows


def effects(windows: dict[tuple[int, ...], np.ndarray]) -> dict[tuple[int, ...], float]:
    """Reconstruct equal-token-length window means and parent-relative effects."""
    parent = float(np.mean(windows[()]))
    return {arch: float(np.mean(values)) - parent for arch, values in windows.items()}


def predict(
    calibration: dict[tuple[int, ...], float], features: dict[tuple[int, ...], float]
) -> dict[str, np.ndarray]:
    """Fit coefficients on calibration labels only; accept explicit feature split."""
    design = np.array([[1, calibration[(i,)], calibration[(j,)]] for i, j in PAIRS])
    targets = np.array([calibration[pair] for pair in PAIRS])
    coefficients, _, rank, _ = np.linalg.lstsq(design, targets, rcond=None)
    if rank != 3:
        raise ValueError("Rank-deficient calibration design")
    eval_design = np.array([[1, features[(i,)], features[(j,)]] for i, j in PAIRS])
    additive = eval_design[:, 1:].sum(axis=1)
    offset = float(np.mean(targets - design[:, 1:].sum(axis=1)))
    return {
        "additive": additive,
        "mean_interaction": additive + offset,
        "linear_regression": eval_design @ coefficients,
    }


def order(values: np.ndarray) -> list[int]:
    """Break exact ties by the fixed lexicographic architecture order."""
    return sorted(range(len(values)), key=lambda index: (values[index], index))


def metrics(predicted: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    """Independently check the archived rank helpers with SciPy statistics."""
    error = predicted - actual
    count = math.ceil(len(actual) / 4)
    return {
        "mae": float(np.mean(np.abs(error))),
        "median_absolute_error": float(np.median(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "pearson": float(stats.pearsonr(predicted, actual).statistic),
        "spearman": float(stats.spearmanr(predicted, actual).statistic),
        "kendall_tau_b": float(stats.kendalltau(predicted, actual).statistic),
        "top_quartile_overlap": len(
            set(order(predicted)[:count]) & set(order(actual)[:count])
        )
        / count,
    }


def selection(
    predicted: np.ndarray, actual: np.ndarray, calibration_actual: np.ndarray
) -> dict[str, Any]:
    """Separate hindsight shortlist quality from calibration-only reranking."""
    ranked, oracle = order(predicted), order(actual)
    chosen, best = ranked[0], oracle[0]
    shortlists = {}
    for count in TOP_K:
        shortlist = ranked[:count]
        hindsight = min(shortlist, key=lambda index: (actual[index], index))
        deploy = min(shortlist, key=lambda index: (calibration_actual[index], index))
        shortlists[str(count)] = {
            "overlap_count": len(set(shortlist) & set(oracle[:count])),
            "overlap_fraction": len(set(shortlist) & set(oracle[:count])) / count,
            "predicted_pairs": [PAIRS[i] for i in shortlist],
            "hindsight_best_pair": PAIRS[hindsight],
            "hindsight_best_validation_delta": float(actual[hindsight]),
            "hindsight_regret": float(actual[hindsight] - actual[best]),
            "calibration_rerank_pair": PAIRS[deploy],
            "calibration_rerank_validation_delta": float(actual[deploy]),
            "calibration_rerank_regret": float(actual[deploy] - actual[best]),
        }
    return {
        "selected_pair": PAIRS[chosen],
        "reference_best_pair": PAIRS[best],
        "selected_prediction": float(predicted[chosen]),
        "selected_validation_delta": float(actual[chosen]),
        "simple_regret": float(actual[chosen] - actual[best]),
        "predicted_top3": [PAIRS[i] for i in ranked[:3]],
        "reference_top3": [PAIRS[i] for i in oracle[:3]],
        "shortlists": shortlists,
    }


def paired_resamples(matrix: np.ndarray, replicates: int, seed: int) -> np.ndarray:
    """Resample windows jointly across rows, not architectures or overlapping pairs."""
    rng = np.random.default_rng(seed)
    return np.array(
        [
            matrix[:, rng.integers(0, matrix.shape[1], size=matrix.shape[1])].mean(
                axis=1
            )
            for _ in range(replicates)
        ]
    )


def bootstrap(windows: dict[tuple[int, ...], np.ndarray]) -> dict[str, list[float]]:
    """Independently reproduce all five original paired-window 95% intervals."""
    sampled = paired_resamples(
        np.stack([windows[a] for a in ARCHITECTURES]), 2000, SEED
    )
    samples: dict[str, list[float]] = {
        key: []
        for key in (
            "additive_mae",
            "median_relative_residual",
            "spearman",
            "kendall_tau_b",
            "top_quartile_overlap",
        )
    }
    for means in sampled:
        delta = dict(zip(ARCHITECTURES, means - means[0]))
        prediction = np.array([delta[(i,)] + delta[(j,)] for i, j in PAIRS])
        truth = np.array([delta[pair] for pair in PAIRS])
        denominator = np.array(
            [max(abs(delta[(i,)]) + abs(delta[(j,)]), 1e-12) for i, j in PAIRS]
        )
        result = metrics(prediction, truth)
        samples["additive_mae"].append(result["mae"])
        samples["median_relative_residual"].append(
            float(np.median(abs(truth - prediction) / denominator))
        )
        for key in ("spearman", "kendall_tau_b", "top_quartile_overlap"):
            samples[key].append(result[key])
    return {
        key: np.percentile(value, [2.5, 97.5]).tolist()
        for key, value in samples.items()
    }


def compare(actual: Any, expected: Any, path: str = "root") -> list[str]:
    """Locate numeric or structural discrepancies without modifying evidence."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            return [path + ": keys differ"]
        return [
            issue
            for key in expected
            for issue in compare(actual[key], expected[key], f"{path}.{key}")
        ]
    if isinstance(expected, list):
        if len(actual) != len(expected):
            return [path + ": lengths differ"]
        return [
            issue
            for i, value in enumerate(expected)
            for issue in compare(actual[i], value, f"{path}[{i}]")
        ]
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        matches = np.isclose(actual, expected, atol=1e-10, rtol=1e-9)
    else:
        matches = actual == expected
    return [] if matches else [f"{path}: {actual!r} != {expected!r}"]


def candidate_manifest(calibration: dict[tuple[int, ...], float]) -> dict[str, Any]:
    """Design-only deterministic samples; calibration singles are the only labels."""
    manifest = {}
    for count in (4, 6, 8):
        universe = list(itertools.combinations(LAYERS, count))

        def draw_key(
            architecture: tuple[int, ...], stratum: str
        ) -> tuple[str, tuple[int, ...]]:
            key = f"{CANDIDATE_SEED}:{count}:{stratum}:" + ",".join(
                map(str, architecture)
            )
            return hashlib.sha256(key.encode()).hexdigest(), architecture

        uniform = sorted(universe, key=lambda arch: draw_key(arch, "global"))[:15]
        low_pool = sorted(
            universe, key=lambda arch: (sum(calibration[(i,)] for i in arch), arch)
        )[: math.ceil(len(universe) / 4)]
        eligible = [arch for arch in low_pool if arch not in uniform]
        low = sorted(eligible, key=lambda arch: draw_key(arch, "low"))[:15]
        manifest[str(count)] = {
            "universe_count": len(universe),
            "low_pool_count": len(low_pool),
            "global": uniform,
            "low": low,
        }
    return manifest


def exploratory(
    cal: dict[tuple[int, ...], float],
    val: dict[tuple[int, ...], float],
    predictions: dict[str, dict[str, np.ndarray]],
) -> dict[str, Any]:
    """Describe leverage and selection relevance without changing the H1 verdict."""
    truth = np.array([val[pair] for pair in PAIRS])
    additive = predictions["A"]["additive"]
    errors = abs(truth - additive)
    high = sorted(LAYERS, key=lambda i: (-cal[(i,)], i))
    slices = {}
    for count in (0, 1, 2, 3):
        keep = np.array([not set(pair).intersection(high[:count]) for pair in PAIRS])
        slices[f"exclude_calibration_top_{count}_layers"] = {
            "excluded_layers": high[:count],
            "pair_count": int(keep.sum()),
            "A": metrics(additive[keep], truth[keep]),
            "B": metrics(predictions["B"]["additive"][keep], truth[keep]),
        }
    groups = {
        "distance_1_to_5": np.array([j - i <= 5 for i, j in PAIRS]),
        "distance_6_to_12": np.array([5 < j - i <= 12 for i, j in PAIRS]),
        "distance_13_plus": np.array([j - i > 12 for i, j in PAIRS]),
    }
    for name, mask in groups.items():
        slices[name] = {
            "pair_count": int(mask.sum()),
            "A": metrics(additive[mask], truth[mask]),
        }
    contains_high = np.array([high[0] in pair for pair in PAIRS])
    between = sum(
        mask.sum() * (truth[mask].mean() - truth.mean()) ** 2
        for mask in (contains_high, ~contains_high)
    )
    relevance = {}
    for case in ("A", "B"):
        shortlist = set(order(predictions[case]["additive"])[:17])
        mask = np.array([i in shortlist for i in range(66)])
        relevance[case] = {
            "top17_additive_mae_same_split": float(errors[mask].mean()),
            "outside_top17_absolute_interaction_share": float(
                errors[~mask].sum() / errors.sum()
            ),
            "outside_top17_squared_interaction_share": float(
                (errors[~mask] ** 2).sum() / (errors**2).sum()
            ),
            "local_additive_metrics": metrics(
                predictions[case]["additive"][mask], truth[mask]
            ),
        }
    return {
        "label": "exploratory; not a replacement verdict",
        "calibration_single_damage_descending": high,
        "highest_layer_pair_count": int(contains_high.sum()),
        "highest_layer_between_group_validation_variance_fraction": float(
            between / np.sum((truth - truth.mean()) ** 2)
        ),
        "highest_layer_absolute_interaction_share": float(
            errors[contains_high].sum() / errors.sum()
        ),
        "highest_layer_squared_interaction_share": float(
            (errors[contains_high] ** 2).sum() / (errors**2).sum()
        ),
        "slices": slices,
        "selection_relevance": relevance,
        "largest_absolute_interactions": [
            {
                "pair": PAIRS[i],
                "interaction": float(truth[i] - additive[i]),
                "actual_delta": float(truth[i]),
                "actual_rank": order(truth).index(i) + 1,
                "A_additive_rank": order(additive).index(i) + 1,
                "B_additive_rank": order(predictions["B"]["additive"]).index(i) + 1,
            }
            for i in sorted(range(66), key=lambda i: (-errors[i], i))[:10]
        ],
    }


def review() -> dict[str, Any]:
    """Audit immutable evidence and return reproducible CPU-only diagnostics."""
    blobs = {name: archive_bytes(RESULTS + name) for name in FILES}
    archived = json.loads(blobs["metrics.json"])
    config = yaml.safe_load(
        archive_bytes("paper5/configs/canary/direction_01/h1_config.yaml")
    )
    used = yaml.safe_load(blobs["config_used.yaml"])
    issues = compare(used.pop("run_metadata"), archived["metadata"], "metadata")
    issues += compare(used, config, "config")
    metadata = archived["metadata"]
    windows = load_windows(
        blobs["per_window_nll.csv"].decode(), metadata["selected_source_window_indices"]
    )
    cal, val = (effects(windows[split]) for split in ("calibration", "validation"))
    # Replay only CPU analysis from checked archived code; no model/GPU imports.
    helpers = "paper5/experiments/canary/direction_01/smoke/analysis.py"
    for path in (RUNNER, helpers):
        if (ROOT / path).read_bytes() != archive_bytes(path):
            raise ValueError(
                "Analysis dependency differs from the frozen commit: " + path
            )
    runner = importlib.import_module(
        "paper5.experiments.canary.direction_01.run_h1_small"
    )
    rows, replay = runner.analyze_h1_small(
        windows["calibration"], windows["validation"], list(LAYERS), 3, 2000, SEED
    )
    issues += compare(replay, archived["analysis"], "analysis_replay")
    csv_rows = list(csv.DictReader(io.StringIO(blobs["results.csv"].decode())))
    parsed_rows = [
        {key: (value if key == "split" else float(value)) for key, value in row.items()}
        for row in csv_rows
    ]
    issues += compare(rows, parsed_rows, "results_csv")
    predictions = {"A": predict(cal, val), "B": predict(cal, cal)}
    truth, cal_truth = (np.array([data[pair] for pair in PAIRS]) for data in (val, cal))
    independent = {
        case: {name: metrics(pred, truth) for name, pred in baselines.items()}
        for case, baselines in predictions.items()
    }
    for case, key in (
        ("A", "baseline_metrics"),
        ("B", "deployment_metrics_using_calibration_singles"),
    ):
        issues += compare(
            independent[case], archived["analysis"][key], "independent_" + case
        )
    intervals = bootstrap(windows["validation"])
    issues += compare(
        intervals,
        archived["analysis"]["paired_validation_window_bootstrap_95pct"],
        "independent_bootstrap",
    )
    # Independently check rounded human-readable and SVG evidence too.
    summary = blobs["summary.md"].decode()
    for name, values in independent["A"].items():
        line = (
            f"| {name} | "
            + " | ".join(
                f"{values[key]:.6g}"
                for key in (
                    "mae",
                    "rmse",
                    "spearman",
                    "kendall_tau_b",
                    "top_quartile_overlap",
                )
            )
            + " |"
        )
        if line not in summary:
            issues.append("Summary baseline row mismatch: " + name)
    svg = ET.fromstring(blobs["interaction_heatmap.svg"])
    cells = 0
    for title in svg.findall(".//{http://www.w3.org/2000/svg}title"):
        coordinates, value = title.text.split(": ")
        i, j = map(int, coordinates.split(", "))
        if i != j:
            cells += 1
            residual = val[tuple(sorted((i, j)))] - val[(i,)] - val[(j,)]
            issues += compare(float(value), residual, "heatmap_cell")
    if cells != 132:
        issues.append("Heatmap does not contain both triangles of 66 pairs")
    provenance = {
        "runner_sha_matches": hashlib.sha256(archive_bytes(RUNNER)).hexdigest()
        == metadata["code_sha256"]["runner"],
        "core_sha_matches": hashlib.sha256(
            archive_bytes("paper5/experiments/canary/direction_01/smoke/core.py")
        ).hexdigest()
        == metadata["code_sha256"]["edit_core"],
        "code_commit_is_archive_parent": subprocess.check_output(
            ["git", "rev-parse", ARCHIVE + "^"], cwd=ROOT, text=True
        ).strip()
        == metadata["started_from_commit"],
        "model_revision_matches": config["model"]["revision"]
        == metadata["resolved_revision"]
        == metadata["requested_revision"],
        "data_hash_manifest_matches": all(
            config["dataset"][split]["sha256"] == metadata["source_sha256"][split]
            for split in windows
        ),
        "seed_matches": config["random_seed"] == SEED,
    }
    issues += [key for key, value in provenance.items() if not value]
    return {
        "archive": ARCHIVE,
        "input_sha256": {
            name: hashlib.sha256(blob).hexdigest() for name, blob in blobs.items()
        },
        "cpu_versions": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "python": sys.version.split()[0],
        },
        "provenance": provenance,
        "discrepancies": issues,
        "window_contract": {
            "calibration": 96,
            "validation": 64,
            "architectures_per_split": 79,
            "rows": 12640,
            "aligned": True,
        },
        "means": {
            split: {
                (",".join(map(str, arch)) or "parent"): float(np.mean(values))
                for arch, values in data.items()
            }
            for split, data in windows.items()
        },
        "singles": {
            str(i): {"calibration_delta": cal[(i,)], "validation_delta": val[(i,)]}
            for i in LAYERS
        },
        "baselines": independent,
        "selection": {
            case: {
                name: selection(pred, truth, cal_truth)
                for name, pred in baselines.items()
            }
            for case, baselines in predictions.items()
        },
        "bootstrap_95pct": intervals,
        "original_analysis_replayed": replay,
        "exploratory": exploratory(cal, val, predictions),
        "proposed_candidate_seed": CANDIDATE_SEED,
        "proposed_candidates_design_only": candidate_manifest(cal),
    }


def main() -> None:
    """Print the audit or save it only under the ignored local-deps directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and not args.output.resolve().is_relative_to(ROOT / ".local-deps"):
        parser.error("Output is restricted to this checkout's .local-deps directory")
    result = review()
    text = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
        print(
            json.dumps(
                {
                    "archive": ARCHIVE,
                    "discrepancies": result["discrepancies"],
                    "verdict": result["original_analysis_replayed"]["h1_verdict"],
                }
            )
        )
    else:
        print(text, end="")
    if result["discrepancies"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
