#!/usr/bin/env python3
"""Run the fixed Direction 1 H1-small structural-interaction experiment."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import platform
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
H1_CACHE_ROOT = REPOSITORY_ROOT / "paper5/.cache/h1_small"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ["HF_HOME"] = str(H1_CACHE_ROOT / "huggingface")
os.environ["HF_XET_CACHE"] = str(H1_CACHE_ROOT / "huggingface/xet")

import numpy as np
import yaml

if TYPE_CHECKING:
    import torch
    from torch import Tensor, nn

from paper5.experiments.canary.direction_01.smoke.analysis import (
    kendall_tau_b,
    pearson,
    spearman,
    write_csv,
    write_json,
)

EXPECTED_LAYERS = [1, 3, 6, 8, 10, 12, 15, 17, 19, 21, 24, 26]
EXPECTED_OUTPUT = "paper5/results/canary/direction_01/H1_small"
EXPECTED_REVISION = "8faed761d45a263340a0528343f099c05c9a4323"
EXPECTED_DATASET = {
    "name": "WikiText-2 raw",
    "calibration": {
        "split": "train",
        "url": "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/train.txt",
        "sha256": "9e9fa1ad55b1c2c95b08e37dd8e653f638fac2c6de904b79e813611eefbc985f",
    },
    "validation": {
        "split": "valid",
        "url": "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/valid.txt",
        "sha256": "f0737ed31fc1329026e95cb8b98e19c2a182c39c240ab909dc31abf2f8af58e8",
    },
}
EXPECTED_BASELINES = {
    "fit_split": "calibration",
    "evaluation_split": "validation",
    "prediction_features_split": "validation",
    "additive": "single_effect_sum",
    "mean_interaction": "calibration_mean_pair_residual_offset",
    "linear_regression": {
        "estimator": "ordinary_least_squares",
        "features": ["delta_first", "delta_second"],
        "target": "delta_pair",
        "fit_intercept": True,
    },
}

Architecture = tuple[int, ...]
WindowNll = Mapping[Architecture, np.ndarray]


def _prepare_sequences_with_indices(
    text: str,
    tokenizer: Any,
    sequence_length: int,
    sequence_count: int,
    seed: int,
) -> tuple[Tensor, list[int]]:
    """Select deterministic non-overlapping token windows and retain indices."""
    import torch

    token_ids = tokenizer(
        text,
        add_special_tokens=False,
        return_tensors="pt",
        verbose=False,
    ).input_ids[0]
    chunk_count = token_ids.numel() // sequence_length
    if chunk_count < sequence_count:
        raise ValueError(
            f"Dataset has {chunk_count} chunks, fewer than {sequence_count} requested."
        )
    chunks = token_ids[: chunk_count * sequence_length].reshape(
        chunk_count, sequence_length
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    indices = torch.randperm(chunk_count, generator=generator)[:sequence_count]
    return chunks[indices].contiguous(), indices.tolist()


def _validate_windows(windows: WindowNll, layers: Sequence[int], count: int) -> None:
    expected: set[Architecture] = {()}
    expected.update((layer,) for layer in layers)
    expected.update(itertools.combinations(layers, 2))
    if set(windows) != expected:
        raise ValueError("H1 window evidence must contain exactly 79 architectures.")
    for values in windows.values():
        if values.shape != (count,) or not np.isfinite(values).all():
            raise ValueError("H1 window NLL has an invalid shape or nonfinite value.")


def _effects(windows: WindowNll, layers: Sequence[int]) -> dict[Architecture, float]:
    parent = float(np.mean(windows[()]))
    architectures: list[Architecture] = [(layer,) for layer in layers]
    architectures.extend(itertools.combinations(layers, 2))
    return {
        architecture: float(np.mean(windows[architecture])) - parent
        for architecture in architectures
    }


def _top_overlap(predicted: Sequence[float], actual: Sequence[float]) -> float:
    top_count = math.ceil(len(actual) / 4)
    predicted_best = set(
        sorted(range(len(actual)), key=lambda index: (predicted[index], index))[
            :top_count
        ]
    )
    actual_best = set(
        sorted(range(len(actual)), key=lambda index: (actual[index], index))[:top_count]
    )
    return len(predicted_best & actual_best) / top_count


def _baseline_metrics(
    predicted: Sequence[float], actual: Sequence[float]
) -> dict[str, float]:
    error = np.asarray(predicted, dtype=np.float64) - np.asarray(
        actual, dtype=np.float64
    )
    return {
        "mae": float(np.mean(np.abs(error))),
        "median_absolute_error": float(np.median(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "pearson": pearson(predicted, actual),
        "spearman": spearman(predicted, actual),
        "kendall_tau_b": kendall_tau_b(predicted, actual),
        "top_quartile_overlap": _top_overlap(predicted, actual),
    }


def _additive_arrays(
    effects: Mapping[Architecture, float], pairs: Sequence[tuple[int, int]]
) -> tuple[list[float], list[float], list[float], list[float]]:
    additive = [effects[(first,)] + effects[(second,)] for first, second in pairs]
    actual = [effects[pair] for pair in pairs]
    interaction = [
        observed - predicted for observed, predicted in zip(actual, additive)
    ]
    relative = [
        abs(residual) / max(abs(effects[(first,)]) + abs(effects[(second,)]), 1e-12)
        for (first, second), residual in zip(pairs, interaction)
    ]
    return additive, actual, interaction, relative


def _bootstrap_additive(
    windows: WindowNll,
    layers: Sequence[int],
    pairs: Sequence[tuple[int, int]],
    replicates: int,
    seed: int,
) -> dict[str, list[float]]:
    """Bootstrap validation windows jointly across every architecture."""
    rng = np.random.default_rng(seed)
    count = len(windows[()])
    samples: dict[str, list[float]] = {
        "median_relative_residual": [],
        "additive_mae": [],
        "spearman": [],
        "kendall_tau_b": [],
        "top_quartile_overlap": [],
    }
    for _ in range(replicates):
        indices = rng.integers(0, count, size=count)
        sampled = {
            architecture: values[indices] for architecture, values in windows.items()
        }
        effects = _effects(sampled, layers)
        additive, actual, _, relative = _additive_arrays(effects, pairs)
        ranked = _baseline_metrics(additive, actual)
        samples["median_relative_residual"].append(float(np.median(relative)))
        for key, value in (
            ("additive_mae", ranked["mae"]),
            ("spearman", ranked["spearman"]),
            ("kendall_tau_b", ranked["kendall_tau_b"]),
            ("top_quartile_overlap", ranked["top_quartile_overlap"]),
        ):
            samples[key].append(value)
    return {
        key: [float(value) for value in np.percentile(values, [2.5, 97.5])]
        for key, values in samples.items()
    }


def analyze_h1_small(
    calibration: WindowNll,
    validation: WindowNll,
    layers: Sequence[int],
    calibration_shards: int,
    bootstrap_replicates: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Fit on calibration only and score all baselines on validation."""
    pairs = list(itertools.combinations(layers, 2))
    if list(layers) != EXPECTED_LAYERS or len(pairs) != 66:
        raise ValueError("Only the approved 12-edit, 66-pair scale is supported.")
    if calibration_shards != 3 or bootstrap_replicates < 1:
        raise ValueError("H1-small requires three shards and a positive bootstrap.")
    _validate_windows(calibration, layers, 96)
    _validate_windows(validation, layers, 64)

    calibration_effects = _effects(calibration, layers)
    validation_effects = _effects(validation, layers)
    _, _, calibration_interactions, _ = _additive_arrays(calibration_effects, pairs)

    # These are the only learned baseline parameters. Both are fit exclusively
    # from calibration pair labels and calibration single-edit features.
    mean_interaction = float(np.mean(calibration_interactions))
    calibration_design = np.asarray(
        [
            [1.0, calibration_effects[(first,)], calibration_effects[(second,)]]
            for first, second in pairs
        ],
        dtype=np.float64,
    )
    calibration_targets = np.asarray(
        [calibration_effects[pair] for pair in pairs], dtype=np.float64
    )
    coefficients, _, rank, _ = np.linalg.lstsq(
        calibration_design, calibration_targets, rcond=None
    )
    if rank != 3 or not np.isfinite(coefficients).all():
        raise ValueError("The calibration linear regression design is rank deficient.")

    # Validation pair labels enter only the final metric computation. All
    # predictions use validation single-edit features, as frozen in the protocol.
    additive, actual, interaction, relative = _additive_arrays(
        validation_effects, pairs
    )
    mean_predictions = [prediction + mean_interaction for prediction in additive]
    linear_predictions = [
        float(
            coefficients[0]
            + coefficients[1] * validation_effects[(first,)]
            + coefficients[2] * validation_effects[(second,)]
        )
        for first, second in pairs
    ]
    rows = [
        {
            "split": "validation",
            "first_layer": first,
            "second_layer": second,
            "layer_distance": second - first,
            "delta_first_nll": validation_effects[(first,)],
            "delta_second_nll": validation_effects[(second,)],
            "actual_pair_delta_nll": actual[index],
            "additive_prediction_nll": additive[index],
            "mean_interaction_prediction_nll": mean_predictions[index],
            "linear_regression_prediction_nll": linear_predictions[index],
            "interaction_nll": interaction[index],
            "relative_abs_interaction": relative[index],
        }
        for index, (first, second) in enumerate(pairs)
    ]
    baseline_metrics = {
        "additive": _baseline_metrics(additive, actual),
        "mean_interaction": _baseline_metrics(mean_predictions, actual),
        "linear_regression": _baseline_metrics(linear_predictions, actual),
    }

    # This separately named diagnostic deliberately uses calibration singles.
    # It is never substituted for the held-out primary comparison above.
    calibration_singles_additive = [
        calibration_effects[(first,)] + calibration_effects[(second,)]
        for first, second in pairs
    ]
    deployment_metrics = {
        "additive": _baseline_metrics(calibration_singles_additive, actual),
        "mean_interaction": _baseline_metrics(
            [value + mean_interaction for value in calibration_singles_additive], actual
        ),
        "linear_regression": _baseline_metrics(
            [
                float(
                    coefficients[0]
                    + coefficients[1] * calibration_effects[(first,)]
                    + coefficients[2] * calibration_effects[(second,)]
                )
                for first, second in pairs
            ],
            actual,
        ),
    }

    residual_values = np.asarray(interaction, dtype=np.float64)
    relative_values = np.asarray(relative, dtype=np.float64)
    bootstrap = _bootstrap_additive(
        validation, layers, pairs, bootstrap_replicates, seed
    )
    shard_size = len(calibration[()]) // calibration_shards
    shard_summaries: list[dict[str, float | int]] = []
    shard_pair_scores: list[list[float]] = []
    shard_interactions: list[list[float]] = []
    for shard in range(calibration_shards):
        start = shard * shard_size
        stop = start + shard_size
        shard_windows = {
            architecture: values[start:stop]
            for architecture, values in calibration.items()
        }
        shard_effects = _effects(shard_windows, layers)
        shard_additive, shard_actual, shard_interaction, _ = _additive_arrays(
            shard_effects, pairs
        )
        shard_pair_scores.append(shard_actual)
        shard_interactions.append(shard_interaction)
        shard_summaries.append(
            {
                "shard": shard,
                "mean_abs_interaction": float(np.mean(np.abs(shard_interaction))),
                "additive_spearman": spearman(shard_additive, shard_actual),
                "interaction_sign_agreement_with_full_calibration": float(
                    np.mean(
                        np.sign(shard_interaction) == np.sign(calibration_interactions)
                    )
                ),
            }
        )
    shard_pairwise = [
        {
            "first_shard": first,
            "second_shard": second,
            "pair_damage_spearman": spearman(
                shard_pair_scores[first], shard_pair_scores[second]
            ),
            "interaction_sign_agreement": float(
                np.mean(
                    np.sign(shard_interactions[first])
                    == np.sign(shard_interactions[second])
                )
            ),
        }
        for first, second in itertools.combinations(range(calibration_shards), 2)
    ]

    median_relative = float(np.median(relative_values))
    rho = baseline_metrics["additive"]["spearman"]
    overlap = baseline_metrics["additive"]["top_quartile_overlap"]
    confirmed_conditions = [median_relative >= 0.10, rho <= 0.90, overlap <= 0.75]
    if all(confirmed_conditions):
        point_verdict = "CONFIRMED"
    elif sum(confirmed_conditions) == 2:
        point_verdict = "WEAK SIGNAL"
    elif median_relative < 0.03 and rho >= 0.95 and overlap >= 0.90:
        point_verdict = "NOT CONFIRMED"
    else:
        point_verdict = "INCONCLUSIVE"
    stable_confirmation = (
        point_verdict == "CONFIRMED"
        and bootstrap["median_relative_residual"][0] > 0.03
        and bootstrap["spearman"][1] < 0.95
        and bootstrap["top_quartile_overlap"][1] < 0.90
    )
    h1_verdict = (
        "INCONCLUSIVE"
        if point_verdict == "CONFIRMED" and not stable_confirmation
        else point_verdict
    )
    metrics: dict[str, Any] = {
        "scientific_scope": "H1-small empirical observation",
        "pair_count": 66,
        "single_edit_count": 12,
        "split_contract": {
            "fit_split": "calibration",
            "evaluation_split": "validation",
            "primary_prediction_features_split": "validation",
            "validation_pair_labels_used_for_fitting": False,
        },
        "baseline_metrics": baseline_metrics,
        "deployment_metrics_using_calibration_singles": deployment_metrics,
        "mean_calibration_interaction_nll": mean_interaction,
        "linear_regression_coefficients": [float(value) for value in coefficients],
        "interaction_distribution_nll": {
            "minimum": float(np.min(residual_values)),
            "q25": float(np.percentile(residual_values, 25)),
            "median": float(np.median(residual_values)),
            "q75": float(np.percentile(residual_values, 75)),
            "maximum": float(np.max(residual_values)),
            "mean": float(np.mean(residual_values)),
            "mean_absolute": float(np.mean(np.abs(residual_values))),
            "median_absolute": float(np.median(np.abs(residual_values))),
            "positive_count": int(np.sum(residual_values > 0)),
            "negative_count": int(np.sum(residual_values < 0)),
            "zero_count": int(np.sum(residual_values == 0)),
            "amplification_count": int(np.sum(residual_values > 0)),
            "cancellation_count": int(np.sum(residual_values < 0)),
            "relative_absolute": {
                "minimum": float(np.min(relative_values)),
                "q25": float(np.percentile(relative_values, 25)),
                "median": median_relative,
                "q75": float(np.percentile(relative_values, 75)),
                "maximum": float(np.max(relative_values)),
                "mean": float(np.mean(relative_values)),
                "fraction_greater_than_0_10": float(np.mean(relative_values > 0.10)),
            },
            "median_relative_absolute": median_relative,
            "small_single_denominator_pair_count": sum(
                abs(validation_effects[(first,)]) + abs(validation_effects[(second,)])
                < 0.01
                for first, second in pairs
            ),
        },
        "paired_validation_window_bootstrap_95pct": bootstrap,
        "calibration_shards": shard_summaries,
        "calibration_shard_pairwise": shard_pairwise,
        "protocol_point_verdict": point_verdict,
        "stable_confirmation": stable_confirmation,
        "h1_verdict": h1_verdict,
    }
    return rows, metrics


def write_interaction_heatmap(
    path: Path, rows: Sequence[Mapping[str, Any]], layers: Sequence[int]
) -> None:
    """Write a signed pair-interaction SVG without plotting dependencies."""
    cell = 36
    margin = 70
    size = margin + cell * len(layers) + 45
    values = {
        (int(row["first_layer"]), int(row["second_layer"])): float(
            row["interaction_nll"]
        )
        for row in rows
    }
    maximum_absolute = max((abs(value) for value in values.values()), default=0.0)
    scale = maximum_absolute or 1.0
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}">',
        '<rect width="100%" height="100%" fill="white" />',
        '<text x="20" y="25" font-family="sans-serif" font-size="16">H1-small interaction residual (NLL)</text>',
    ]
    for index, layer in enumerate(layers):
        axis = margin + index * cell + cell / 2
        parts.append(
            f'<text x="{axis}" y="{margin - 8}" text-anchor="middle" font-size="11">{layer}</text>'
        )
        parts.append(
            f'<text x="{margin - 8}" y="{axis + 4}" text-anchor="end" font-size="11">{layer}</text>'
        )
    for row, first in enumerate(layers):
        for column, second in enumerate(layers):
            pair = tuple(sorted((first, second)))
            value = values.get(pair) if first != second else None
            if value is None:
                fill = "#eeeeee"
            else:
                intensity = min(abs(value) / scale, 1.0)
                pale = round(255 - 150 * intensity)
                fill = (
                    f"#ff{pale:02x}{pale:02x}"
                    if value > 0
                    else f"#{pale:02x}{pale:02x}ff"
                )
            x = margin + column * cell
            y = margin + row * cell
            label = value if value is not None else "diagonal"
            parts.append(
                f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{fill}" stroke="white"><title>{first}, {second}: {label}</title></rect>'
            )
    parts.append(
        f'<text x="{margin}" y="{size - 10}" font-family="sans-serif" font-size="11">Blue: cancellation; red: amplification; maximum |I| = {maximum_absolute:.6g} NLL</text>'
    )
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Accept only the reviewed H1-small YAML configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "paper5/configs/canary/direction_01/h1_config.yaml",
    )
    return parser.parse_args()


def load_h1_config(path: Path) -> dict[str, Any]:
    """Reject any change to the approved H1-small experimental scale."""
    if not path.resolve().is_relative_to(REPOSITORY_ROOT):
        raise ValueError("H1 config must be inside the repository.")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("H1 config must be a YAML mapping.")
    required = {
        "scope": "h1_small_hypothesis_validation",
        "status": "approved_h1_small",
        "sequence_length": 512,
        "batch_size": 1,
        "output_directory": EXPECTED_OUTPUT,
    }
    for key, expected in required.items():
        if config.get(key) != expected:
            raise ValueError(f"H1-small requires {key}={expected!r}.")
    if config["model"]["id"] != "Qwen/Qwen2.5-1.5B":
        raise ValueError("H1-small requires Qwen2.5-1.5B.")
    if config["model"]["revision"] != EXPECTED_REVISION:
        raise ValueError("H1-small requires the reviewed model revision.")
    if config["model"]["dtype"] != "bfloat16":
        raise ValueError("H1-small requires bfloat16 evaluation.")
    if config["dataset"] != EXPECTED_DATASET:
        raise ValueError("H1-small requires the reviewed WikiText-2 sources.")
    if config["edit"]["family"] != "whole_transformer_block_skip":
        raise ValueError("H1-small requires whole-block intervention.")
    if (
        config["edit"]["role"] != "structural_intervention_operator"
        or config["edit"]["purpose"] != "measure_architecture_interaction"
        or config["edit"]["pruning_objective"] is not False
    ):
        raise ValueError("H1-small must remain an interaction measurement.")
    if config["edit"]["layer_indices"] != EXPECTED_LAYERS:
        raise ValueError("H1-small edit positions differ from the reviewed protocol.")
    if config["calibration_size"] != {
        "sequences": 96,
        "shards": 3,
        "sequences_per_shard": 32,
    }:
        raise ValueError("H1-small calibration size must be 96 windows in 3 shards.")
    if config["validation_size"] != {"sequences": 64}:
        raise ValueError("H1-small validation size must be 64 windows.")
    if config["pair_sampling"] != {
        "strategy": "exhaustive_unordered",
        "pair_count": 66,
    }:
        raise ValueError("H1-small must evaluate all 66 unordered pairs.")
    if config["baselines"] != EXPECTED_BASELINES:
        raise ValueError("H1-small baselines differ from the reviewed protocol.")
    if config.get("h1_extension") != {
        "condition": "stable_decision_relevant_additive_breakdown",
        "plan": "expand_block_locations_and_evaluate_more_pairs",
        "status": "requires_separate_protocol_and_approval",
    }:
        raise ValueError("H1 extension must remain separately gated and unapproved.")
    if config["variance"] != {
        "bootstrap_replicates": 2000,
        "bootstrap_unit": "validation_window",
    }:
        raise ValueError("H1-small bootstrap setting differs from the protocol.")
    if config["random_seed"] != 20260918:
        raise ValueError("H1-small requires the reviewed random seed.")
    return config


def _checked_text(url: str, expected_sha256: str, destination: Path) -> str:
    """Load one public source file after verifying its frozen SHA-256 digest."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        content = destination.read_bytes()
    else:
        with urllib.request.urlopen(url, timeout=120) as response:  # noqa: S310
            content = response.read()
    actual_sha256 = hashlib.sha256(content).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(f"Source hash mismatch for {destination.name}.")
    if not destination.exists():
        destination.write_bytes(content)
    return content.decode("utf-8")


def _require_a800() -> tuple[torch.device, str, str]:
    """Require the dynamically selected single NVIDIA A800 80GB device."""
    import torch

    if os.environ.get("CONDA_DEFAULT_ENV") != "autoresearch_paper5":
        raise RuntimeError("Activate the autoresearch_paper5 environment first.")
    selected = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not selected.isdecimal():
        raise RuntimeError("Source select_gpu.sh before running H1-small.")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("H1-small requires exactly one visible CUDA GPU.")
    device = torch.device("cuda:0")
    name = torch.cuda.get_device_name(device)
    memory_gib = torch.cuda.get_device_properties(device).total_memory / (1024**3)
    if "A800" not in name or memory_gib < 70:
        raise RuntimeError("Selected CUDA device is not an NVIDIA A800 80GB.")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    return device, selected, name


def evaluate_window_nll(
    model: nn.Module, sequences: Tensor, batch_size: int, device: torch.device
) -> np.ndarray:
    """Return token-mean next-token NLL separately for every fixed window."""
    import torch
    import torch.nn.functional as functional

    with torch.inference_mode():
        values: list[float] = []
        for start in range(0, sequences.shape[0], batch_size):
            batch = sequences[start : start + batch_size].to(device)
            logits = model(input_ids=batch, use_cache=False).logits[:, :-1, :].float()
            labels = batch[:, 1:]
            per_token = functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                reduction="none",
            ).reshape(batch.shape[0], -1)
            values.extend(float(value) for value in per_token.mean(dim=1))
            del batch, logits, labels, per_token
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (sequences.shape[0],) or not np.isfinite(result).all():
        raise ValueError("Nonfinite or missing per-window NLL values.")
    return result


def _evaluate_architecture(
    model: nn.Module,
    sequences: Tensor,
    architecture: Architecture,
    device: torch.device,
) -> np.ndarray:
    import torch

    from paper5.experiments.canary.direction_01.smoke.core import whole_block_skip

    with whole_block_skip(model, architecture):
        result = evaluate_window_nll(model, sequences, 1, device)
    torch.cuda.empty_cache()
    return result


def _check_edit_semantics(
    model: nn.Module,
    example: Tensor,
    device: torch.device,
    parent_nll: float,
) -> dict[str, float | bool]:
    """Check no-edit equality, pair order, determinism, and hook restoration."""
    from paper5.experiments.canary.direction_01.smoke.core import (
        max_repeated_logit_difference,
    )

    no_edit = float(_evaluate_architecture(model, example, (), device)[0])
    pair = (EXPECTED_LAYERS[0], EXPECTED_LAYERS[1])
    forward = float(_evaluate_architecture(model, example, pair, device)[0])
    backward = float(
        _evaluate_architecture(model, example, tuple(reversed(pair)), device)[0]
    )
    restored = float(evaluate_window_nll(model, example, 1, device)[0])
    repeated_logits = max_repeated_logit_difference(model, example, device)
    tolerance = 1e-7
    checks: dict[str, float | bool] = {
        "no_edit_matches_parent": abs(no_edit - parent_nll) <= tolerance,
        "pair_order_independent": abs(forward - backward) <= tolerance,
        "parent_restored": abs(restored - parent_nll) <= tolerance,
        "repeated_no_edit_max_logit_difference": repeated_logits,
        "repeated_no_edit_logits_match": bool(
            np.isfinite(repeated_logits) and repeated_logits <= 1e-5
        ),
    }
    if not all(value for value in checks.values() if isinstance(value, bool)):
        raise ValueError("H1-small intervention safety check failed.")
    return checks


def _write_per_window_csv(
    path: Path,
    windows: dict[str, dict[Architecture, np.ndarray]],
    source_indices: dict[str, list[int]],
) -> None:
    rows: list[dict[str, Any]] = []
    for split, architectures in windows.items():
        for architecture, values in architectures.items():
            for window, nll in enumerate(values):
                rows.append(
                    {
                        "split": split,
                        "first_layer": architecture[0] if architecture else "",
                        "second_layer": (
                            architecture[1] if len(architecture) == 2 else ""
                        ),
                        "sample_index": window,
                        "source_window_index": source_indices[split][window],
                        "nll": float(nll),
                    }
                )
    write_csv(path, rows)


def _write_summary(
    path: Path, metadata: dict[str, Any], metrics: dict[str, Any]
) -> None:
    interaction = metrics["interaction_distribution_nll"]
    baselines = metrics["baseline_metrics"]
    lines = [
        "# Direction 1 H1-small empirical observation",
        "",
        "This is a 12-edit, 66-pair hypothesis-validation run using whole-block structural interventions. It does not evaluate a transport predictor or establish a method claim.",
        "",
        "## Run",
        "",
        f"- GPU: `{metadata['selected_gpu']}` (`{metadata['gpu_name']}`)",
        f"- Runtime: {metadata['runtime_seconds']:.2f} seconds",
        f"- Model: `{metadata['model_id']}` at `{metadata['resolved_revision']}`",
        "- Dataset: raw WikiText-2 train calibration and valid evaluation",
        "- Scope: 12 single interventions and all 66 unordered pairs",
        "",
        "## Validation interaction",
        "",
        f"- Mean absolute interaction: {interaction['mean_absolute']:.6g} NLL",
        f"- Median relative absolute interaction: {interaction['median_relative_absolute']:.6g}",
        f"- Signed residual range: [{interaction['minimum']:.6g}, {interaction['maximum']:.6g}] NLL",
        f"- Signed residual Q1 / median / Q3: {interaction['q25']:.6g} / {interaction['median']:.6g} / {interaction['q75']:.6g} NLL",
        f"- Positive / negative / zero pairs: {interaction['positive_count']} / {interaction['negative_count']} / {interaction['zero_count']}",
        "",
        "## Validation prediction baselines",
        "",
        "| Baseline | MAE (NLL) | RMSE (NLL) | Spearman | Kendall tau-b | Lowest-17 overlap |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in ("additive", "mean_interaction", "linear_regression"):
        values = baselines[name]
        lines.append(
            f"| {name} | {values['mae']:.6g} | {values['rmse']:.6g} | {values['spearman']:.6g} | {values['kendall_tau_b']:.6g} | {values['top_quartile_overlap']:.6g} |"
        )
    lines.extend(
        [
            "",
            "## Protocol interpretation",
            "",
            f"- H1 verdict: `{metrics['h1_verdict']}`",
            f"- Point verdict: `{metrics['protocol_point_verdict']}`",
            f"- Stable confirmation under the prespecified bootstrap rule: `{metrics['stable_confirmation']}`",
            f"- Paired-window 95% interval for median relative residual: {metrics['paired_validation_window_bootstrap_95pct']['median_relative_residual']}",
            f"- Paired-window 95% interval for additive Spearman: {metrics['paired_validation_window_bootstrap_95pct']['spearman']}",
            f"- Paired-window 95% interval for lowest-17 overlap: {metrics['paired_validation_window_bootstrap_95pct']['top_quartile_overlap']}",
            "- All uncertainty intervals and calibration-shard diagnostics are in `metrics.json`.",
            "- This observation covers one model, one corpus, and one intervention family. It cannot establish interaction transport or a final algorithm.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_h1_small(config: dict[str, Any]) -> dict[str, Any]:
    """Execute the approved fixed-size H1-small run on one selected A800."""
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from paper5.experiments.canary.direction_01.smoke.core import (
        locate_decoder_layers,
    )
    from paper5.experiments.canary.direction_01.smoke.run_smoke import (
        dtype_from_name,
        git_commit,
        set_reproducibility,
    )

    started = time.monotonic()
    started_utc = datetime.now(UTC).isoformat()
    set_reproducibility(int(config["random_seed"]))
    device, selected_gpu, gpu_name = _require_a800()
    output_dir = REPOSITORY_ROOT / str(config["output_directory"])
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("H1-small output directory already contains files.")
    cache_dir = H1_CACHE_ROOT
    calibration_text = _checked_text(
        str(config["dataset"]["calibration"]["url"]),
        str(config["dataset"]["calibration"]["sha256"]),
        cache_dir / "wikitext-2-train.txt",
    )
    validation_text = _checked_text(
        str(config["dataset"]["validation"]["url"]),
        str(config["dataset"]["validation"]["sha256"]),
        cache_dir / "wikitext-2-valid.txt",
    )
    revision = str(config["model"]["revision"])
    model_id = str(config["model"]["id"])
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, revision=revision, cache_dir=cache_dir / "huggingface"
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        cache_dir=cache_dir / "huggingface",
        dtype=dtype_from_name(str(config["model"]["dtype"])),
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    resolved_revision = getattr(model.config, "_commit_hash", None) or revision
    if resolved_revision != revision:
        raise ValueError("Loaded model revision differs from the frozen H1 config.")
    if len(locate_decoder_layers(model)) != 28:
        raise ValueError("The H1 model does not expose the expected 28 decoder blocks.")
    seed = int(config["random_seed"])
    calibration, calibration_indices = _prepare_sequences_with_indices(
        calibration_text, tokenizer, 512, 96, seed
    )
    validation, validation_indices = _prepare_sequences_with_indices(
        validation_text, tokenizer, 512, 64, seed + 1
    )
    sequences = {"calibration": calibration, "validation": validation}
    source_indices = {
        "calibration": calibration_indices,
        "validation": validation_indices,
    }
    architectures: list[Architecture] = [
        (),
        *((layer,) for layer in EXPECTED_LAYERS),
        *itertools.combinations(EXPECTED_LAYERS, 2),
    ]
    initial_parent_nll = float(
        evaluate_window_nll(model, calibration[:1], 1, device)[0]
    )
    checks = _check_edit_semantics(model, calibration[:1], device, initial_parent_nll)
    evaluation_started = time.monotonic()
    windows: dict[str, dict[Architecture, np.ndarray]] = {}
    for split, tokens in sequences.items():
        windows[split] = {}
        for index, architecture in enumerate(architectures, 1):
            windows[split][architecture] = _evaluate_architecture(
                model, tokens, architecture, device
            )
            if time.monotonic() - evaluation_started > 4 * 3600:
                raise RuntimeError("H1-small evaluation exceeded its four-hour budget.")
            if index % 10 == 0 or index == len(architectures):
                print(
                    json.dumps(
                        {"split": split, "architectures_completed": index, "total": 79}
                    ),
                    flush=True,
                )
    final_parent_nll = float(evaluate_window_nll(model, calibration[:1], 1, device)[0])
    checks["parent_restored_after_all_edits"] = (
        abs(final_parent_nll - initial_parent_nll) <= 1e-7
    )
    if not checks["parent_restored_after_all_edits"]:
        raise ValueError("The parent model changed during H1-small evaluation.")
    rows, metrics = analyze_h1_small(
        windows["calibration"],
        windows["validation"],
        EXPECTED_LAYERS,
        3,
        int(config["variance"]["bootstrap_replicates"]),
        seed,
    )
    runtime_seconds = time.monotonic() - started
    evaluation_runtime_seconds = time.monotonic() - evaluation_started
    metadata = {
        "scientific_scope": "H1-small empirical observation",
        "started_at_utc": started_utc,
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "started_from_commit": git_commit(),
        "model_id": model_id,
        "requested_revision": revision,
        "resolved_revision": resolved_revision,
        "tokenizer_id": model_id,
        "selected_gpu": selected_gpu,
        "gpu_name": gpu_name,
        "gpu_memory_gib": torch.cuda.get_device_properties(device).total_memory
        / (1024**3),
        "peak_allocated_gpu_memory_mib": torch.cuda.max_memory_allocated(device)
        / (1024**2),
        "runtime_seconds": runtime_seconds,
        "evaluation_runtime_seconds": evaluation_runtime_seconds,
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "transformers_version": str(transformers.__version__),
        "numpy_version": str(np.__version__),
        "pyyaml_version": str(yaml.__version__),
        "cuda_runtime_version": torch.version.cuda,
        "code_sha256": {
            "runner": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "edit_core": hashlib.sha256(
                Path(__file__).parent.joinpath("smoke/core.py").read_bytes()
            ).hexdigest(),
        },
        "selected_source_window_indices": source_indices,
        "source_sha256": {
            "calibration": config["dataset"]["calibration"]["sha256"],
            "validation": config["dataset"]["validation"]["sha256"],
        },
        "checks": checks,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "results.csv", rows)
    _write_per_window_csv(output_dir / "per_window_nll.csv", windows, source_indices)
    write_json(output_dir / "metrics.json", {"metadata": metadata, "analysis": metrics})
    write_interaction_heatmap(
        output_dir / "interaction_heatmap.svg", rows, EXPECTED_LAYERS
    )
    _write_summary(output_dir / "summary.md", metadata, metrics)
    config_used = dict(config)
    config_used["run_metadata"] = metadata
    (output_dir / "config_used.yaml").write_text(
        yaml.safe_dump(config_used, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return {"metadata": metadata, "analysis": metrics}


def main() -> None:
    """Run H1-small and print a compact completion record."""
    if Path.cwd().resolve() != REPOSITORY_ROOT:
        raise RuntimeError("Launch H1-small from the repository root.")
    args = parse_args()
    config = load_h1_config(args.config)
    result = run_h1_small(config)
    print(
        json.dumps(
            {
                "status": "H1-small empirical observation complete",
                "selected_gpu": result["metadata"]["selected_gpu"],
                "runtime_seconds": result["metadata"]["runtime_seconds"],
                "h1_verdict": result["analysis"]["h1_verdict"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
