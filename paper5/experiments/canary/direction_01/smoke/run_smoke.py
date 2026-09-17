#!/usr/bin/env python3
"""Run the RTX 3050 Direction 1 H1 pipeline smoke test."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from paper5.experiments.canary.direction_01.smoke.analysis import (
    summarize_pairs,
    write_csv,
    write_interaction_svg,
    write_json,
    write_summary,
)
from paper5.experiments.canary.direction_01.smoke.core import (
    NextTokenMetrics,
    clear_device_cache,
    evaluate_next_token_metrics,
    finite_metrics,
    locate_decoder_layers,
    max_repeated_logit_difference,
    prepare_sequences,
    sample_layer_pairs,
    select_evenly_spaced_layers,
    whole_block_skip,
)


def parse_args() -> argparse.Namespace:
    """Parse smoke-test command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    """Load and validate the explicit smoke-test configuration."""
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "run_label",
        "model_id",
        "revision",
        "dtype",
        "seed",
        "sequence_length",
        "calibration_sequences",
        "validation_sequences",
        "batch_size",
        "edit_family",
        "edit_count",
        "pair_count",
        "edge_exclusion",
        "calibration_data_url",
        "validation_data_url",
    }
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"Missing required config fields: {missing}")
    if config["run_label"] != "SMOKE TEST / PRELIMINARY":
        raise ValueError("Smoke runs must use the exact non-scientific run label.")
    if config["edit_family"] != "whole_transformer_block_skip":
        raise ValueError("This smoke test supports exactly one frozen edit family.")
    return config


def set_reproducibility(seed: int) -> None:
    """Set deterministic seeds for all libraries used by the smoke test."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def download_text(url: str, destination: Path) -> tuple[str, str]:
    """Download one public text file into the repository-local cache."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        with urllib.request.urlopen(url, timeout=120) as response:  # noqa: S310
            destination.write_bytes(response.read())
    content = destination.read_bytes()
    return content.decode("utf-8"), hashlib.sha256(content).hexdigest()


def git_commit() -> str:
    """Return the current Git commit without changing repository state."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "UNVERIFIED"


def dtype_from_name(name: str) -> torch.dtype:
    """Resolve the small allowlist of inference dtypes."""
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[name]


def metric_row(
    run_label: str,
    split: str,
    layer: int,
    parent: NextTokenMetrics,
    edited: NextTokenMetrics,
) -> dict[str, Any]:
    """Build one complete single-edit CSV row."""
    return {
        "run_label": run_label,
        "split": split,
        "edit_family": "whole_transformer_block_skip",
        "layer": layer,
        "parent_nll": parent.nll,
        "edited_nll": edited.nll,
        "delta_nll": edited.nll - parent.nll,
        "parent_perplexity": parent.perplexity,
        "edited_perplexity": edited.perplexity,
        "delta_perplexity": edited.perplexity - parent.perplexity,
        "token_count": edited.token_count,
    }


def pair_row(
    run_label: str,
    split: str,
    first: int,
    second: int,
    parent: NextTokenMetrics,
    edited: NextTokenMetrics,
    first_delta: float,
    second_delta: float,
) -> dict[str, Any]:
    """Build one complete pair-edit CSV row with additive residuals."""
    actual = edited.nll - parent.nll
    additive = first_delta + second_delta
    interaction = actual - additive
    denominator = max(abs(first_delta) + abs(second_delta), 1e-12)
    return {
        "run_label": run_label,
        "split": split,
        "edit_family": "whole_transformer_block_skip",
        "first_layer": first,
        "second_layer": second,
        "layer_distance": second - first,
        "first_delta_nll": first_delta,
        "second_delta_nll": second_delta,
        "additive_delta_nll": additive,
        "actual_delta_nll": actual,
        "interaction_nll": interaction,
        "relative_abs_interaction": abs(interaction) / denominator,
        "parent_perplexity": parent.perplexity,
        "edited_perplexity": edited.perplexity,
        "delta_perplexity": edited.perplexity - parent.perplexity,
        "token_count": edited.token_count,
    }


def evaluate_with_edits(
    model: torch.nn.Module,
    sequences: torch.Tensor,
    layer_indices: list[int],
    batch_size: int,
    device: torch.device,
) -> NextTokenMetrics:
    """Evaluate one static architecture and restore the parent afterwards."""
    with whole_block_skip(model, layer_indices):
        metrics = evaluate_next_token_metrics(
            model=model,
            sequences=sequences,
            batch_size=batch_size,
            device=device,
        )
    clear_device_cache(device)
    return metrics


def run_smoke(config: dict[str, Any], output_dir: Path, cache_dir: Path) -> None:
    """Execute the complete local H1 pipeline smoke test."""
    start_time = time.monotonic()
    seed = int(config["seed"])
    set_reproducibility(seed)
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable; refusing to fabricate a GPU smoke result."
        )
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)

    cache_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_dir / "huggingface")
    calibration_text, calibration_sha = download_text(
        str(config["calibration_data_url"]), cache_dir / "wikitext-2-train.txt"
    )
    validation_text, validation_sha = download_text(
        str(config["validation_data_url"]), cache_dir / "wikitext-2-valid.txt"
    )

    dtype = dtype_from_name(str(config["dtype"]))
    tokenizer = AutoTokenizer.from_pretrained(
        str(config["model_id"]),
        revision=str(config["revision"]),
        cache_dir=cache_dir / "huggingface",
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(config["model_id"]),
        revision=str(config["revision"]),
        cache_dir=cache_dir / "huggingface",
        dtype=dtype,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()

    calibration = prepare_sequences(
        calibration_text,
        tokenizer,
        int(config["sequence_length"]),
        int(config["calibration_sequences"]),
        seed,
    )
    validation = prepare_sequences(
        validation_text,
        tokenizer,
        int(config["sequence_length"]),
        int(config["validation_sequences"]),
        seed + 1,
    )
    splits = {"calibration": calibration, "validation": validation}
    block_count = len(locate_decoder_layers(model))
    edit_layers = select_evenly_spaced_layers(
        block_count,
        int(config["edit_count"]),
        int(config["edge_exclusion"]),
    )
    pairs = sample_layer_pairs(edit_layers, int(config["pair_count"]), seed)
    batch_size = int(config["batch_size"])

    baselines: dict[str, NextTokenMetrics] = {}
    no_edit_context_metrics: dict[str, NextTokenMetrics] = {}
    for split, sequences in splits.items():
        baselines[split] = evaluate_next_token_metrics(
            model, sequences, batch_size, device
        )
        no_edit_context_metrics[split] = evaluate_with_edits(
            model, sequences, [], batch_size, device
        )

    no_edit_logit_difference = max_repeated_logit_difference(
        model, calibration[:1], device
    )
    clear_device_cache(device)
    baseline_allocated_mib = torch.cuda.memory_allocated(device) / (1024**2)
    single_rows: list[dict[str, Any]] = []
    single_lookup: dict[tuple[str, int], float] = {}
    for split, sequences in splits.items():
        for layer in edit_layers:
            edited = evaluate_with_edits(model, sequences, [layer], batch_size, device)
            row = metric_row(
                str(config["run_label"]), split, layer, baselines[split], edited
            )
            single_rows.append(row)
            single_lookup[(split, layer)] = float(row["delta_nll"])

    pair_rows: list[dict[str, Any]] = []
    for split, sequences in splits.items():
        for first, second in pairs:
            edited = evaluate_with_edits(
                model, sequences, [first, second], batch_size, device
            )
            pair_rows.append(
                pair_row(
                    str(config["run_label"]),
                    split,
                    first,
                    second,
                    baselines[split],
                    edited,
                    single_lookup[(split, first)],
                    single_lookup[(split, second)],
                )
            )

    first_pair = pairs[0]
    ordered = evaluate_with_edits(
        model, calibration, list(first_pair), batch_size, device
    )
    reversed_order = evaluate_with_edits(
        model, calibration, list(reversed(first_pair)), batch_size, device
    )
    repeated_single = evaluate_with_edits(
        model, calibration, [edit_layers[0]], batch_size, device
    )
    restored = evaluate_next_token_metrics(model, calibration, batch_size, device)
    clear_device_cache(device)
    final_allocated_mib = torch.cuda.memory_allocated(device) / (1024**2)

    tolerance = float(config.get("reproducibility_tolerance", 1e-8))
    memory_tolerance_mib = float(config.get("memory_release_tolerance_mib", 16.0))
    no_edit_delta = max(
        abs(baselines[split].nll - no_edit_context_metrics[split].nll)
        for split in splits
    )
    original_single = next(
        row
        for row in single_rows
        if row["split"] == "calibration" and row["layer"] == edit_layers[0]
    )
    checks = {
        "cuda_available": True,
        "no_edit_context_matches_parent": no_edit_delta <= tolerance,
        "no_edit_max_nll_difference": no_edit_delta,
        "repeated_no_edit_max_logit_abs_difference": no_edit_logit_difference,
        "single_edit_restored_parent": abs(restored.nll - baselines["calibration"].nll)
        <= tolerance,
        "pair_order_independent": abs(ordered.nll - reversed_order.nll) <= tolerance,
        "same_seed_single_reproducible": abs(
            repeated_single.nll - float(original_single["edited_nll"])
        )
        <= tolerance,
        "candidate_memory_released": final_allocated_mib
        <= baseline_allocated_mib + memory_tolerance_mib,
        "baseline_allocated_memory_mib": baseline_allocated_mib,
        "final_allocated_memory_mib": final_allocated_mib,
        "all_metrics_finite": all(
            math.isfinite(float(row[key]))
            for row in single_rows + pair_rows
            for key in row
            if key
            not in {
                "run_label",
                "split",
                "edit_family",
            }
        )
        and all(finite_metrics(value) for value in baselines.values()),
        "h2_real_model_runtime": "DEFERRED_TO_A800",
    }
    required_boolean_checks = [
        value for value in checks.values() if isinstance(value, bool)
    ]
    pipeline_status = (
        "PIPELINE READY" if all(required_boolean_checks) else "PIPELINE NOT READY"
    )
    analysis = {
        split: summarize_pairs([row for row in pair_rows if row["split"] == split])
        for split in splits
    }
    runtime_seconds = time.monotonic() - start_time
    peak_mib = torch.cuda.max_memory_allocated(device) / (1024**2)
    resolved_revision = getattr(model.config, "_commit_hash", None) or str(
        config["revision"]
    )
    metadata = {
        "run_label": config["run_label"],
        "started_from_commit": git_commit(),
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "model_id": config["model_id"],
        "requested_revision": config["revision"],
        "resolved_revision": resolved_revision,
        "gpu_name": torch.cuda.get_device_name(device),
        "gpu_total_memory_mib": torch.cuda.get_device_properties(device).total_memory
        / (1024**2),
        "peak_gpu_memory_mib": peak_mib,
        "baseline_allocated_memory_mib": baseline_allocated_mib,
        "final_allocated_memory_mib": final_allocated_mib,
        "runtime_seconds": runtime_seconds,
        "gpu_hours": runtime_seconds / 3600.0,
        "command": " ".join(
            [
                "PYTHONPATH=. python -m paper5.experiments.canary.direction_01.smoke.run_smoke",
                *sys.argv[1:],
            ]
        ),
        "edit_family": config["edit_family"],
        "edit_layers": edit_layers,
        "single_edit_count": len(edit_layers),
        "pair_count": len(pairs),
        "sequence_length": config["sequence_length"],
        "calibration_sequences": config["calibration_sequences"],
        "validation_sequences": config["validation_sequences"],
        "batch_size": config["batch_size"],
        "seed": seed,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_runtime": torch.version.cuda,
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "calibration_data_sha256": calibration_sha,
        "validation_data_sha256": validation_sha,
    }
    payload = {
        "schema_version": 1,
        "scientific_status": "SMOKE TEST / PRELIMINARY",
        "formal_h1_verdict": "NOT_EVALUATED",
        "formal_h2_verdict": "NOT_RUN",
        "pipeline_status": pipeline_status,
        "oom": False,
        "metadata": metadata,
        "config": config,
        "checks": checks,
        "baselines": {
            split: {
                "nll": value.nll,
                "perplexity": value.perplexity,
                "token_count": value.token_count,
            }
            for split, value in baselines.items()
        },
        "smoke_only_analysis": analysis,
        "single_edits": single_rows,
        "pair_edits": pair_rows,
    }
    write_csv(output_dir / "smoke_single_edit.csv", single_rows)
    write_csv(output_dir / "smoke_pair_edit.csv", pair_rows)
    write_json(output_dir / "smoke_metrics.json", payload)
    write_summary(
        output_dir / "smoke_summary.md",
        metadata,
        checks,
        analysis,
        pipeline_status,
    )
    write_interaction_svg(
        output_dir / "smoke_additive_scatter.svg",
        [row for row in pair_rows if row["split"] == "validation"],
        "SMOKE TEST / PRELIMINARY — validation",
    )
    print(
        json.dumps(
            {
                "pipeline_status": pipeline_status,
                "peak_gpu_memory_mib": peak_mib,
                "runtime_seconds": runtime_seconds,
            },
            indent=2,
            sort_keys=True,
        )
    )


def write_failure(output_dir: Path, config: dict[str, Any], error: Exception) -> None:
    """Record a transparent smoke failure without fabricating measurements."""
    output_dir.mkdir(parents=True, exist_ok=True)
    is_oom = (
        isinstance(error, torch.cuda.OutOfMemoryError)
        or "out of memory" in str(error).lower()
    )
    payload = {
        "schema_version": 1,
        "scientific_status": "SMOKE TEST / PRELIMINARY",
        "formal_h1_verdict": "NOT_EVALUATED",
        "formal_h2_verdict": "NOT_RUN",
        "pipeline_status": "PIPELINE NOT READY",
        "oom": is_oom,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "config": config,
    }
    write_json(output_dir / "smoke_metrics.json", payload)
    summary = "\n".join(
        [
            "# Direction 1 Local Smoke Test",
            "",
            "> **THIS IS NOT A FORMAL SCIENTIFIC CANARY RESULT.**",
            "",
            "## Pipeline status: PIPELINE NOT READY",
            "",
            f"- OOM: `{is_oom}`",
            f"- Error type: `{type(error).__name__}`",
            "- No scientific result was generated.",
            "- H2 runtime validation deferred to A800.",
        ]
    )
    (output_dir / "smoke_summary.md").write_text(summary + "\n", encoding="utf-8")


def main() -> None:
    """CLI entry point with explicit OOM/failure artifacts."""
    args = parse_args()
    config = load_config(args.config)
    try:
        run_smoke(config, args.output_dir, args.cache_dir)
    except Exception as error:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        write_failure(args.output_dir, config, error)
        raise


if __name__ == "__main__":
    main()
