"""Run the bounded Direction 1 H1 pipeline sanity check on one A800."""

from __future__ import annotations

import argparse
import csv
import math
import os
import platform
import time
from pathlib import Path
from typing import Any

import torch
import transformers
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from paper5.experiments.canary.direction_01.smoke.core import (
    NextTokenMetrics,
    evaluate_next_token_metrics,
    finite_metrics,
    locate_decoder_layers,
    prepare_sequences,
)
from paper5.experiments.canary.direction_01.smoke.run_smoke import (
    download_text,
    dtype_from_name,
    evaluate_with_edits,
    git_commit,
    set_reproducibility,
)

CONFIG_PATH = Path("paper5/configs/canary/direction_01/h1_sanity.yaml")
RESULT_FIELDS = (
    "edit_type",
    "first_layer",
    "second_layer",
    "nll",
    "perplexity",
    "token_count",
    "delta_nll",
    "additive_prediction_nll",
    "interaction_residual_nll",
)


def parse_args() -> argparse.Namespace:
    """Read the fixed sanity configuration path from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    """Reject configurations that expand this run beyond the approved scope."""
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("The sanity configuration must be a YAML mapping.")
    if config.get("status") != "pipeline_sanity_only":
        raise ValueError("Only the pipeline sanity scope is allowed.")
    model = config["model"]
    dataset = config["dataset"]
    edit = config["edit"]
    if (
        model["id"] != "Qwen/Qwen2.5-1.5B"
        or model["revision"] != "8faed761d45a263340a0528343f099c05c9a4323"
        or model["dtype"] != "bfloat16"
    ):
        raise ValueError("The sanity model must be Qwen2.5-1.5B in bfloat16.")
    if (
        dataset["name"] != "WikiText-2 raw"
        or dataset["validation"]["url"]
        != "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/valid.txt"
        or dataset["validation"]["sha256"]
        != "f0737ed31fc1329026e95cb8b98e19c2a182c39c240ab909dc31abf2f8af58e8"
    ):
        raise ValueError("The sanity dataset must be WikiText-2 raw.")
    if edit["family"] != "whole_transformer_block_skip":
        raise ValueError("The sanity intervention must skip whole blocks.")
    if edit["layer_indices"] != [1, 3]:
        raise ValueError("The sanity run is limited to block positions 1 and 3.")
    if not 1 <= int(config["validation_sequences"]) <= 2:
        raise ValueError("The sanity run is limited to two validation windows.")
    if not 2 <= int(config["sequence_length"]) <= 128:
        raise ValueError("The sanity run is limited to 128-token windows.")
    if int(config["batch_size"]) != 1:
        raise ValueError("The sanity batch size must be one.")
    if config["output_directory"] != str(
        Path("paper5/results/canary/direction_01/H1_sanity")
    ):
        raise ValueError("The sanity output directory is fixed.")
    if config["cache_directory"] != "paper5/.cache/h1_sanity":
        raise ValueError("The sanity cache directory is fixed.")
    return config


def metric_row(
    edit_type: str,
    first_layer: int | None,
    second_layer: int | None,
    metrics: NextTokenMetrics,
    parent_nll: float,
    additive_prediction: float | None = None,
) -> dict[str, str | int | float]:
    """Build one architecture result without drawing a scientific conclusion."""
    delta = metrics.nll - parent_nll
    return {
        "edit_type": edit_type,
        "first_layer": "" if first_layer is None else first_layer,
        "second_layer": "" if second_layer is None else second_layer,
        "nll": metrics.nll,
        "perplexity": metrics.perplexity,
        "token_count": metrics.token_count,
        "delta_nll": delta,
        "additive_prediction_nll": (
            "" if additive_prediction is None else additive_prediction
        ),
        "interaction_residual_nll": (
            "" if additive_prediction is None else delta - additive_prediction
        ),
    }


def run(config: dict[str, Any]) -> None:
    """Load inputs, evaluate one parent, two singles and one pair, then save."""
    started = time.monotonic()
    if os.environ.get("CONDA_DEFAULT_ENV") != "autoresearch_paper5":
        raise RuntimeError("Activate autoresearch_paper5 before the sanity run.")
    selected_gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not selected_gpu.isdecimal() or not torch.cuda.is_available():
        raise RuntimeError("Select one available CUDA GPU before the sanity run.")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("The sanity run must see exactly one GPU.")
    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(device)
    gpu_memory_gib = torch.cuda.get_device_properties(device).total_memory / 1024**3
    if "A800" not in gpu_name or gpu_memory_gib < 75:
        raise RuntimeError(f"Expected an A800 80GB GPU, got {gpu_name}.")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    set_reproducibility(int(config["random_seed"]))

    cache_dir = Path(config["cache_directory"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_dir / "huggingface")
    validation = config["dataset"]["validation"]
    text, source_hash = download_text(
        str(validation["url"]), cache_dir / "wikitext-2-valid.txt"
    )
    if source_hash != validation["sha256"]:
        raise ValueError("WikiText-2 validation source SHA-256 mismatch.")

    model_config = config["model"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["id"],
        revision=model_config["revision"],
        cache_dir=cache_dir / "huggingface",
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_config["id"],
        revision=model_config["revision"],
        cache_dir=cache_dir / "huggingface",
        dtype=dtype_from_name(model_config["dtype"]),
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    if len(locate_decoder_layers(model)) != 28:
        raise RuntimeError("The pinned model does not expose 28 decoder blocks.")
    sequences = prepare_sequences(
        text,
        tokenizer,
        int(config["sequence_length"]),
        int(config["validation_sequences"]),
        int(config["random_seed"]),
    )
    batch_size = int(config["batch_size"])
    parent = evaluate_next_token_metrics(model, sequences, batch_size, device)
    no_edit = evaluate_with_edits(model, sequences, [], batch_size, device)
    first_layer, second_layer = config["edit"]["layer_indices"]
    first = evaluate_with_edits(model, sequences, [first_layer], batch_size, device)
    second = evaluate_with_edits(model, sequences, [second_layer], batch_size, device)
    pair = evaluate_with_edits(
        model, sequences, [first_layer, second_layer], batch_size, device
    )
    restored = evaluate_next_token_metrics(model, sequences, batch_size, device)
    if not all(
        finite_metrics(item)
        for item in (parent, no_edit, first, second, pair, restored)
    ):
        raise RuntimeError("At least one evaluation metric is not finite.")
    if max(abs(parent.nll - no_edit.nll), abs(parent.nll - restored.nll)) > 1e-6:
        raise RuntimeError("No-edit equivalence or hook restoration failed.")

    additive = (first.nll - parent.nll) + (second.nll - parent.nll)
    rows = [
        metric_row("parent", None, None, parent, parent.nll),
        metric_row("single", first_layer, None, first, parent.nll),
        metric_row("single", second_layer, None, second, parent.nll),
        metric_row("pair", first_layer, second_layer, pair, parent.nll, additive),
    ]
    if not math.isfinite(float(rows[-1]["interaction_residual_nll"])):
        raise RuntimeError("The interaction residual is not finite.")

    output_dir = Path(config["output_directory"])
    runtime = time.monotonic() - started
    peak_memory_mib = torch.cuda.max_memory_allocated(device) / 1024**2
    provenance = {
        **config,
        "actual_run": {
            "git_commit": git_commit(),
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "transformers": transformers.__version__,
            "cuda_runtime": torch.version.cuda,
            "source_sha256": source_hash,
            "resolved_model_revision": getattr(model.config, "_commit_hash", None)
            or model_config["revision"],
            "selected_gpu": selected_gpu,
            "cuda_visible_devices": selected_gpu,
            "gpu_name": gpu_name,
            "runtime_seconds": runtime,
            "peak_gpu_memory_mib": peak_memory_mib,
            "pipeline_success": True,
        },
    }
    config_text = yaml.safe_dump(provenance, sort_keys=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "config_used.yaml").write_text(config_text, encoding="utf-8")
    (output_dir / "summary.md").write_text(
        "\n".join(
            [
                "# H1 sanity pipeline",
                "",
                f"- Model: {model_config['id']}",
                f"- Dataset: {config['dataset']['name']} validation",
                f"- GPU: {selected_gpu} ({gpu_name})",
                f"- Runtime: {runtime:.2f} s",
                f"- Peak GPU memory: {peak_memory_mib:.2f} MiB",
                "- Results generated successfully: yes",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    """Execute the fixed, command-line reproducible sanity run."""
    args = parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()
