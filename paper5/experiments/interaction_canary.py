#!/usr/bin/env python3
"""Measure non-additive quality interactions between post-training layer removals."""

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
import urllib.request
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import Any, Iterator, Sequence

import torch
import torch.nn.functional as functional
import transformers
from torch import Tensor, nn
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    return parser.parse_args()


def sha256_bytes(content: bytes) -> str:
    """Return the hexadecimal SHA-256 digest for bytes."""
    return hashlib.sha256(content).hexdigest()


def download_text(url: str, destination: Path) -> tuple[str, str]:
    """Download a UTF-8 text file once and return its content and digest."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        content = destination.read_bytes()
    else:
        with urllib.request.urlopen(url, timeout=120) as response:  # noqa: S310
            content = response.read()
        destination.write_bytes(content)
    return content.decode("utf-8"), sha256_bytes(content)


def prepare_sequences(
    text: str,
    tokenizer: Any,
    sequence_length: int,
    max_tokens: int,
) -> Tensor:
    """Tokenize text into contiguous, fixed-length evaluation sequences."""
    token_ids = tokenizer(
        text, add_special_tokens=False, return_tensors="pt"
    ).input_ids[0]
    usable = min(max_tokens, token_ids.numel())
    usable -= usable % sequence_length
    if usable < sequence_length:
        raise ValueError("The dataset produced fewer than one complete sequence.")
    return token_ids[:usable].reshape(-1, sequence_length)


def _identity_block_hook(
    _module: nn.Module, inputs: tuple[Any, ...], output: Any
) -> Any:
    """Replace a Transformer block output by its residual-stream input."""
    hidden_states = inputs[0]
    if isinstance(output, tuple):
        return (hidden_states, *output[1:])
    return hidden_states


@contextmanager
def bypass_layers(model: nn.Module, layer_indices: Sequence[int]) -> Iterator[None]:
    """Temporarily bypass GPT-style residual blocks by index."""
    blocks = getattr(getattr(model, "transformer"), "h")
    handles = [
        blocks[index].register_forward_hook(_identity_block_hook)
        for index in layer_indices
    ]
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


@torch.inference_mode()
def evaluate_nll(
    model: nn.Module,
    sequences: Tensor,
    batch_size: int,
    device: torch.device,
) -> float:
    """Compute token-weighted causal negative log likelihood."""
    total_loss = 0.0
    total_tokens = 0
    for start in range(0, sequences.shape[0], batch_size):
        batch = sequences[start : start + batch_size].to(device)
        logits = model(input_ids=batch, use_cache=False).logits[:, :-1, :]
        labels = batch[:, 1:]
        loss_sum = functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            reduction="sum",
        )
        total_loss += float(loss_sum)
        total_tokens += labels.numel()
    return total_loss / total_tokens


def average_ranks(values: Sequence[float]) -> list[float]:
    """Compute one-based average ranks, including ties."""
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(indexed):
        end = cursor + 1
        while end < len(indexed) and indexed[end][1] == indexed[cursor][1]:
            end += 1
        rank = (cursor + 1 + end) / 2.0
        for index, _ in indexed[cursor:end]:
            ranks[index] = rank
        cursor = end
    return ranks


def pearson_correlation(left: Sequence[float], right: Sequence[float]) -> float:
    """Compute Pearson correlation without a third-party statistics package."""
    if len(left) != len(right) or not left:
        raise ValueError("Correlation inputs must be non-empty and equally sized.")
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    left_scale = math.sqrt(sum((x - left_mean) ** 2 for x in left))
    right_scale = math.sqrt(sum((y - right_mean) ** 2 for y in right))
    if left_scale == 0.0 or right_scale == 0.0:
        return 0.0
    return numerator / (left_scale * right_scale)


def spearman_correlation(left: Sequence[float], right: Sequence[float]) -> float:
    """Compute Spearman rank correlation with average tie handling."""
    return pearson_correlation(average_ranks(left), average_ranks(right))


def classify_signal(
    median_interaction_ratio: float,
    spearman: float,
    top_quartile_overlap: float,
) -> str:
    """Apply the preregistered scientific-decision thresholds."""
    confirmed_tests = (
        median_interaction_ratio >= 0.10,
        spearman <= 0.90,
        top_quartile_overlap <= 0.75,
    )
    if all(confirmed_tests):
        return "A: Problem confirmed"
    if sum(confirmed_tests) >= 2:
        return "B: Weak signal"
    if (
        median_interaction_ratio < 0.03
        and spearman >= 0.95
        and top_quartile_overlap >= 0.90
    ):
        return "C: Problem not confirmed"
    return "D: Experiment inconclusive"


def git_commit() -> str:
    """Return the repository HEAD, or UNVERIFIED when Git is unavailable."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "UNVERIFIED"


def hardware_record(device: torch.device) -> dict[str, Any]:
    """Collect a compact hardware and runtime record."""
    record: dict[str, Any] = {
        "device": str(device),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "processor": platform.processor() or "UNVERIFIED",
        "python": platform.python_version(),
        "torch_threads": torch.get_num_threads(),
    }
    if device.type == "cuda":
        record["gpu_name"] = torch.cuda.get_device_name(device)
        record["cuda_version"] = torch.version.cuda
    return record


def main() -> None:
    """Run all single- and pair-removal evaluations and save raw evidence."""
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    seed = int(config["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(int(config["torch_threads"]))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(args.cache_dir / "huggingface"))
    data_path = args.cache_dir / "wikitext-2-test.txt"
    text, data_sha256 = download_text(config["data_url"], data_path)
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_id"],
        revision=config["revision"],
        cache_dir=args.cache_dir / "huggingface",
    )
    model = AutoModelForCausalLM.from_pretrained(
        config["model_id"],
        revision=config["revision"],
        cache_dir=args.cache_dir / "huggingface",
    ).to(device)
    model.eval()
    sequences = prepare_sequences(
        text,
        tokenizer,
        int(config["sequence_length"]),
        int(config["max_tokens"]),
    )

    baseline_nll = evaluate_nll(model, sequences, int(config["batch_size"]), device)
    block_count = len(getattr(getattr(model, "transformer"), "h"))
    singles: list[dict[str, Any]] = []
    for layer in range(block_count):
        with bypass_layers(model, [layer]):
            nll = evaluate_nll(model, sequences, int(config["batch_size"]), device)
        singles.append({"layer": layer, "nll": nll, "delta_nll": nll - baseline_nll})

    pairs: list[dict[str, Any]] = []
    for first in range(block_count):
        for second in range(first + 1, block_count):
            with bypass_layers(model, [first, second]):
                nll = evaluate_nll(model, sequences, int(config["batch_size"]), device)
            first_delta = float(singles[first]["delta_nll"])
            second_delta = float(singles[second]["delta_nll"])
            actual_delta = nll - baseline_nll
            additive_delta = first_delta + second_delta
            interaction = actual_delta - additive_delta
            denominator = max(abs(first_delta) + abs(second_delta), 1e-12)
            pairs.append(
                {
                    "layers": [first, second],
                    "nll": nll,
                    "actual_delta_nll": actual_delta,
                    "additive_delta_nll": additive_delta,
                    "interaction_delta_nll": interaction,
                    "absolute_interaction_ratio": abs(interaction) / denominator,
                }
            )

    predicted = [float(item["additive_delta_nll"]) for item in pairs]
    actual = [float(item["actual_delta_nll"]) for item in pairs]
    ratios = [float(item["absolute_interaction_ratio"]) for item in pairs]
    spearman = spearman_correlation(predicted, actual)
    quartile_size = max(1, math.ceil(len(pairs) / 4))
    predicted_best = set(
        sorted(range(len(pairs)), key=predicted.__getitem__)[:quartile_size]
    )
    actual_best = set(sorted(range(len(pairs)), key=actual.__getitem__)[:quartile_size])
    overlap = len(predicted_best & actual_best) / quartile_size
    median_ratio = median(ratios)
    decision = classify_signal(median_ratio, spearman, overlap)

    result = {
        "schema_version": 1,
        "started_from_commit": git_commit(),
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "command": " ".join(sys.argv),
        "config": config,
        "dataset": {
            "url": config["data_url"],
            "sha256": data_sha256,
            "sequence_count": int(sequences.shape[0]),
            "sequence_length": int(sequences.shape[1]),
        },
        "model": {
            "id": config["model_id"],
            "requested_revision": config["revision"],
            "resolved_commit": getattr(model.config, "_commit_hash", None)
            or "UNVERIFIED",
            "block_count": block_count,
        },
        "environment": {
            **hardware_record(device),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "baseline": {
            "nll": baseline_nll,
            "perplexity": math.exp(baseline_nll),
        },
        "single_layer_removals": singles,
        "pair_layer_removals": pairs,
        "summary": {
            "median_absolute_interaction_ratio": median_ratio,
            "spearman_additive_vs_actual": spearman,
            "top_quartile_overlap": overlap,
            "decision": decision,
        },
        "preregistered_thresholds": {
            "confirmed": {
                "median_absolute_interaction_ratio_min": 0.10,
                "spearman_max": 0.90,
                "top_quartile_overlap_max": 0.75,
                "rule": "all three",
            },
            "weak_signal": {"rule": "any two confirmed conditions"},
            "not_confirmed": {
                "median_absolute_interaction_ratio_max_exclusive": 0.03,
                "spearman_min": 0.95,
                "top_quartile_overlap_min": 0.90,
                "rule": "all three",
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
