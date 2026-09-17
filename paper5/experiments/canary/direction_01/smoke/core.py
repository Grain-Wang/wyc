"""Core edit, data, and metric operations for the Direction 1 smoke test."""

from __future__ import annotations

import itertools
import math
import random
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Sequence

import torch
import torch.nn.functional as functional
from torch import Tensor, nn


@dataclass(frozen=True)
class NextTokenMetrics:
    """Token-weighted next-token evaluation metrics."""

    nll: float
    perplexity: float
    token_count: int


def locate_decoder_layers(model: nn.Module) -> Sequence[nn.Module]:
    """Return the decoder layer sequence for supported Hugging Face causal LMs."""
    base_model = getattr(model, "model", None)
    layers = getattr(base_model, "layers", None)
    if layers is not None:
        return layers

    transformer = getattr(model, "transformer", None)
    layers = getattr(transformer, "h", None)
    if layers is not None:
        return layers

    raise TypeError("Unsupported model: decoder layers were not found.")


def _skip_block_output(_module: nn.Module, inputs: tuple[Any, ...], output: Any) -> Any:
    """Replace one decoder block output with its residual-stream input."""
    if not inputs or not isinstance(inputs[0], Tensor):
        raise TypeError(
            "A decoder block must receive hidden states as its first input."
        )
    hidden_states = inputs[0]
    if isinstance(output, tuple):
        return (hidden_states, *output[1:])
    return hidden_states


@contextmanager
def whole_block_skip(model: nn.Module, layer_indices: Sequence[int]) -> Iterator[None]:
    """Temporarily skip whole Transformer blocks and always restore hooks."""
    layers = locate_decoder_layers(model)
    normalized = tuple(int(index) for index in layer_indices)
    if len(normalized) != len(set(normalized)):
        raise ValueError("Duplicate layer indices are not a valid static edit.")
    if any(index < 0 or index >= len(layers) for index in normalized):
        raise IndexError("A requested layer index is outside the decoder.")

    handles = [
        layers[index].register_forward_hook(_skip_block_output) for index in normalized
    ]
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def select_evenly_spaced_layers(
    block_count: int, edit_count: int, edge_exclusion: int
) -> list[int]:
    """Select distinct edit locations spanning early, middle, and late depth."""
    first = edge_exclusion
    last = block_count - edge_exclusion - 1
    available = last - first + 1
    if edit_count < 2:
        raise ValueError("At least two edits are required for pair testing.")
    if available < edit_count:
        raise ValueError("The requested edit count exceeds eligible decoder layers.")

    positions = [
        round(first + index * (last - first) / (edit_count - 1))
        for index in range(edit_count)
    ]
    if len(set(positions)) != edit_count:
        raise ValueError("Even spacing produced duplicate layer locations.")
    return positions


def sample_layer_pairs(
    layer_indices: Sequence[int], pair_count: int, seed: int
) -> list[tuple[int, int]]:
    """Deterministically sample pairs while covering multiple layer distances."""
    candidates = list(itertools.combinations(sorted(layer_indices), 2))
    if pair_count < 1 or pair_count > len(candidates):
        raise ValueError("pair_count must be within the available unique pairs.")
    if pair_count == len(candidates):
        return candidates

    distances: dict[int, list[tuple[int, int]]] = {}
    for pair in candidates:
        distances.setdefault(pair[1] - pair[0], []).append(pair)
    rng = random.Random(seed)
    for bucket in distances.values():
        rng.shuffle(bucket)

    selected: list[tuple[int, int]] = []
    ordered_distances = sorted(distances)
    while len(selected) < pair_count:
        made_progress = False
        for distance in ordered_distances:
            bucket = distances[distance]
            if bucket and len(selected) < pair_count:
                selected.append(bucket.pop())
                made_progress = True
        if not made_progress:
            break
    return sorted(selected)


def prepare_sequences(
    text: str,
    tokenizer: Any,
    sequence_length: int,
    sequence_count: int,
    seed: int,
) -> Tensor:
    """Tokenize text and deterministically select fixed non-overlapping sequences."""
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
    return chunks[indices].contiguous()


@torch.inference_mode()
def evaluate_next_token_metrics(
    model: nn.Module,
    sequences: Tensor,
    batch_size: int,
    device: torch.device,
) -> NextTokenMetrics:
    """Compute token-weighted next-token NLL and perplexity."""
    total_loss = 0.0
    total_tokens = 0
    for start in range(0, sequences.shape[0], batch_size):
        batch = sequences[start : start + batch_size].to(device)
        logits = model(input_ids=batch, use_cache=False).logits[:, :-1, :].float()
        labels = batch[:, 1:]
        loss_sum = functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            reduction="sum",
        )
        total_loss += float(loss_sum)
        total_tokens += labels.numel()
        del batch, logits, labels, loss_sum
    nll = total_loss / total_tokens
    return NextTokenMetrics(
        nll=nll,
        perplexity=math.exp(min(nll, 80.0)),
        token_count=total_tokens,
    )


@torch.inference_mode()
def max_repeated_logit_difference(
    model: nn.Module, input_ids: Tensor, device: torch.device
) -> float:
    """Return the maximum logit difference between two no-edit evaluations."""
    batch = input_ids.to(device)
    first = model(input_ids=batch, use_cache=False).logits
    second = model(input_ids=batch, use_cache=False).logits
    difference = float((first - second).abs().max())
    del batch, first, second
    return difference


def clear_device_cache(device: torch.device) -> None:
    """Release transient CUDA cache without modifying model state."""
    if device.type == "cuda":
        torch.cuda.empty_cache()


def finite_metrics(metrics: NextTokenMetrics) -> bool:
    """Return whether all scalar metrics are finite and well-defined."""
    return (
        math.isfinite(metrics.nll)
        and math.isfinite(metrics.perplexity)
        and metrics.token_count > 0
    )
